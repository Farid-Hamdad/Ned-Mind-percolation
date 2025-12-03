#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Reflexive monitor v10.2K - meta evaluation & safety controller
Corrections v10.2K:
- Fixed buffer clamp to prevent memory leak
- Corrected feedback normalization
- Added proper circular buffer logic
"""
from __future__ import annotations
import numpy as np
import logging
from typing import Dict, Any, Tuple
from pydantic import BaseModel, Field
from utils_v10_2k import safe_scalar

logger = logging.getLogger("NeD-Mind.reflexive")

class ReflexiveConfig(BaseModel):
    buffer_size: int = Field(200, gt=10, le=10000)
    safety_threshold: float = Field(2.0, gt=0.0, le=20.0)
    lead_window: int = Field(50, gt=5, le=2000)
    corr_threshold: float = Field(0.08, ge=0.0, le=1.0)
    feedback_strength: float = Field(0.15, ge=0.0, le=1.0)

class ReflexiveMonitor:
    def __init__(self, cfg: ReflexiveConfig, seed: int = 42):
        self.cfg = cfg
        self.rng = np.random.default_rng(seed)
        self.reward_buffer = np.zeros(cfg.buffer_size, dtype=np.float32)
        self.meta_buffer = np.zeros((cfg.buffer_size, 4), dtype=np.float32)
        self.coherence_buffer = np.zeros(cfg.buffer_size, dtype=np.float32)
        self.R_buffer = np.zeros(cfg.buffer_size, dtype=np.float32)
        self.step_buffer = np.zeros(cfg.buffer_size, dtype=np.int32)
        self.ptr = 0
        self.lead_times = []
        self.feedback_signal = 0.0
        self.full_cycles = 0  # [v10.2K] Compteur de cycles complets

    def observe(self, meta_state: Dict[str, float], reward: float, step: int, 
                coherence: float, variance: float) -> float:
        reward = safe_scalar(reward)
        coherence = safe_scalar(coherence)
        variance = safe_scalar(variance)
        speed = safe_scalar(meta_state.get('speed', 1.0))
        stability = safe_scalar(meta_state.get('stability', 1.0))
        pull = safe_scalar(meta_state.get('pull', 1.0))
        reflexivity = safe_scalar(meta_state.get('reflexivity', 0.5))
        margin = np.tanh(max(0.0, stability - self.cfg.safety_threshold) / 2.0)
        pred = 0.5
        try:
            recent = self.reward_buffer[:self.ptr] if self.ptr > 0 else np.array([])
            if recent.size >= 5:
                hist, _ = np.histogram(recent, bins=min(15, len(np.unique(recent))), density=True)
                hist = hist[hist > 0]
                if len(hist) > 0:
                    H = -np.sum(hist * np.log(hist + 1e-12))
                    pred = np.clip(1.0 - H / np.log(len(hist) + 1), 0.0, 1.0)
        except Exception:
            pred = 0.5
        explore = 0.5
        accel = 0.5
        if self.ptr > 5:
            recent_meta = self.meta_buffer[:self.ptr]
            if recent_meta.size:
                try:
                    meta_var = np.nanmean(np.var(recent_meta, axis=0))
                    explore = np.exp(-abs(meta_var - 0.1) * 5)
                except Exception:
                    pass
            recent_rewards = self.reward_buffer[:self.ptr]
            if recent_rewards.size > 2:
                try:
                    accel = np.exp(-np.mean(np.abs(np.diff(recent_rewards, n=2))) * 3)
                except Exception:
                    pass
        R = 0.35 * margin + 0.25 * pred + 0.2 * explore + 0.2 * accel
        R += reflexivity * 0.1 * (self.rng.uniform(-0.5, 0.5))
        R = float(np.clip(R, 0.0, 1.0))
        
        # [v10.2K] Buffer circulaire propre
        idx = self.ptr % self.cfg.buffer_size
        self.reward_buffer[idx] = reward
        self.meta_buffer[idx] = np.array([speed, stability, pull, reflexivity], dtype=np.float32)
        self.coherence_buffer[idx] = coherence
        self.R_buffer[idx] = R
        self.step_buffer[idx] = int(step)
        
        self.ptr += 1
        if self.ptr >= self.cfg.buffer_size:
            self.full_cycles += 1
            
        if self.ptr >= self.cfg.lead_window:
            self._update_lead_time_and_feedback()
        return R

    def _update_lead_time_and_feedback(self):
        try:
            window = self.cfg.lead_window
            # [v10.2K] Accès sécurisé avec wrap-around
            if self.ptr < window:
                recent_R = np.concatenate([self.R_buffer[-(window-self.ptr):], self.R_buffer[:self.ptr]])
                recent_coh = np.concatenate([self.coherence_buffer[-(window-self.ptr):], self.coherence_buffer[:self.ptr]])
            else:
                start = (self.ptr - window) % self.cfg.buffer_size
                if start + window <= self.cfg.buffer_size:
                    recent_R = self.R_buffer[start:start+window]
                    recent_coh = self.coherence_buffer[start:start+window]
                else:
                    part1 = window - (self.cfg.buffer_size - start)
                    recent_R = np.concatenate([self.R_buffer[start:], self.R_buffer[:part1]])
                    recent_coh = np.concatenate([self.coherence_buffer[start:], self.coherence_buffer[:part1]])
            
            if recent_R.std() < 1e-8 or recent_coh.std() < 1e-8:
                return
            Rn = (recent_R - recent_R.mean()) / (recent_R.std() + 1e-8)
            Cn = (recent_coh - recent_coh.mean()) / (recent_coh.std() + 1e-8)
            from scipy.signal import correlate
            corr = correlate(Rn, Cn, mode='full')
            lags = np.arange(-len(Rn)+1, len(Rn))
            lead_mask = lags > 0
            if not lead_mask.any():
                return
            best_idx = np.argmax(np.abs(corr[lead_mask]))
            best_lag = int(lags[lead_mask][best_idx])
            best_corr = float(corr[lead_mask][best_idx])
            if abs(best_corr) > self.cfg.corr_threshold:
                self.lead_times.append(best_lag)
                self.feedback_signal = float(np.clip(best_corr * self.cfg.feedback_strength / window, -0.3, 0.3))
        except Exception:
            logger.debug("lead_time calculation failed", exc_info=True)

    def get_feedback(self) -> float:
        s = self.feedback_signal
        self.feedback_signal = 0.0
        return float(s)

    def validate(self) -> Tuple[bool, Dict[str, Any]]:
        if len(self.lead_times) < 3:
            return False, {"reason": "insufficient_data", "n": len(self.lead_times)}
        try:
            arr = np.array(self.lead_times, dtype=np.int32)
            median = float(np.median(arr))
            significant = bool(np.mean(arr > 0) > 0.6)
            return significant, {
                "median_lead_time": median,
                "mean_lead_time": float(arr.mean()),
                "std_lead_time": float(arr.std()),
                "n_points": int(len(arr)),
                "significant": significant
            }
        except Exception as e:
            return False, {"reason": "computation_error", "error": str(e)}
