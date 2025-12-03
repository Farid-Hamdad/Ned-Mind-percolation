#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Temporal engine v10.2K - unchanged from v10
"""
from __future__ import annotations
import numpy as np
import logging
from typing import Optional

logger = logging.getLogger("NeD-Mind.temporal")

class EpisodicBuffer:
    def __init__(self, capacity: int = 1000, vector_dim: int = 8, decay: float = 0.01, seed: int = 42):
        self.capacity = int(capacity)
        self.vector_dim = int(vector_dim)
        self.decay = float(decay)
        self.rng = np.random.default_rng(seed)
        self.buffer = np.zeros((self.capacity, self.vector_dim), dtype=np.float32)
        self.times = -np.ones(self.capacity, dtype=np.int32)
        self.ptr = 0
        self.size = 0

    def push(self, vec: np.ndarray, t: int):
        vec = np.asarray(vec).astype(np.float32)
        if vec.size != self.vector_dim:
            tmp = np.zeros(self.vector_dim, dtype=np.float32)
            tmp[:min(vec.size, self.vector_dim)] = vec[:self.vector_dim]
            vec = tmp
        self.buffer[self.ptr] = vec
        self.times[self.ptr] = int(t)
        self.ptr = (self.ptr + 1) % self.capacity
        if self.size < self.capacity:
            self.size += 1

    def sample_recent(self, n: int = 10):
        if self.size == 0:
            return np.zeros((0, self.vector_dim), dtype=np.float32)
        idx = (self.ptr - np.arange(1, min(self.size, n)+1)) % self.capacity
        return self.buffer[idx]

    def decay_all(self):
        self.buffer *= (1.0 - self.decay)
        self.buffer[np.abs(self.buffer) < 1e-8] = 0.0

class TemporalEngine:
    def __init__(self, cfg: dict):
        self.clock = 0
        self.circadian = cfg.get("circadian", False)
        self.scale = cfg.get("scale", 1.0)
        self.episodic = EpisodicBuffer(capacity=cfg.get("episodic_capacity", 1000),
                                       vector_dim=cfg.get("vector_dim", 8),
                                       decay=cfg.get("episodic_decay", 0.01),
                                       seed=cfg.get("seed", 42))
        self.valence_history = []

    def tick(self):
        self.clock += 1
        if self.circadian:
            self.scale = 1.0 + 0.1 * np.sin(self.clock / 24.0 * 2 * np.pi)
        self.episodic.decay_all()

    def push_episode(self, vec: np.ndarray):
        self.episodic.push(vec, self.clock)

    def recent_mean(self, n: int = 10):
        arr = self.episodic.sample_recent(n)
        if arr.size == 0:
            return np.zeros(self.episodic.vector_dim, dtype=np.float32)
        return arr.mean(axis=0)

    def record_valence(self, v: float):
        self.valence_history.append(float(v))
        if len(self.valence_history) > 1000:
            self.valence_history.pop(0)

    def valence_moving_average(self, window: int = 20):
        if not self.valence_history:
            return 0.0
        arr = np.asarray(self.valence_history[-window:], dtype=np.float32)
        return float(arr.mean())
