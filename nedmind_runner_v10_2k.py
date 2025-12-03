#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NeD-Mind v10.2K runner - orchestrates Core, Temporal, Reflexive and Scenario
Corrections v10.2K:
- Added centralized log_topology() method
- Fixed reflexive buffer clamp logic
- Integrated scenario engine logging
"""
from __future__ import annotations
import argparse
import json
import logging
import yaml
from pathlib import Path
from typing import Dict, Any, Optional, List
import numpy as np
from tqdm import tqdm
from scipy import stats
from scipy import sparse
from utils_v10_2k import setup_logging, safe_scalar, NumpyEncoder
from nedmind_core_v10_2k import CoreConfig, build_topology, Concept, compute_coherence, compute_coherence_gradient, compute_delta_vectorized, sparse_coactivity_update, density_correct
from reflexive_monitor_v10_2k import ReflexiveConfig, ReflexiveMonitor
from temporal_engine_v10_2k import TemporalEngine
from scenario_engine_v10_2k import ScenarioEngine
from utils_v10_2k import save_checkpoint, load_checkpoint

logger = logging.getLogger("NeD-Mind.runner")

class SimulationPipeline:
    def __init__(self, cfg: CoreConfig, reflexive_cfg: Optional[ReflexiveConfig]=None, 
                 temporal_cfg: Optional[dict]=None, scenario_events: Optional[List[Dict]]=None):
        self.cfg = cfg
        self.rng = np.random.default_rng(cfg.SEED)
        self.reflexive_cfg = reflexive_cfg or ReflexiveConfig()
        self.temporal_cfg = temporal_cfg or {}
        self.evaluator_ctx = {}
        # state
        self.W = None
        self.concepts = []
        self.meta_mgr = None
        self.reflexive = ReflexiveMonitor(self.reflexive_cfg, seed=cfg.SEED+123)
        self.temporal = TemporalEngine(self.temporal_cfg)
        self.scenario = ScenarioEngine(scenario_events)
        self.scenario_temps: Dict[str, Any] = {}
        # buffers
        self.A_buffer = np.empty((cfg.N, cfg.C), dtype=np.float32)
        self.influence_buffer = np.zeros(cfg.N, dtype=np.float32)
        self.grad_buffer = np.zeros(cfg.N, dtype=np.float32)
        # traces
        n_trace = max(1, int(cfg.STEPS * cfg.TRACE_SAMPLING_RATE))
        self.trace_buffer = np.full((cfg.N, n_trace, cfg.C), np.nan, dtype=np.float32)
        self.trace_idx = 0
        self.step = 0
        self.metrics = []

    def initialize_network(self):
        logger.info("Initializing network...")
        k_opt = max(4, min(self.cfg.N - 1, int(np.sqrt(self.cfg.N))))
        self.W = build_topology(self.cfg.N, k_opt, self.cfg.TOPO_REWIRE_PROB, self.cfg.SEED, self.cfg.BASE_DENSITY)
        self.concepts = [Concept(i, self.cfg.C, self.cfg.TAU_MEM, self.cfg.H_CLIP, self.cfg.SEED + i) for i in range(self.cfg.N)]
        logger.info("Network initialized: %d synapses", self.W.nnz)
        return self

    def current_noise_amp(self):
        if 'NOISE_AMP' in self.scenario_temps:
            v, expiry = self.scenario_temps['NOISE_AMP']
            if expiry is None or expiry > self.step:
                return float(v)
            else:
                del self.scenario_temps['NOISE_AMP']
        return float(self.cfg.NOISE_AMP)

    def log_topology(self, step: int, phase: str = "ROUTINE"):
        """Log structuréal sécurisé avec calcul lazy"""
        if not self.cfg.ENABLE_STRUCTURE_LOG:
            return
            
        if step % self.cfg.STRUCTURE_LOG_INTERVAL != 0:
            return
            
        try:
            degrees = np.diff(self.W.indptr)
            if len(degrees) == 0:
                return
                
            hub_cutoff = np.percentile(degrees, 90)
            hub_indices = np.where(degrees >= hub_cutoff)[0]
            
            unique_deg = np.unique(degrees)
            skew = stats.skew(degrees) if len(unique_deg) > 2 else 0.0
            
            logger.info(
                f"\n{'='*40}\n"
                f"TOPOLOGY {phase} | Step {step:4d}\n"
                f"{'='*40}\n"
                f"  Degré max:        {int(degrees.max()):3d}\n"
                f"  Degré moyen:      {degrees.mean():5.1f}\n"
                f"  Hubs (top 10%):   {len(hub_indices):2d}\n"
                f"  Degré moyen hubs: {degrees[hubs_indices].mean():5.1f}\n"
                f"  Skewness:         {skew:5.2f}\n"
                f"  Densité:          {self.W.nnz/(self.cfg.N**2):.4f}\n"
                f"{'='*40}\n"
            )
        except Exception as e:
            logger.debug(f"Topology logging failed: {e}")

    def step_once(self) -> Optional[Dict[str,Any]]:
        try:
            if self.scenario:
                self.scenario.apply(self.step, self)

            np.stack([c.A for c in self.concepts], out=self.A_buffer)

            coherence_val = compute_coherence(self.A_buffer, self.cfg.GLOBAL_GOAL)
            A_mean = self.A_buffer.mean(axis=0)
            variance = float(self.A_buffer.var())

            np.dot(self.W, self.A_buffer.mean(axis=1), out=self.influence_buffer)

            # Log structurel centralisé
            self.log_topology(self.step, phase="ROUTINE")

            if (self.step % 10) == 0:
                self.grad_buffer[:] = compute_coherence_gradient(
                    self.A_buffer, self.cfg.GLOBAL_GOAL, 
                    epsilon=0.02, 
                    sample_ratio=0.2,
                    rng=self.rng
                )
            else:
                self.grad_buffer.fill(0.0)

            noise_amp = self.current_noise_amp()
            eta = safe_scalar(self.cfg.ETA)

            reflexive_boost = 1.0 + self.reflexive.get_feedback()

            delta_all = compute_delta_vectorized(self.A_buffer, self.cfg.GLOBAL_GOAL,
                                                 self.influence_buffer, eta, noise_amp,
                                                 self.grad_buffer, self.cfg.TAU_MEM,
                                                 reflexive_boost, self.rng)

            for i, c in enumerate(self.concepts):
                c.update(delta_all[i])

            if self.cfg.ENABLE_TOPO_PLASTICITY and (self.step % 3 == 0):
                activity_norms = np.linalg.norm(self.A_buffer, axis=1)
                self.W = sparse_coactivity_update(self.W, activity_norms,
                                                 eps=self.cfg.TOPO_EPS*(1.0+np.linalg.norm(self.grad_buffer)*0.1),
                                                 top_frac=self.cfg.TOPO_TOP_FRAC,
                                                 k_per_node=self.cfg.TOPO_K,
                                                 seed=self.cfg.SEED + self.step)
                self.W = density_correct(self.W, self.cfg.BASE_DENSITY, self.cfg.SEED + self.step)

            R_t = self.reflexive.observe({'speed':1.0,'stability':1.0,'pull':1.0,'reflexivity':0.5}, 
                                        reward=coherence_val, step=self.step, 
                                        coherence=coherence_val, variance=variance)
            self.temporal.tick()
            self.temporal.push_episode(A_mean)
            self.temporal.record_valence(coherence_val)

            if (self.step % max(1, int(1.0/self.cfg.TRACE_SAMPLING_RATE))) == 0 and self.trace_idx < self.trace_buffer.shape[1]:
                self.trace_buffer[:, self.trace_idx] = self.A_buffer
                self.trace_idx += 1

            metric = {
                'step': int(self.step),
                'coherence': float(coherence_val),
                'R_t': float(R_t),
                'density': float(self.W.nnz / (self.cfg.N*self.cfg.N)),
                'spectral_radius_est': float(np.max(np.abs(np.linalg.eigvals(self.W.toarray()))) if self.W.shape[0] < 200 else 0.0)
            }
            self.metrics.append(metric)
            self.step += 1
            return metric
        except Exception:
            logger.exception("Error during step %s", self.step)
            return None

    def run(self, outdir: Path, quick_test: bool = False):
        outdir = Path(outdir)
        outdir.mkdir(parents=True, exist_ok=True)
        setup_logging(outdir)
        steps = min(100, self.cfg.STEPS) if quick_test else self.cfg.STEPS
        try:
            with tqdm(total=steps, desc="NeD-Mind v10.2K", ncols=120) as pbar:
                for _ in range(steps):
                    metric = self.step_once()
                    if metric is not None:
                        pbar.set_postfix({'coh': f"{metric['coherence']:.3f}", 'R': f"{metric['R_t']:.2f}"})
                    pbar.update(1)
        except KeyboardInterrupt:
            logger.warning("Interrupted by user")
        finally:
            np.savez_compressed(outdir/"traces.npz", A=self.trace_buffer[:,:self.trace_idx], steps=np.arange(self.step))
            sparse.save_npz(outdir/"final_W.npz", self.W)
            with open(outdir/"summary.json","w") as f:
                json.dump(self.summary(), f, indent=2, cls=NumpyEncoder)
        return self.summary()

    def summary(self):
        return {
            "success": True,
            "steps_run": int(self.step),
            "final_metrics": (self.metrics[-1] if self.metrics else {}),
            "reflexive": self.reflexive.validate(),
            "temporal": {
                "clock": int(self.temporal.clock),
                "valence_avg": float(self.temporal.valence_moving_average())
            }
        }

def load_yaml_config(path: Path):
    with open(path) as f:
        cfg = yaml.safe_load(f)
    if 'CORE' in cfg:
        core_cfg = cfg['CORE']
    else:
        core_keys = {k.upper():v for k,v in cfg.items() if k.upper() in CoreConfig.__fields__}
        core_cfg = {k:v for k,v in cfg.items() if k.upper() in CoreConfig.__fields__}
        if not core_cfg:
            core_cfg = {k:v for k,v in cfg.items() if k in CoreConfig.__fields__}
    reflexive_cfg_data = cfg.get('REFLEXIVE', {})
    scenario_events = cfg.get('SCENARIO', cfg.get('SCENARIO_EVENTS', []))
    temporal_cfg = cfg.get('TEMPORAL', {})
    return core_cfg, reflexive_cfg_data, temporal_cfg, scenario_events

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config","-c", required=True)
    p.add_argument("--outdir","-o", default=None)
    p.add_argument("--quick", action="store_true")
    args = p.parse_args()
    core_cfg, reflexive_cfg_data, temporal_cfg, scenario_events = load_yaml_config(Path(args.config))
    cfg = CoreConfig(**core_cfg)
    reflexive_cfg = ReflexiveConfig(**reflexive_cfg_data) if reflexive_cfg_data else None
    pipeline = SimulationPipeline(cfg, reflexive_cfg, temporal_cfg, scenario_events).initialize_network()
    outdir = Path(args.outdir) if args.outdir else Path(cfg.OUTDIR)
    summary = pipeline.run(outdir, quick_test=args.quick)
    print(json.dumps(summary, indent=2, cls=NumpyEncoder))

if __name__ == "__main__":
    main()
