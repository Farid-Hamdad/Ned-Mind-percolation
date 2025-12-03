#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Scenario engine v10.2K
Corrections v10.2K:
- Refactored ablate_hubs into dedicated method with integrated logging
- Added apply_density_correct parameter for experimental control
- Enhanced parameter validation
"""
from __future__ import annotations
import numpy as np
import logging
from typing import List, Dict, Any, Optional
from scipy import sparse
from scipy import stats

logger = logging.getLogger("NeD-Mind.scenario")

class ScenarioEngine:
    def __init__(self, events: Optional[List[Dict[str,Any]]] = None):
        self.events = sorted(events or [], key=lambda e: int(e.get('step',0)))
        self.temporaries = {}

    def events_at(self, step:int) -> List[Dict[str, Any]]:
        return [e for e in self.events if int(e.get('step',0)) == int(step)]

    def apply(self, step:int, pipeline) -> None:
        evs = self.events_at(step)
        for e in evs:
            try:
                self._apply_event(e, pipeline, step)
            except Exception:
                logger.exception("Scenario event failed at step %s: %s", step, e)

        expired = [k for k,(v,expiry) in self.temporaries.items() if expiry is not None and expiry <= step]
        for k in expired:
            del self.temporaries[k]

    def _blend_injection(self, pipeline, vec, mode='add', alpha=0.3):
        vec = np.asarray(vec, dtype=np.float32)
        if vec.size != pipeline.cfg.C:
            tmp = np.zeros(pipeline.cfg.C, dtype=np.float32)
            tmp[:min(vec.size, pipeline.cfg.C)] = vec[:pipeline.cfg.C]
            vec = tmp
        if mode == 'blend':
            for c in pipeline.concepts:
                c.A = (1.0-alpha)*c.A + alpha*vec
        else:
            for c in pipeline.concepts:
                c.A += vec * 0.5

    def _log_topology_phase(self, pipeline, step: int, phase: str, W: sparse.spmatrix):
        """Log standardisé pour les événements topologiques"""
        if not pipeline.cfg.ENABLE_STRUCTURE_LOG:
            return
            
        try:
            degrees = np.diff(W.indptr)
            if len(degrees) == 0:
                return
                
            hub_cutoff = np.percentile(degrees, 90)
            hub_indices = np.where(degrees >= hub_cutoff)[0]
            skew = stats.skew(degrees) if len(np.unique(degrees)) > 2 else 0.0
            
            logger.info(
                f"\n{'='*40}\n"
                f"TOPOLOGY {phase} | Step {step}\n"
                f"{'='*40}\n"
                f"  Degré max:        {int(degrees.max()):3d}\n"
                f"  Degré moyen:      {degrees.mean():5.1f}\n"
                f"  Nombre de hubs:   {len(hub_indices):2d}\n"
                f"  Degré moyen hubs: {degrees[hub_indices].mean():5.1f}\n"
                f"  Skewness:         {skew:5.2f}\n"
                f"  Densité:          {W.nnz/(pipeline.cfg.N**2):.4f}\n"
                f"{'='*40}\n"
            )
        except Exception as e:
            logger.debug(f"Topology phase logging failed: {e}")

    def _apply_ablate_hubs(self, e: Dict, pipeline, step: int):
        """Opération d'ablation avec logging intégré"""
        
        # [v10.2K] Validation des paramètres
        fraction = float(e.get('ablation_fraction', 0.5))
        threshold = float(e.get('hub_threshold', 0.9))
        
        fraction = np.clip(fraction, 0.01, 1.0)
        threshold = np.clip(threshold, 0.0, 1.0)
        
        # Phase 1 : Mesure avant
        self._log_topology_phase(pipeline, step, "PRE-ABLATION", pipeline.W)
        
        # Phase 2 : Identification des hubs
        degrees = np.diff(pipeline.W.indptr)
        hub_cutoff = np.quantile(degrees, threshold)
        hub_indices = np.where(degrees >= hub_cutoff)[0]
        
        logger.warning(f"⚠️  ABLATION: {len(hub_indices)} hubs (threshold={threshold:.2f})")
        
        # Phase 3 : Ablation
        W_lil = pipeline.W.tolil()
        connections_removed = 0
        
        for hub in hub_indices:
            row_data = W_lil[hub, :].toarray().ravel()
            connections = np.where(row_data > 0)[0]
            
            if len(connections) > 0:
                n_remove = max(1, int(len(connections) * fraction))
                to_remove = np.random.choice(connections, n_remove, replace=False)
                for tgt in to_remove:
                    W_lil[hub, tgt] = 0.0
                connections_removed += n_remove
        
        W_after = W_lil.tocsr()
        
        # Phase 4 : Mesure après ablation
        self._log_topology_phase(pipeline, step, "POST-ABLATION", W_after)
        
        # Phase 5 : Recorrection optionnelle
        if e.get('apply_density_correct', True):
            from nedmind_core_v10_2k import density_correct
            pipeline.W = density_correct(W_after, float(pipeline.cfg.BASE_DENSITY), 
                                        int(pipeline.cfg.SEED) + step)
            logger.info("  → Recorrection appliquée")
            self._log_topology_phase(pipeline, step, "POST-CORRECTION", pipeline.W)
        else:
            # [v10.2K-FIX] Empêche toute future correction
            pipeline.cfg.BASE_DENSITY = float(W_after.nnz / (pipeline.cfg.N**2))
            pipeline.W = W_after
            logger.critical(f"  → DENSITÉ FIXÉE à {pipeline.cfg.BASE_DENSITY:.4f} - PAS DE RÉGÉNÉRATION")

    def _apply_event(self, e:Dict[str,Any], pipeline, step:int):
        if 'inject' in e:
            mode = e.get('mode','add')
            alpha = float(e.get('alpha', 0.3))
            self._blend_injection(pipeline, e['inject'], mode=mode, alpha=alpha)
            logger.info("SCENARIO: inject at step %s mode=%s", step, mode)
        
        if 'goal_shift' in e:
            goal = np.asarray(e['goal_shift'], dtype=np.float32)
            if e.get('blend', False):
                current = pipeline.cfg.GLOBAL_GOAL
                beta = float(e.get('beta', 0.5))
                pipeline.cfg.GLOBAL_GOAL = (1.0-beta)*current + beta*goal
            else:
                pipeline.cfg.GLOBAL_GOAL = goal
            logger.info("SCENARIO: goal_shift at step %s", step)
        
        if 'noise_pulse' in e:
            amp = float(e['noise_pulse'])
            dur = int(e.get('duration', 1))
            pipeline.scenario_temps['NOISE_AMP'] = (amp, step + dur)
            logger.info("SCENARIO: noise_pulse %s for %s steps", amp, dur)
        
        if 'topology_shock' in e:
            frac = float(e['topology_shock'])
            W = pipeline.W.tolil()
            N = W.shape[0]
            rng = pipeline.rng
            nchange = int(frac * N * N)
            for _ in range(nchange):
                i = int(rng.integers(0, N))
                j = int(rng.integers(0, N))
                if i != j:
                    W[i,j] = 1.0 if rng.random() > 0.5 else 0.0
            pipeline.W = W.tocsr()
            logger.info("SCENARIO: topology_shock frac=%s", frac)
        
        if 'param_override' in e:
            p = e['param_override']
            dur = int(e.get('duration',1))
            for k, v in p.items():
                pipeline.scenario_temps[k] = (v, step + dur)
            logger.info("SCENARIO: param_override %s", p)
        
        if e.get('ablate_hubs'):
            self._apply_ablate_hubs(e, pipeline, step)
