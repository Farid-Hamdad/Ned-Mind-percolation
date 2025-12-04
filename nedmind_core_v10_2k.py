#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NeD-Mind Core v10.2K - RESTORED with physical coherence
"""

from __future__ import annotations
import numpy as np
from scipy import sparse
import logging
from typing import Optional
from pydantic import BaseModel, Field, validator
logger = logging.getLogger("NeD-Mind.core")

# ========== CORE CONFIG ==========
class CoreConfig(BaseModel):
    N: int = Field(128, gt=0, le=20000)
    C: int = Field(8, gt=0, le=512)
    STEPS: int = Field(1000, gt=0, le=1_000_000)
    SEED: int = Field(42, ge=0)
    META_MOMENTUM: float = Field(0.92, ge=0.0, le=0.999)
    META_LEARNING_RATE: float = Field(0.03, gt=0.0, le=1.0)
    BASE_DENSITY: float = Field(0.08, ge=0.0001, le=0.5)
    TOPO_REWIRE_PROB: float = Field(0.15, ge=0.0, le=1.0)
    TOPO_EPS: float = Field(0.015, gt=0.0, le=1.0)
    TOPO_K: int = Field(6, gt=0, le=100)
    TOPO_TOP_FRAC: float = Field(0.15, gt=0.0, le=1.0)
    TAU_MEM: float = Field(0.75, gt=0.0, le=100.0)
    ETA: float = Field(0.01, gt=0.0, le=1.0)
    H_CLIP: float = Field(1.5, gt=0.0, le=100.0)
    NOISE_AMP: float = Field(0.02, ge=0.0, le=1.0)
    TRACE_SAMPLING_RATE: float = Field(0.1, ge=0.0, le=1.0)
    ENABLE_TOPO_PLASTICITY: bool = Field(True)
    ENABLE_STRUCTURE_LOG: bool = Field(False)
    STRUCTURE_LOG_INTERVAL: int = Field(50, gt=0, le=10000)
    GLOBAL_GOAL: Optional[np.ndarray] = Field(default_factory=lambda: np.zeros(8, dtype=np.float32))
    OUTDIR: str = Field("./output")

    class Config:
        arbitrary_types_allowed = True
    
    @validator('GLOBAL_GOAL', pre=True)
    def parse_goal(cls, v):
        if v is None:
            return np.zeros(8, dtype=np.float32)
        if isinstance(v, str):
            arr = [float(x) for x in v.split(',')]
            return np.asarray(arr, dtype=np.float32)
        return np.asarray(v, dtype=np.float32)

# ========== TOPOLOGY ==========
def small_world_vectorized(N: int, k: int, p: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    W = np.zeros((N, N), dtype=np.float32)
    for offset in range(1, k // 2 + 1):
        W += np.eye(N, k=offset, dtype=np.float32)
        W += np.eye(N, k=-offset, dtype=np.float32)
    rows, cols = np.where(W > 0)
    rewire_mask = rng.random(len(rows)) < p
    if rewire_mask.any():
        new_cols = rng.integers(0, N, size=rewire_mask.sum())
        self_loop_mask = new_cols == rows[rewire_mask]
        new_cols[self_loop_mask] = (new_cols[self_loop_mask] + 1) % N
        W[rows[rewire_mask], cols[rewire_mask]] = 0.0
        W[rows[rewire_mask], new_cols] = 1.0
    row_sums = W.sum(axis=1)
    row_sums[row_sums == 0] = 1.0
    return (W.T / row_sums).T.astype(np.float32)

def build_topology(N: int, k: int, p: float, seed: int, target_density: float) -> sparse.csr_matrix:
    W_dense = small_world_vectorized(N, k, p, seed)
    W = sparse.csr_matrix(W_dense)
    current_density = W.nnz / (N * N)
    if abs(current_density - target_density) > 0.01:
        W = density_correct(W, target_density, seed)
    return W

def density_correct(W: sparse.csr_matrix, target: float, seed: int) -> sparse.csr_matrix:
    rng = np.random.default_rng(seed)
    N = W.shape[0]
    current = W.nnz / (N * N)
    W_lil = W.tolil()
    if current > target:
        to_remove = int((current - target) * N * N)
        rows, cols = W.nonzero()
        if len(rows) > 0 and to_remove > 0:
            remove_idx = rng.choice(len(rows), min(to_remove, len(rows)), replace=False)
            for idx in remove_idx:
                W_lil[rows[idx], cols[idx]] = 0.0
    else:
        to_add = int((target - current) * N * N)
        for _ in range(to_add):
            i, j = rng.integers(0, N, 2)
            if i != j:
                W_lil[i, j] = 1.0
    return W_lil.tocsr()

# ========== CONCEPT ==========
class Concept:
    __slots__ = ('idx', 'C', 'A', 'tau_mem', 'h_clip', 'rng')
    def __init__(self, idx: int, C: int, tau_mem: float, h_clip: float, seed: int):
        self.idx = idx
        self.C = C
        self.tau_mem = tau_mem
        self.h_clip = h_clip
        self.rng = np.random.default_rng(seed)
        self.A = self.rng.uniform(-0.05, 0.05, size=C).astype(np.float32)
    
    def update(self, delta: np.ndarray):
        self.A += delta
        np.clip(self.A, -self.h_clip, self.h_clip, out=self.A)

# ========== DYNAMICS ==========
def compute_delta_vectorized(A_all: np.ndarray, goal: np.ndarray,
                             influence: np.ndarray, eta: float, noise: float,
                             grad: np.ndarray, tau_mem: float,
                             reflexive_boost: float, rng: np.random.Generator) -> np.ndarray:
    recall = A_all * (1.0 - tau_mem)
    external = influence[:, None] * (1.0 - tau_mem) * 0.15
    goal_pull = (goal - A_all) * 0.08 * reflexive_boost
    coherence_drive = grad[:, None] * 0.02
    noise_arr = rng.normal(0, noise, size=A_all.shape).astype(np.float32)
    delta = eta * (recall + external + goal_pull + coherence_drive + noise_arr)
    return delta

def compute_cohesion_gradient(A_all: np.ndarray, goal: np.ndarray, 
                               epsilon: float = 0.02, sample_ratio: float = 0.2,
                               rng: Optional[np.random.Generator] = None) -> np.ndarray:
    N = A_all.shape[0]
    grad = np.zeros(N, dtype=np.float32)
    if rng is None:
        rng = np.random.default_rng(42)
    
    sample_size = max(1, int(N * sample_ratio))
    indices = rng.choice(N, size=min(sample_size, N), replace=False)
    
    for i in indices:
        A_plus = A_all.copy()
        A_plus[i] += epsilon
        coh_plus = compute_coherence(A_plus, goal)
        A_minus = A_all.copy()
        A_minus[i] -= epsilon
        coh_minus = compute_coherence(A_minus, goal)
        grad[i] = (coh_plus - coh_minus) / (2 * epsilon)
    return grad

# ========== PHYSICAL COHERENCE (CORRECTED) ==========
def compute_coherence(A_all: np.ndarray, goal: np.ndarray) -> float:
    """
    PHYSICAL COHERENCE: energy projected on goal (no normalization of A_all)
    Returns 0.0 when network activity collapses (mort physique).
    Returns 1.0 when fully aligned with full energy.
    """
    # Safety: zero goal
    if np.linalg.norm(goal) < 1e-8:
        logger.warning("Goal vector is zero - returning 0.0")
        return 0.0
    
    # Safety: NaN/Inf
    if not np.isfinite(A_all).all():
        logger.warning("NaN/Inf in activity - returning 0.0")
        return 0.0
    
    # METRIC 1: Total energy (average L2 norm of concepts)
    # Si A_all → 0, alors total_energy → 0 = mort physique
    total_energy = np.mean(np.linalg.norm(A_all, axis=1))
    
    # METRIC 2: Raw alignment (dot product, PAS DE NORMALISATION)
    # On garde l'amplitude : si A_all → 0, alors alignment → 0
    goal_unit = goal / np.linalg.norm(goal)
    alignment = np.mean(np.abs(np.dot(A_all, goal_unit)))
    
    # PHYSICAL COHERENCE = ENERGIE × ALIGNEMENT
    # Si l'énergie s'effondre → coh = 0
    # Si l'alignement s'effondre → coh = 0
    coherence = float(total_energy * alignment)
    
    # Clip to [0, 1]
    return float(np.clip(coherence, 0.0, 1.0))
# ========== TOPOLOGY UPDATE ==========
def sparse_coactivity_update(W: sparse.csr_matrix, activity: np.ndarray, eps: float, 
                            top_frac: float = 0.15, k_per_node: int = 6, 
                            seed: int = 42) -> sparse.csr_matrix:
    rng = np.random.default_rng(seed)
    N = W.shape[0]
    W_lil = W.tolil()
    
    if activity.size != N:
        activity = np.asarray(activity).ravel()[:N]
    
    threshold = np.percentile(activity, 100 - top_frac*100)
    active_idx = np.where(activity >= threshold)[0]
    
    if len(active_idx) == 0:
        return W
    
    candidates = np.where(activity > np.percentile(activity, 70))[0]
    if len(candidates) == 0:
        return W
    
    boost_matrix = eps * np.outer(activity[active_idx], activity[candidates])
    
    for r_i, i in enumerate(active_idx):
        current_vals = np.array(W_lil[i, candidates].toarray().ravel(), dtype=np.float32)
        updated_vals = np.clip(current_vals + boost_matrix[r_i], 0.0, 1.0)
        weak_mask = updated_vals < 0.1
        if weak_mask.sum() > k_per_node:
            to_weaken = rng.choice(np.where(weak_mask)[0], k_per_node, replace=False)
            updated_vals[to_weaken] *= (1.0 - eps)
        W_lil[i, candidates] = updated_vals
    
    W_lil.setdiag(0)
    return W_lil.tocsr()

# ========== COHERENCE GRADIENT ==========
def compute_coherence_gradient(A_all: np.ndarray, goal: np.ndarray, 
                               epsilon: float = 0.02, sample_ratio: float = 0.2,
                               rng: Optional[np.random.Generator] = None) -> np.ndarray:
    N = A_all.shape[0]
    grad = np.zeros(N, dtype=np.float32)
    if rng is None:
        rng = np.random.default_rng(42)
    
    sample_size = max(1, int(N * sample_ratio))
    indices = rng.choice(N, size=min(sample_size, N), replace=False)
    
    for i in indices:
        A_plus = A_all.copy()
        A_plus[i] += epsilon
        coh_plus = compute_coherence(A_plus, goal)
        A_minus = A_all.copy()
        A_minus[i] -= epsilon
        coh_minus = compute_coherence(A_minus, goal)
        grad[i] = (coh_plus - coh_minus) / (2 * epsilon)
    return grad

# ========== CONCEPTS & DYNAMICS ==========
class Concept:
    __slots__ = ('idx', 'C', 'A', 'tau_mem', 'h_clip', 'rng')
    def __init__(self, idx: int, C: int, tau_mem: float, h_clip: float, seed: int):
        self.idx = idx
        self.C = C
        self.tau_mem = tau_mem
        self.h_clip = h_clip
        self.rng = np.random.default_rng(seed)
        self.A = self.rng.uniform(-0.05, 0.05, size=C).astype(np.float32)
    
    def update(self, delta: np.ndarray):
        self.A += delta
        np.clip(self.A, -self.h_clip, self.h_clip, out=self.A)

# ========== END OF FILE ==========
