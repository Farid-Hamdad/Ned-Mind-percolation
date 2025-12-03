#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Utilities for NeD-Mind v10.2K - unchanged from v10.1
"""
from __future__ import annotations
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict
import numpy as np
from pydantic import BaseModel

_logger = logging.getLogger("NeD-Mind.utils")

def setup_logging(outdir: Path, level: int = logging.INFO) -> logging.Logger:
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    log_path = outdir / "nedmind_v10_2k.log"

    file_handler = logging.FileHandler(log_path, mode="a", encoding="utf-8")
    console_handler = logging.StreamHandler(sys.stdout)

    fmt = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S"
    formatter = logging.Formatter(fmt, datefmt=datefmt)
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    root = logging.getLogger()
    root.setLevel(level)
    if not root.handlers:
        root.addHandler(file_handler)
        root.addHandler(console_handler)
    else:
        existing = [type(h) for h in root.handlers]
        if logging.FileHandler not in existing:
            root.addHandler(file_handler)
        if logging.StreamHandler not in existing:
            root.addHandler(console_handler)

    _logger.info("Logging initialized at %s", log_path)
    return _logger

def safe_scalar(val: Any) -> float:
    if isinstance(val, (float, int)):
        return float(val)
    if isinstance(val, np.ndarray):
        if val.size == 1:
            return float(val.item())
        return float(np.mean(val))
    try:
        return float(val)
    except Exception:
        return 0.0

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        return super().default(obj)

_SAFE_FUNCS = {
    'abs': abs, 'min': min, 'max': max, 'sum': sum, 'len': len,
    'float': float, 'int': int, 'np': np
}

def safe_eval(expr: str, ctx: Dict[str, Any] = None, default: Any = None):
    try:
        safe_ctx = {}
        if ctx:
            for k, v in ctx.items():
                if isinstance(v, (int, float, bool, np.ndarray, list, tuple)):
                    safe_ctx[k] = v
        safe_ctx.update(_SAFE_FUNCS)
        return eval(expr, {"__builtins__": {}}, safe_ctx)
    except Exception:
        return default

def save_checkpoint(path: Path, data: Dict[str, Any]):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp.npz")
    np.savez_compressed(tmp, **{k: (v if isinstance(v, np.ndarray) else np.asarray(v)) for k,v in data.items()})
    tmp.replace(path)
    _logger.info("Checkpoint saved: %s", path)

def load_checkpoint(path: Path):
    arr = np.load(path, allow_pickle=False)
    return dict(arr)

class ConfigModel(BaseModel):
    class Config:
        arbitrary_types_allowed = True
