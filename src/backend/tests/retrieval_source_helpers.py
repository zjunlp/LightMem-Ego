#!/usr/bin/env python3
"""Read-only deterministic audit of active retrieval source.

Runs source AST definitions with NumPy, fake embeddings, and inert heavy-runtime
dependencies. All cache files and sample media live in TemporaryDirectory.
It never imports project modules, loads models, or invokes HTTP clients.
"""
from __future__ import annotations

import argparse
import ast
import copy
import hashlib
import io
import json
import logging
import math
import os
import pickle
import re
import sys
import tempfile
import threading
import time
import types
import zipfile
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
assert (ROOT / "src/em2mem/memory/EM2Memory.py").exists(), ROOT


class SilentProgress:
    def __init__(self, **kwargs):
        pass

    def update(self, value):
        pass

    def close(self):
        pass


class NoModels:
    def __init__(self, **kwargs):
        self.calls = []

    def generate(self, *args, **kwargs):
        raise AssertionError("The repro must not call any LLM")


class NeverOOM(Exception):
    pass


TORCH = types.SimpleNamespace(
    OutOfMemoryError=NeverOOM,
    cuda=types.SimpleNamespace(is_available=lambda: False, empty_cache=lambda: None),
)


def definitions(relative_path, overrides=None):
    """Compile unchanged active definitions, excluding imports of heavy clients."""
    path = ROOT / relative_path
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    tree.body = [ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)] + [
        node for node in tree.body if not isinstance(node, (ast.Import, ast.ImportFrom))
    ]
    ast.fix_missing_locations(tree)
    module_name = "audit_" + relative_path.replace("/", "_").replace(".", "_")
    module = types.ModuleType(module_name)
    sys.modules[module_name] = module
    module.__dict__.update(
        ast=ast, copy=copy, hashlib=hashlib, io=io, json=json, logging=logging,
        math=math, os=os, pickle=pickle, re=re, time=time, zipfile=zipfile,
        threading=threading, defaultdict=defaultdict, dataclass=dataclass,
        field=field, Path=Path, np=np, Image=Image, torch=TORCH,
        tqdm=SilentProgress, LLMModel=NoModels, PromptTemplateManager=NoModels,
    )
    if overrides:
        module.__dict__.update(overrides)
    exec(compile(tree, str(path), "exec"), module.__dict__)
    return module


QUERY_INDEX = definitions("src/em2mem/memory/query_index.py")
CACHE_IDENTITY = definitions("src/em2mem/embedding/cache_identity.py")
TIMESTAMP_UTILS = definitions("src/em2mem/memory/timestamp_utils.py")
UTIL = definitions("src/em2mem/memory/utils.py")
EPISODIC = definitions("src/em2mem/memory/episodic/EpisodicMemory_rag.py", {
    "embedding_cache_identity": CACHE_IDENTITY.embedding_cache_identity,
    "memory_timestamp_range": TIMESTAMP_UTILS.memory_timestamp_range,
})
VISUAL = definitions("src/em2mem/memory/visual/memory.py", {"query_index_property": QUERY_INDEX.query_index_property})
SEMANTIC = definitions("src/em2mem/memory/semantic/memory.py", {"query_index_property": QUERY_INDEX.query_index_property, "embedding_cache_identity": CACHE_IDENTITY.embedding_cache_identity})
UNIFIED = definitions("src/em2mem/memory/EM2Memory.py", {
    "EpisodicMemoryRAG": EPISODIC.EpisodicMemoryRAG,
    "SemanticMemory": SEMANTIC.SemanticMemory,
    "VisualMemory": VISUAL.VisualMemory,
    "QAResult": UTIL.QAResult,
    "RetrievedItem": UTIL.RetrievedItem,
    "transform_timestamp": UTIL.transform_timestamp,
})


class FakeEmbedding:
    def __init__(self, model_name="fake/A", dimension=2, rotate=False):
        self.text_model_name = model_name
        self.dimension = dimension
        self.rotate = rotate
        self.calls = []

    def encode_text(self, texts, **kwargs):
        if isinstance(texts, str):
            texts = [texts]
        self.calls.extend(texts)
        rows = []
        for text in texts:
            # Orthogonal documents and an exact-match query. Rotating the whole
            # space preserves all true cosine relations for model B.
            index = 1 if text == "beta task" else 0
            if self.rotate:
                index = 1 - index
            row = np.zeros(self.dimension, dtype=np.float32)
            row[index] = 1.0
            rows.append(row)
        return np.stack(rows)


def caption(doc_id, start, end, text, **kwargs):
    return {"doc_id": doc_id, "date": "DAY1", "start_time": start,
            "end_time": end, "text": text, **kwargs}


EARLY = caption("early", "00000000", "00003000", "alpha task")
LATE = caption("late", "00010000", "00013000", "beta task")
EARLY_CUTOFF = 100003000
LATE_CUTOFF = 100013000


def make_memory(cache_dir, data=None):
    memory = UNIFIED.EM2Memory(
        embedding_model=FakeEmbedding(), retriever_llm_model=NoModels(),
        prompt_template_manager=NoModels(), episodic_granularities=["30sec", "3min"],
    )
    memory.episodic_memory.dense_cache_dir = str(cache_dir)
    memory.load_episodic_captions(caption_data=data or {"30sec": [EARLY, LATE]})
    return memory
