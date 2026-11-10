"""Shared, content-addressed array cache for the immutable case data.

`sequence.npz` is compressed: slicing one window decompresses the whole array (~2.7 s per window). The cache stores
each needed array ONCE per host, uncompressed, under  $FSI_ARRAY_CACHE/<sequence sha256[:16]>/<key>.npy , written
atomically (tmp + rename, O_EXCL lock) and opened read-only with mmap by every process, so concurrent ranks share
one page-cache copy. Values are the npz arrays bit for bit. Without $FSI_ARRAY_CACHE the npz is read directly.
The cache is derived data and may be deleted at any time; the source npz is never modified.

file_sha256 memoises SHA-256 per (device, inode, size, mtime_ns) within a process (data and checkpoints are read-only).
"""
from __future__ import annotations

import hashlib
import os
import time
from pathlib import Path

import numpy as np

_SHA: dict = {}
_ARR: dict = {}


def file_sha256(path) -> str:
    st = os.stat(path); key = (str(path), st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns)
    if key not in _SHA:
        h = hashlib.sha256()
        with open(path, 'rb') as f:
            for block in iter(lambda: f.read(1 << 20), b''): h.update(block)
        _SHA[key] = h.hexdigest()
    return _SHA[key]


def case_array(folder, key: str, sequence_sha256: str):
    """Full array `key` of folder/sequence.npz (read-only). Caller slices/copies; never mutates."""
    folder = Path(folder); ident = (sequence_sha256, key)
    if ident in _ARR: return _ARR[ident]
    root = os.environ.get('FSI_ARRAY_CACHE')
    if not root:
        with np.load(folder / 'sequence.npz', allow_pickle=False) as z: arr = z[key]
        _ARR[ident] = arr; return arr
    d = Path(root) / sequence_sha256[:16]; d.mkdir(parents=True, exist_ok=True)
    target = d / (key + '.npy'); lock = d / (key + '.lock')
    if not target.exists():
        try:
            fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY); os.close(fd); owner = True
        except FileExistsError:
            owner = False
        if owner:
            try:
                if file_sha256(folder / 'sequence.npz') != sequence_sha256: raise ValueError('Cache source SHA mismatch')
                with np.load(folder / 'sequence.npz', allow_pickle=False) as z: arr = z[key]
                tmp = d / ('.%s.%d.tmp.npy' % (key, os.getpid())); np.save(tmp, arr); os.replace(tmp, target)
            finally:
                lock.unlink(missing_ok=True)
        else:
            for _ in range(3600):
                if target.exists() and not lock.exists(): break
                try:
                    stale = time.time() - lock.stat().st_mtime > 600   # writer died: stale lock
                except FileNotFoundError:
                    stale = False
                if stale:
                    lock.unlink(missing_ok=True); return case_array(folder, key, sequence_sha256)
                time.sleep(0.5)
            else:
                raise TimeoutError('array cache writer did not finish: ' + str(target))
    arr = np.load(target, mmap_mode='r', allow_pickle=False)
    _ARR[ident] = arr; return arr
