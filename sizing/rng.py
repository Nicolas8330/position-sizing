"""The random number generator, written to be identical in Python and JavaScript.

Why not use numpy's generator? Because the site runs the same simulation in
JavaScript, and the only way to prove the two programs agree is to feed them
the same random numbers. mulberry32 is a small generator whose whole state is
one 32-bit integer, so it can be written with the same integer arithmetic in
both languages. A test in the site repository (scripts/verify-position-sizing.mjs)
compares the two on every run, to a tolerance of 1e-9.

Every simulated path has its own generator, seeded from (seed, path number).
That way path k does not depend on how many random numbers the earlier paths
used, the results for 2,000 paths are the first 2,000 of the results for
5,000, and the paths can be computed side by side in numpy.

mulberry32 is a fast, simple generator, not a cryptographic one. Its state is
only 32 bits, so two paths can in rare cases share a stretch of numbers. With
a few thousand paths this has no visible effect on the results (see REVIEW.md).
"""
from __future__ import annotations

import numpy as np

MASK = 0xFFFFFFFF
GOLDEN = 0x9E3779B1
STEP = 0x6D2B79F5


def path_seed(seed: int, k: int) -> int:
    """Seed of path number k: a 32-bit hash (murmur3 finaliser) of seed and k."""
    h = ((((k + 1) * GOLDEN) & MASK) ^ (seed & MASK)) & MASK
    h ^= h >> 16
    h = (h * 0x85EBCA6B) & MASK
    h ^= h >> 13
    h = (h * 0xC2B2AE35) & MASK
    h ^= h >> 16
    return h & MASK


class Mulberry32:
    """One generator. Call the object to get the next number in [0, 1)."""

    def __init__(self, seed: int):
        self.a = seed & MASK

    def __call__(self) -> float:
        self.a = (self.a + STEP) & MASK
        t = self.a
        t = ((t ^ (t >> 15)) * (t | 1)) & MASK
        t = (t ^ ((t + (((t ^ (t >> 7)) * (t | 61)) & MASK)) & MASK)) & MASK
        return ((t ^ (t >> 14)) & MASK) / 4294967296.0


def path_seeds(seed: int, n: int) -> np.ndarray:
    """Vector of path seeds, the numpy twin of path_seed (uint64 holding 32-bit values)."""
    k = np.arange(n, dtype=np.uint64)
    m = np.uint64(MASK)
    h = (((k + np.uint64(1)) * np.uint64(GOLDEN)) & m) ^ np.uint64(seed & MASK)
    h ^= h >> np.uint64(16)
    h = (h * np.uint64(0x85EBCA6B)) & m
    h ^= h >> np.uint64(13)
    h = (h * np.uint64(0xC2B2AE35)) & m
    h ^= h >> np.uint64(16)
    return h & m


def step_vector(a: np.ndarray, mask: np.ndarray | None = None):
    """Advance the generators in `a` (uint64 array) and return (new_a, u).

    Only the entries where `mask` is True move; the others keep their state,
    exactly as a path that did not ask for a number keeps its generator still.
    """
    m = np.uint64(MASK)
    a2 = (a + np.uint64(STEP)) & m
    if mask is not None:
        a2 = np.where(mask, a2, a)
    t = a2
    t = ((t ^ (t >> np.uint64(15))) * (t | np.uint64(1))) & m
    inner = ((t ^ (t >> np.uint64(7))) * (t | np.uint64(61))) & m
    t = (t ^ ((t + inner) & m)) & m
    u = ((t ^ (t >> np.uint64(14))) & m).astype(np.float64) / 4294967296.0
    return a2, u
