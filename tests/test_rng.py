import numpy as np

from sizing.rng import Mulberry32, path_seed, path_seeds, step_vector


def test_first_numbers_match_the_javascript_reference():
    # Computed with the standard mulberry32 from the JavaScript specification (node).
    g = Mulberry32(12345)
    got = [g() for _ in range(3)]
    assert got == [0.9797282677609473, 0.3067522644996643, 0.484205421525985]


def test_numbers_are_in_unit_interval_and_roughly_uniform():
    g = Mulberry32(99)
    u = np.array([g() for _ in range(20000)])
    assert u.min() >= 0.0 and u.max() < 1.0
    assert abs(u.mean() - 0.5) < 0.01
    assert abs(u.var() - 1 / 12) < 0.005


def test_path_seeds_are_reproducible_and_distinct():
    a = [path_seed(2026, k) for k in range(2000)]
    assert a == [path_seed(2026, k) for k in range(2000)]
    assert len(set(a)) == 2000
    assert path_seed(2026, 0) != path_seed(2027, 0)


def test_vector_generator_matches_scalar():
    seeds = path_seeds(7, 50)
    assert [int(s) for s in seeds] == [path_seed(7, k) for k in range(50)]
    gens = [Mulberry32(int(s)) for s in seeds]
    a = seeds.copy()
    for _ in range(5):
        a, u = step_vector(a)
        for g, v in zip(gens, u):
            assert g() == v


def test_masked_step_leaves_other_paths_alone():
    a = path_seeds(1, 4)
    mask = np.array([True, False, True, False])
    a2, _ = step_vector(a, mask)
    assert a2[1] == a[1] and a2[3] == a[3] and a2[0] != a[0]
