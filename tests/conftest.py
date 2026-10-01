import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sizing import streams as S  # noqa: E402


@pytest.fixture(scope='session')
def data():
    return S.load()


def toy_stream(x_rows, st_rows, years=None):
    """A hand-made stream: one list of R per rule, and the matching 0/1/2 states."""
    n = len(x_rows[0])
    return S.Stream(
        rules=[f'rule{j}' for j in range(len(x_rows))],
        dates=[f'2020-01-{i + 1:02d}' if i < 28 else '2020-02-01' for i in range(n)],
        year=np.array(years if years is not None else [2020] * n),
        x=[np.array(r, dtype=np.float64) for r in x_rows],
        st=[np.array(s, dtype=np.uint8) for s in st_rows],
        trades=[[] for _ in x_rows],
    )
