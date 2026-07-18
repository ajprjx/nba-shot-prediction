import numpy as np
from src.forecast import compositional as comp


def test_alr_roundtrip_sums_to_one():
    shares = np.array([0.5, 0.3, 0.2])
    back = comp.inverse_alr(comp.alr(shares))
    assert np.allclose(back, shares)
    assert np.isclose(back.sum(), 1.0)


def test_inverse_alr_always_sums_to_one():
    coords = np.array([2.5, -1.0, 0.3, 4.0])
    out = comp.inverse_alr(coords)
    assert np.isclose(out.sum(), 1.0)
    assert (out > 0).all()


def test_shares_matrix_shape_and_order(synthetic_canonical):
    zones = ["restricted_area", "above_break_3"]
    years, mat = comp.shares_matrix(synthetic_canonical, zones)
    assert mat.shape == (10, 2)
    assert years[0] == 2015 and years[-1] == 2024
    # each row sums to ~1 across these two zones (they're the only ones present)
    assert np.allclose(mat.sum(axis=1), 1.0)
