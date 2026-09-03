import numpy as np
from preprocessing import glcm_mean_variance_maps, uisi_map


def test_uisi_formula_and_zero_denominator():
    result = uisi_map(np.array([[.4]], np.float32), np.array([[.2]], np.float32), np.array([[.1]], np.float32))
    assert np.isclose(result[0, 0], .5)
    zeros = np.zeros((1, 1), np.float32)
    assert uisi_map(zeros, zeros, zeros)[0, 0] == 0


def test_glcm_constant_field():
    mean, variance = glcm_mean_variance_maps(np.full((9, 9), .5, np.float32))
    assert np.isfinite(mean).all() and np.isfinite(variance).all()
    assert np.isclose(variance[4, 4], 0)


if __name__ == "__main__":
    test_uisi_formula_and_zero_denominator()
    test_glcm_constant_field()
    print("feature tests: OK")
