import warnings

import numpy as np
import pytest

import pyloudnorm as pyln


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("channels", [1, 2])
@pytest.mark.parametrize("mode", ["peak", "loudness"])
def test_silent_normalization(dtype, channels, mode):
    shape = (48000,) if channels == 1 else (48000, channels)
    data = np.zeros(shape, dtype=dtype)
    data.flags.writeable = False
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        if mode == "peak":
            actual = pyln.normalize.peak(data, -1.0)
        else:
            actual = pyln.normalize.loudness(data, -np.inf, -23.0)
    np.testing.assert_array_equal(actual, data)
    assert actual.dtype == data.dtype
    assert not np.shares_memory(actual, data)
