import numpy as np
import pytest

import pyloudnorm as pyln


@pytest.mark.parametrize("rate", [44100, 48000])
@pytest.mark.parametrize("duration", [0.46, 0.49, 0.56, 0.59, 0.66, 0.69])
def test_incomplete_gating_block_is_discarded(rate, duration):
    data = np.ones(round(duration * rate))
    meter = pyln.Meter(rate, filter_class="custom")

    assert meter.integrated_loudness(data) == pytest.approx(-0.691)
    block_samples = rate * 2 // 5
    hop_samples = rate // 10
    assert len(meter.blockwise_loudness) == (
        (len(data) - block_samples) // hop_samples + 1
    )


@pytest.mark.parametrize("rate", [44100, 48000])
@pytest.mark.parametrize("duration", [0.4, 0.5, 0.6, 0.7, 1.0, 1.1, 1.2])
def test_complete_gating_block_is_retained(rate, duration):
    data = np.ones(round(duration * rate))
    meter = pyln.Meter(rate, filter_class="custom")

    assert meter.integrated_loudness(data) == pytest.approx(-0.691)
    assert len(meter.blockwise_loudness) == round((duration - 0.4) / 0.1) + 1
