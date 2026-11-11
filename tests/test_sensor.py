import cv2
import numpy as np
import pytest

from synthrig.opticsrig.schema import Sensor
from synthrig.sensor import _BAYER_CODES, apply_sensor, mosaic


@pytest.mark.parametrize("pattern", ["RGGB", "BGGR", "GRBG", "GBRG"])
@pytest.mark.parametrize("channel", [0, 1, 2])  # pure R, G or B scene (RGB order)
def test_demosaic_keeps_colour(pattern, channel):
    rgb = np.zeros((16, 16, 3), np.uint16)
    rgb[..., channel] = 1000
    bgr = cv2.cvtColor(mosaic(rgb, pattern), _BAYER_CODES[pattern])
    assert int(np.argmax(bgr[8, 8])) == 2 - channel  # BGR output: R lands in channel 2


def test_colour_sensor_writes_red_as_red():
    rad = np.zeros((16, 16, 3))
    rad[..., 0] = 0.5
    s = Sensor(mono=False, bayer="RGGB", rgb_response=[1, 1, 1], shot_noise=False, read_noise_e=0)
    out = apply_sensor(rad, s, np.random.default_rng(0)).image
    assert out[8, 8, 2] > 0 and out[8, 8, 0] == 0
