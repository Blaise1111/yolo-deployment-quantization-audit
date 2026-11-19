"""Verify split isolation, image geometry, and paired metric definitions."""
import json

import numpy as np
import pytest

from src.coco_split import ANNOTATIONS, split_ids, load_config
from src.flips_and_ece import attribution, flip_rate, greedy_matches, reliability, temperature_fit, transform
from src.unified_infer import letterbox, restore_boxes


def test_official_frozen_split():
    ids = [row["id"] for row in json.loads(ANNOTATIONS.read_text(encoding="utf-8"))["images"]]
    initial = split_ids(ids, 2027)
    assert initial == split_ids(list(reversed(ids)), 2027)
    calib, evaluation, fast = map(set, initial)
    assert (len(calib), len(evaluation), len(fast)) == (200, 4800, 1000)
    assert not (calib & evaluation)
    assert fast <= evaluation
    assert calib | evaluation == set(ids)


@pytest.mark.parametrize("shape", [(371, 651), (651, 371), (300, 300)])
def test_letterbox_inverse_and_rgb(shape):
    frame = np.empty((*shape, 3), np.uint8)
    frame[:] = (10, 20, 30)
    image, geometry = letterbox(frame, 480)
    assert image.shape == (1, 3, 480, 480)
    assert image[0, :, geometry.top, geometry.left] == pytest.approx([30/255, 20/255, 10/255])
    if geometry.left or geometry.top:
        assert image[0, :, 0, 0] == pytest.approx([114/255] * 3)
    box = np.array([[6, 9, 47, 51]], dtype=np.float64)
    transformed = box.copy()
    transformed[:, [0, 2]] = box[:, [0, 2]] * geometry.resized_width / geometry.width + geometry.left
    transformed[:, [1, 3]] = box[:, [1, 3]] * geometry.resized_height / geometry.height + geometry.top
    np.testing.assert_allclose(restore_boxes(transformed, geometry), box, atol=1e-10, rtol=0)


def test_reliability_metrics():
    metrics, bins = reliability(np.array([0.25, 0.75, 0.9, 1.0]), np.array([0, 1, 0, 1]), 10)
    assert metrics["ece"] == pytest.approx((0.25 + 0.25 + abs(1.9-1))/4)
    assert metrics["mce"] == pytest.approx(0.45)
    assert metrics["brier"] == pytest.approx((0.25**2 + 0.25**2 + 0.9**2)/4)
    assert sum(row["count"] for row in bins) == 4


def test_pairing_and_flip_rate():
    ref = np.array([[1, 1, .5, 0, 0, 10, 10], [2, 2, .7, 4, 4, 11, 11]])
    other = np.array([[1, 1, .51, 0, 0, 10, 10], [3, 4, .8, 4, 4, 11, 11]])
    assert greedy_matches(ref, other, .5) == [(0, 0, 1)]
    assert flip_rate(1, 1, 1) == pytest.approx(2/3)
    assert flip_rate(0, 0, 0) == 0


def test_attribution_requires_suppression_reversal():
    config = load_config()
    a = np.array([0, 1, .8, 0, 0, 10, 10])
    b = np.array([1, 1, .7, 1, 0, 11, 10])
    swapped_a, swapped_b = a.copy(), b.copy()
    swapped_a[2], swapped_b[2] = .6, .9
    assert attribution(a, {0:a, 1:b}, {0:swapped_a, 1:swapped_b}, {0}, {1}, config) == "nms_reversal"
    assert attribution(a, {0:a, 1:b}, {0:swapped_a}, {0}, set(), config) == "unresolved"
    edge = a.copy()
    edge[2] = .26
    assert attribution(edge, {0:edge}, {}, {0}, set(), config) == "threshold_edge"


def test_calibration_preserves_order_and_ties():
    raw = np.array([0, .25, .75, .75, .9, 1])
    t = temperature_fit(raw, np.array([0, 0, 0, 1, 1, 1]))
    assert 0.5 <= t <= 5
    assert np.array_equal(np.sign(np.diff(raw)), np.sign(np.diff(transform(raw, t))))
