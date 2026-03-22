"""Tests for the drift monitor."""

import numpy as np
from unspiral.tracking.drift_monitor import DriftMonitor, HealthSnapshot


def test_initial_health():
    dm = DriftMonitor()
    assert dm.current_health == 1.0
    assert dm.current_level == "green"


def test_healthy_conversation():
    dm = DriftMonitor()
    emb = np.random.randn(384).astype(np.float32)
    dm.set_initial_topic(emb)
    snap = dm.update(0.1, 0.3, 0.5, emb, turn=1)
    assert snap.health_score > 0.5
    assert snap.level == "green"


def test_degrading_health():
    dm = DriftMonitor()
    emb = np.random.randn(384).astype(np.float32)
    dm.set_initial_topic(emb)
    # Feed high sycophancy + high agreement + extreme belief
    for t in range(10):
        dm.update(0.8, 0.9, 0.85, emb, turn=t + 1)
    assert dm.current_health < 0.6
    assert dm.current_level in ("yellow", "orange", "red")


def test_agreement_streak():
    dm = DriftMonitor()
    emb = np.random.randn(384).astype(np.float32)
    dm.set_initial_topic(emb)
    for t in range(5):
        dm.update(0.1, 0.8, 0.5, emb, turn=t + 1)  # agreement > 0.7
    assert dm.current_streak == 5
    dm.update(0.1, 0.3, 0.5, emb, turn=6)  # breaks streak
    assert dm.current_streak == 0


def test_sycophancy_density():
    dm = DriftMonitor()
    emb = np.random.randn(384).astype(np.float32)
    dm.set_initial_topic(emb)
    # 5 turns above threshold, 5 below
    for t in range(5):
        dm.update(0.5, 0.5, 0.5, emb, turn=t + 1)
    for t in range(5):
        dm.update(0.1, 0.5, 0.5, emb, turn=t + 6)
    snap = dm.snapshots[-1]
    assert snap.sycophancy_density == 0.5  # 5/10 in window
