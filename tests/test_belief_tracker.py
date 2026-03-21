"""Tests for the Bayesian belief tracker."""

import numpy as np
from unspiral.tracking.belief_tracker import BeliefTracker, BeliefState


def test_initial_state():
    bt = BeliefTracker(prior_h=0.5)
    assert abs(bt.p_false_belief - 0.5) < 0.1
    assert abs(bt.expected_pi - 0.5) < 0.1
    assert bt.spiral_risk in ("low", "medium")


def test_update_with_low_sycophancy():
    """Low sycophancy + moderate agreement should move toward truth (H=1)."""
    bt = BeliefTracker(prior_h=0.5)
    # Moderate agreement + low sycophancy = impartial bot agreeing = H=1 evidence
    for t in range(10):
        bt.update(sycophancy_score=0.1, agreement_score=0.6, turn=t + 1)
    # p_false should decrease (truth wins with impartial bot)
    assert bt.p_false_belief < 0.6
    assert len(bt.history) == 10


def test_update_with_high_sycophancy():
    """High sycophancy + high agreement should shift beliefs."""
    bt = BeliefTracker(prior_h=0.5)
    for t in range(20):
        bt.update(sycophancy_score=0.9, agreement_score=0.9, turn=t + 1)
    # Pi estimate should be high
    assert bt.expected_pi > 0.5


def test_trajectory():
    bt = BeliefTracker()
    bt.update(0.5, 0.5, 1)
    bt.update(0.6, 0.6, 2)
    traj = bt.trajectory()
    assert len(traj) == 2
    assert traj[0][0] == 1  # turn
    assert isinstance(traj[0][1], float)  # p_false
    assert isinstance(traj[0][2], float)  # pi


def test_posterior_sums_to_one():
    bt = BeliefTracker()
    bt.update(0.5, 0.7, 1)
    total = bt.posterior.sum()
    assert abs(total - 1.0) < 1e-6


def test_spiral_risk_levels():
    bt = BeliefTracker(prior_h=0.05)  # Strong prior toward H=0
    assert bt.p_false_belief > 0.8
    assert bt.spiral_risk in ("high", "critical")
