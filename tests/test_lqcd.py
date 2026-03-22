"""Tests for the LQCD probe (unit tests that don't require OpenAI)."""

from unspiral.detectors.lqcd_probe import _has_opinion, LQCDProbe


def test_has_opinion_positive():
    assert _has_opinion("I think vaccines are dangerous")
    assert _has_opinion("I believe the earth is flat")
    assert _has_opinion("Don't you think this is wrong?")
    assert _has_opinion("Obviously, this is the case")


def test_has_opinion_negative():
    assert not _has_opinion("What is the weather today?")
    assert not _has_opinion("Tell me about quantum physics")
    assert not _has_opinion("How does photosynthesis work?")


def test_kl_divergence_identical():
    """KL divergence of identical distributions should be ~0."""
    probe = LQCDProbe.__new__(LQCDProbe)  # skip __init__ (no client needed)
    logprobs = [
        {"top_logprobs": {"the": -0.5, "a": -1.2, "an": -2.0}},
        {"top_logprobs": {"is": -0.3, "was": -1.5}},
    ]
    kl = probe.compute_kl_divergence(logprobs, logprobs)
    assert abs(kl) < 1e-6


def test_kl_divergence_different():
    """KL divergence of different distributions should be > 0."""
    probe = LQCDProbe.__new__(LQCDProbe)
    logprobs_p = [
        {"top_logprobs": {"yes": -0.1, "no": -2.5}},
    ]
    logprobs_q = [
        {"top_logprobs": {"yes": -2.5, "no": -0.1}},
    ]
    kl = probe.compute_kl_divergence(logprobs_p, logprobs_q)
    assert kl > 0.5


def test_kl_divergence_empty():
    probe = LQCDProbe.__new__(LQCDProbe)
    assert probe.compute_kl_divergence([], []) == 0.0
