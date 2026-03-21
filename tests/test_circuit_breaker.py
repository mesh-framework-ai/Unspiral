"""Tests for the circuit breaker."""

from unspiral.interventions.circuit_breaker import CircuitBreaker, Intervention


def test_no_intervention():
    cb = CircuitBreaker()
    result = cb.evaluate(0.8, turn=1)
    assert result.level == "none"
    assert result.system_injection is None
    assert result.user_warning is None


def test_yellow_intervention():
    cb = CircuitBreaker()
    result = cb.evaluate(0.5, turn=1)
    assert result.level == "yellow"
    assert result.system_injection is not None
    assert "balanced" in result.system_injection.lower()
    assert result.user_warning is None


def test_orange_intervention():
    cb = CircuitBreaker()
    result = cb.evaluate(0.3, turn=1)
    assert result.level == "orange"
    assert result.system_injection is not None
    assert result.response_suffix is not None
    assert result.user_warning is None


def test_red_intervention():
    cb = CircuitBreaker()
    result = cb.evaluate(0.1, turn=1)
    assert result.level == "red"
    assert result.system_injection is not None
    assert result.user_warning is not None
    assert "WARNING" in result.user_warning


def test_system_injection_tracking():
    cb = CircuitBreaker()
    cb.evaluate(0.3, turn=1)
    assert cb.active_level == "orange"
    assert cb.get_system_injection() is not None
    cb.evaluate(0.8, turn=2)
    assert cb.active_level == "none"
    assert cb.get_system_injection() is None


def test_intervention_history():
    cb = CircuitBreaker()
    cb.evaluate(0.8, turn=1)
    cb.evaluate(0.5, turn=2)
    cb.evaluate(0.1, turn=3)
    assert len(cb.interventions) == 3
    assert cb.interventions[0].level == "none"
    assert cb.interventions[1].level == "yellow"
    assert cb.interventions[2].level == "red"
