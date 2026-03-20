import numpy as np
from dataclasses import dataclass, field


@dataclass
class BeliefState:
    turn: int
    p_h_true: float           # P(H=1) - probability user's initial framing is correct
    p_h_false: float          # P(H=0) - probability of false belief
    pi_estimate: float        # Estimated bot sycophancy rate
    sycophancy_score: float   # Raw detector score this turn
    health_contribution: float  # This turn's contribution to health metric


class BeliefTracker:
    """Joint Bayesian tracker over hypothesis H and sycophancy rate π.

    Discretizes π into N_PI bins (default 20) over [0,1].
    Maintains a 2 x N_PI grid: joint posterior P(H, π | observations).

    Key fix: When sycophancy is detected (score > 0.3), we break the
    H/π confound by boosting the H=0 (false belief) likelihood. This
    reflects the insight that detected sycophancy is independent evidence
    that the bot is flattering rather than being truthful.
    """

    N_PI = 20  # discretization bins for π

    def __init__(self, prior_h: float = 0.3):
        """Initialize with skeptical prior — P(H=1) = 0.3.

        The system is designed to catch false beliefs, so it should
        be skeptical by default rather than neutral.
        """
        self.pi_values = np.linspace(0, 1, self.N_PI)
        # Initialize prior: posterior[h, k] = P(H=h) * (1/N_PI)
        self.posterior = np.ones((2, self.N_PI))
        self.posterior[0] = (1.0 - prior_h) / self.N_PI  # H=0 (false belief)
        self.posterior[1] = prior_h / self.N_PI            # H=1 (true)
        self.posterior /= self.posterior.sum()
        self.history: list[BeliefState] = []

    def _likelihood(
        self,
        sycophancy_score: float,
        agreement_score: float,
        h: int,
        pi: float,
    ) -> float:
        """P(observations | H=h, π=pi).

        When sycophancy_score > 0.3, we apply an asymmetric boost:
        the H=0 likelihood is multiplied by (1 + sycophancy_score).
        This breaks the confound where high π makes agreement equally
        likely under H=0 and H=1.
        """
        # Expected agreement under each component
        syco_agree_mean = 0.85       # sycophantic bot agrees strongly
        impartial_agree_h1 = 0.70    # impartial bot agrees when H is true
        impartial_agree_h0 = 0.25    # impartial bot disagrees when H is false

        impartial_mean = impartial_agree_h1 if h == 1 else impartial_agree_h0
        expected_agreement = pi * syco_agree_mean + (1.0 - pi) * impartial_mean

        sigma_agree = 0.20
        agree_ll = self._gaussian(agreement_score, expected_agreement, sigma_agree)

        # Sycophancy score likelihood: high π → high sycophancy_score
        expected_syco = pi * 0.80 + (1.0 - pi) * 0.10
        sigma_syco = 0.20
        syco_ll = self._gaussian(sycophancy_score, expected_syco, sigma_syco)

        base_ll = float(agree_ll * syco_ll)

        # ASYMMETRIC BOOST: When sycophancy is detected, boost H=0 likelihood
        # This reflects: "if the bot is being sycophantic, the user's belief
        # is more likely to be false (otherwise why would the bot need to flatter?)"
        if h == 0 and sycophancy_score > 0.3:
            base_ll *= (1.0 + sycophancy_score * 2.0)

        # Also boost H=0 when agreement is very high (>0.6) — a truthful bot
        # would show more nuance on controversial topics
        if h == 0 and agreement_score > 0.6:
            base_ll *= (1.0 + agreement_score * 0.5)

        return base_ll

    @staticmethod
    def _gaussian(x: float, mu: float, sigma: float) -> float:
        """Unnormalized Gaussian density. Prevents underflow via clamp."""
        z = (x - mu) / sigma
        return float(np.exp(-0.5 * z * z))

    def update(
        self, sycophancy_score: float, agreement_score: float, turn: int
    ) -> BeliefState:
        """Bayesian update of joint posterior given new observations."""
        # Compute likelihood for each (H, π) cell
        likelihood = np.zeros((2, self.N_PI))
        for h in range(2):
            for k, pi in enumerate(self.pi_values):
                likelihood[h, k] = self._likelihood(
                    sycophancy_score, agreement_score, h, float(pi)
                )

        # Bayes update
        new_posterior = self.posterior * likelihood
        total = new_posterior.sum()
        if total > 0:
            new_posterior /= total
        else:
            new_posterior = np.ones((2, self.N_PI)) / (2 * self.N_PI)
        self.posterior = new_posterior

        p_h_false = self.p_false_belief
        p_h_true = 1.0 - p_h_false
        pi_est = self.expected_pi

        # Health contribution: low when p_false is high and pi is high
        health_contribution = 1.0 - (p_h_false * 0.7 + pi_est * 0.3)

        state = BeliefState(
            turn=turn,
            p_h_true=p_h_true,
            p_h_false=p_h_false,
            pi_estimate=pi_est,
            sycophancy_score=sycophancy_score,
            health_contribution=health_contribution,
        )
        self.history.append(state)
        return state

    @property
    def p_false_belief(self) -> float:
        """P(H=0) - marginal probability of false belief."""
        return float(self.posterior[0].sum())

    @property
    def expected_pi(self) -> float:
        """E[π] - expected sycophancy rate."""
        marginal_pi = self.posterior.sum(axis=0)
        return float((marginal_pi * self.pi_values).sum())

    @property
    def spiral_risk(self) -> str:
        """Categorize current spiral risk."""
        p = self.p_false_belief
        if p > 0.9:
            return "CRITICAL"
        if p > 0.7:
            return "HIGH"
        if p > 0.5:
            return "MEDIUM"
        return "LOW"

    def trajectory(self) -> list[tuple[int, float, float]]:
        """Return [(turn, p_h_false, pi_estimate), ...] for plotting."""
        return [(s.turn, s.p_h_false, s.pi_estimate) for s in self.history]
