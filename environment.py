"""
environment.py — Custom Customer Retention MDP Environment.

This module implements a simulated customer retention environment where:
  - A customer has a discrete state (engagement, months since purchase,
    purchase frequency, churn risk).
  - The agent takes marketing actions each time step.
  - Customer behavior transitions probabilistically based on the current
    state and action taken.
  - An episode ends when the customer churns or a fixed time horizon is reached.

The environment does NOT use any prebuilt Gym API — it is entirely custom.
"""

import numpy as np
from utils import encode_state, decode_state, NUM_ACTIONS

# HYPERPARAMETERS

MAX_EPISODE_LENGTH = 24  # Maximum months (time steps) per episode


class CustomerEnvironment:
    """
    Custom MDP environment simulating customer retention dynamics.

    State: (engagement, months_since_purchase, purchase_frequency, churn_risk)
           Each feature ∈ {0, 1, 2} (Low, Medium, High)

    Actions:
        0 — Offer Discount
        1 — Send Personalized Email
        2 — Give Loyalty Points
        3 — Do Nothing
    """

    def __init__(self, max_steps: int = MAX_EPISODE_LENGTH):
        self.max_steps = max_steps
        self.current_step = 0
        self.done = False

        # Customer state components
        self.engagement = 0
        self.months_since = 0
        self.frequency = 0
        self.churn_risk = 0

    # RESET
    def reset(self) -> int:
        """
        Reset the environment to a random initial customer state.

        Returns
        -------
        int : encoded initial state
        """
        self.engagement = np.random.randint(0, 3)
        self.months_since = np.random.randint(0, 3)
        self.frequency = np.random.randint(0, 3)
        self.churn_risk = np.random.randint(0, 3)
        self.current_step = 0
        self.done = False
        return self._get_state()

    # STEP
    def step(self, action: int):
        """
        Execute one time step in the environment.

        Parameters
        ----------
        action : int — action index (0–3)

        Returns
        -------
        next_state : int   — encoded next state
        reward     : float — immediate reward
        done       : bool  — whether episode has ended
        info       : dict  — auxiliary info (purchase flag, churned flag)
        """
        assert 0 <= action < NUM_ACTIONS, f"Invalid action {action}"
        assert not self.done, "Episode already finished. Call reset()."

        self.current_step += 1
        reward = 0.0
        purchased = False
        churned = False

        # Apply action effects
        reward += self._apply_action_effects(action)

        # Check purchase
        purchase_prob = self._compute_purchase_probability(action)
        if np.random.random() < purchase_prob:
            purchased = True
            revenue = [5.0, 15.0, 30.0][self.frequency]
            reward += revenue
            self.months_since = 0
            self.engagement = min(2, self.engagement + 1)
        else:
            self.months_since = min(2, self.months_since + 1)

        # Update churn
        self._update_churn_risk()

        # Check churn
        churn_prob = self._compute_churn_probability()
        if np.random.random() < churn_prob:
            churned = True
            reward -= 50.0
            self.done = True

        # Check time horizon
        if self.current_step >= self.max_steps:
            self.done = True

        next_state = self._get_state()
        info = {"purchased": purchased, "churned": churned}
        return next_state, reward, self.done, info

    # INTERNAL HELPERS
    def _get_state(self) -> int:
        """Return encoded integer state."""
        return encode_state(self.engagement, self.months_since,
                            self.frequency, self.churn_risk)

    def _apply_action_effects(self, action: int) -> float:
        """
        Apply the marketing action's direct effects on the customer state
        and return the immediate cost/reward component.

        Returns
        -------
        float : cost component of the reward (negative for costly actions)
        """
        cost = 0.0

        if action == 0:
            cost = -3.0
            if np.random.random() < 0.70:
                self.engagement = min(2, self.engagement + 1)
            if np.random.random() < 0.30:
                self.frequency = min(2, self.frequency + 1)

        elif action == 1:
            cost = -0.5
            if np.random.random() < 0.45:
                self.engagement = min(2, self.engagement + 1)

        elif action == 2:
            cost = -1.5
            if np.random.random() < 0.55:
                self.engagement = min(2, self.engagement + 1)
            if np.random.random() < 0.20:
                self.frequency = min(2, self.frequency + 1)

        elif action == 3:
            cost = 0.0
            if np.random.random() < 0.30:
                self.engagement = max(0, self.engagement - 1)

        if self.engagement == 2:
            cost += 1.0

        return cost

    def _compute_purchase_probability(self, action: int) -> float:
        """
        Compute the probability of a purchase this time step,
        based on engagement, frequency, months since purchase, and action.
        """
        base = [0.05, 0.15, 0.30][self.engagement]
        freq_mod = [0.0, 0.05, 0.10][self.frequency]
        recency_mod = [0.05, 0.0, -0.05][self.months_since]
        action_mod = [0.15, 0.05, 0.08, 0.0][action]

        prob = base + freq_mod + recency_mod + action_mod
        return np.clip(prob, 0.0, 0.85)

    def _update_churn_risk(self):
        """
        Update the customer's churn risk based on current state.
        Low engagement and long absence increase churn risk;
        high engagement and recent activity decrease it.
        """
        if self.engagement == 0 and self.months_since == 2:
            self.churn_risk = min(2, self.churn_risk + 1)
        elif self.engagement == 0:
            if np.random.random() < 0.35:
                self.churn_risk = min(2, self.churn_risk + 1)

        if self.engagement == 2 and self.months_since == 0:
            self.churn_risk = max(0, self.churn_risk - 1)
        elif self.engagement >= 1 and self.months_since <= 1:
            if np.random.random() < 0.25:
                self.churn_risk = max(0, self.churn_risk - 1)

    def _compute_churn_probability(self) -> float:
        """
        Compute the probability that the customer churns this step.
        """
        base = [0.01, 0.08, 0.25][self.churn_risk]

        if self.engagement == 0:
            base += 0.05
        if self.months_since == 2:
            base += 0.05

        return np.clip(base, 0.0, 0.60)
