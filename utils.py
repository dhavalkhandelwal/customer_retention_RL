"""
utils.py — Constants, state encoding/decoding, and helper functions.

This module defines the state space, action space, and utility functions
used across the project for the Customer Retention RL environment.
"""

import numpy as np

# ============================================================================
# STATE SPACE DEFINITION
# ============================================================================
# Each customer state is a tuple of 4 discrete features:
#   1. Engagement Level:       0=Low, 1=Medium, 2=High
#   2. Months Since Purchase:  0=Recent(0-2), 1=Moderate(3-6), 2=Long(7+)
#   3. Purchase Frequency:     0=Low, 1=Medium, 2=High
#   4. Churn Risk:             0=Low, 1=Medium, 2=High

ENGAGEMENT_LEVELS = ["Low", "Medium", "High"]
MONTHS_SINCE_PURCHASE = ["Recent (0-2)", "Moderate (3-6)", "Long (7+)"]
PURCHASE_FREQUENCY = ["Low", "Medium", "High"]
CHURN_RISK = ["Low", "Medium", "High"]

STATE_DIMENSIONS = [
    len(ENGAGEMENT_LEVELS),
    len(MONTHS_SINCE_PURCHASE),
    len(PURCHASE_FREQUENCY),
    len(CHURN_RISK),
]

TOTAL_STATES = np.prod(STATE_DIMENSIONS)

# ============================================================================
# ACTION SPACE DEFINITION
# ============================================================================
ACTIONS = [
    "Offer Discount",
    "Send Personalized Email",
    "Give Loyalty Points",
    "Do Nothing",
]

NUM_ACTIONS = len(ACTIONS)

# ============================================================================
# STATE ENCODING / DECODING
# ============================================================================

def encode_state(engagement: int, months_since: int, frequency: int, churn: int) -> int:
    """
    Encode a 4-dimensional customer state into a single integer index.

    Parameters
    ----------
    engagement : int  — Engagement level (0, 1, 2)
    months_since : int — Months since last purchase bucket (0, 1, 2)
    frequency : int   — Purchase frequency category (0, 1, 2)
    churn : int       — Churn risk level (0, 1, 2)

    Returns
    -------
    int : Unique state index in [0, TOTAL_STATES)
    """
    return (
        engagement * (3 * 3 * 3)
        + months_since * (3 * 3)
        + frequency * 3
        + churn
    )


def decode_state(state_index: int) -> tuple:
    """
    Decode a single integer state index back into its 4-dimensional tuple.

    Returns
    -------
    tuple : (engagement, months_since, frequency, churn)
    """
    churn = state_index % 3
    state_index //= 3
    frequency = state_index % 3
    state_index //= 3
    months_since = state_index % 3
    state_index //= 3
    engagement = state_index % 3
    return (engagement, months_since, frequency, churn)


def state_to_description(state_index: int) -> str:
    """
    Return a human-readable description of a state.
    """
    eng, msp, freq, churn = decode_state(state_index)
    return (
        f"Engagement={ENGAGEMENT_LEVELS[eng]}, "
        f"MonthsSincePurchase={MONTHS_SINCE_PURCHASE[msp]}, "
        f"PurchaseFreq={PURCHASE_FREQUENCY[freq]}, "
        f"ChurnRisk={CHURN_RISK[churn]}"
    )


def set_random_seed(seed: int = 42):
    """Set numpy random seed for reproducibility."""
    np.random.seed(seed)
