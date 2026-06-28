# Reinforcement Learning Based Customer Lifetime Value Optimization using Q-Learning

## Overview

This project simulates a **customer retention system** where a Reinforcement Learning (RL) agent learns optimal marketing strategies to maximize long-term **Customer Lifetime Value (CLV)**.

Instead of relying on rule-based heuristics, the agent discovers — through trial and error — which marketing actions (discounts, emails, loyalty points, or inaction) are most effective for different customer profiles.

---

## MDP Formulation

The problem is modeled as a **Markov Decision Process (MDP)**:

### State Space (81 discrete states)

Each customer state is defined by four features:

| Feature                 | Values                          |
|-------------------------|---------------------------------|
| Engagement Level        | Low (0), Medium (1), High (2)   |
| Months Since Purchase   | Recent (0), Moderate (1), Long (2) |
| Purchase Frequency      | Low (0), Medium (1), High (2)   |
| Churn Risk              | Low (0), Medium (1), High (2)   |

**Total States:** 3 × 3 × 3 × 3 = **81**

### Action Space (4 actions)

| Action                   | Cost  | Effect                            |
|--------------------------|-------|-----------------------------------|
| Offer Discount           | −3.0  | Strong engagement boost           |
| Send Personalized Email  | −0.5  | Moderate engagement boost         |
| Give Loyalty Points      | −1.5  | Good retention, moderate cost     |
| Do Nothing               |  0.0  | Engagement may decay              |

### Reward Function

| Event                    | Reward     |
|--------------------------|------------|
| Customer makes a purchase | +5 to +30 (based on frequency tier) |
| High engagement reached  | +1.0       |
| Action cost              | −0.5 to −3.0 |
| Customer churns          | **−50.0**  |

The reward design encourages **long-term retention** over short-term profit. Losing a customer is far more costly than any marketing spend.

---

## Algorithm

**Tabular Q-Learning** with the update rule:

```
Q(s, a) ← Q(s, a) + α [ r + γ max_a' Q(s', a') − Q(s, a) ]
```

### Hyperparameters

| Parameter        | Value  |
|------------------|--------|
| Learning rate (α)| 0.1    |
| Discount factor (γ) | 0.95 |
| Initial epsilon  | 1.0    |
| Minimum epsilon  | 0.01   |
| Epsilon decay    | 0.997  |
| Training episodes| 2000   |

---

## Project Structure

```
customer_retention_rl/
├── environment.py        # Custom MDP environment
├── q_learning_agent.py   # Tabular Q-learning agent
├── train.py              # Training loop & entry point
├── visualization.py      # Matplotlib plotting utilities
├── utils.py              # State encoding, constants, helpers
├── requirements.txt      # Python dependencies
├── README.md             # This file
└── plots/                # Generated training plots (after running)
```

---

## How to Run

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Run training

```bash
python train.py
```

This will:
- Train the Q-learning agent for 2000 episodes
- Print progress every 200 episodes
- Display the learned optimal policy for all 81 states
- Print a policy summary and business insights
- Generate and save training plots to `plots/`

---

## Generated Visualizations

After training, four plots are generated:

1. **Reward vs Episodes** — shows the total reward per episode increasing as the agent learns
2. **Average CLV vs Episodes** — shows customer lifetime value improving over training
3. **Churn Rate vs Episodes** — shows churn rate decreasing as the agent learns to retain customers
4. **Epsilon Decay Curve** — shows the exploration rate decreasing over time

---

## Analysis: How RL Improves Long-Term Customer Value

### Early Training (Random Policy)
- The agent explores randomly, applying discounts to already-loyal customers (wasting money) and doing nothing for at-risk customers (causing churn).
- Churn rates are high and CLV is low.

### Mid Training (Learning Phase)
- The Q-table values start to differentiate between states. The agent begins to discover that **proactive retention** for high-churn-risk customers yields better cumulative rewards.
- Churn rate begins to decrease.

### Late Training (Converged Policy)
- The agent has learned to:
  - Use **discounts and loyalty points** for high-risk, low-engagement customers to prevent the costly −50 churn penalty.
  - Use **emails or do nothing** for already-engaged, low-risk customers — saving costs while maintaining revenue.
- This represents a **nuanced, state-dependent marketing strategy** that would be difficult to hand-craft.

### Key Insight
The discount factor γ = 0.95 is crucial — it makes the agent value future revenue streams. A myopic agent (γ ≈ 0) would avoid all marketing costs, leading to higher churn and lower CLV. The RL agent learns that **spending a little now prevents losing a lot later**.

---

## Dependencies

- Python 3.7+
- NumPy ≥ 1.21
- Matplotlib ≥ 3.5

---

## Constraints

- ✅ Pure Python with NumPy and Matplotlib only
- ✅ Tabular Q-learning (no deep learning)
- ✅ Custom environment (no prebuilt Gym)
- ✅ Modular, academically clean code
