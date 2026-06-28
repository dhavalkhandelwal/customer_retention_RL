"""
q_learning_agent.py — Tabular Q-Learning Agent.

Implements the standard Q-learning update rule:
    Q(s, a) = Q(s, a) + α [ r + γ max_a' Q(s', a') − Q(s, a) ]

Features:
    - Epsilon-greedy exploration with configurable decay
    - Q-table stored as a 2D NumPy array (states × actions)
"""

import numpy as np
from utils import TOTAL_STATES, NUM_ACTIONS, ACTIONS, decode_state, state_to_description


class QLearningAgent:

    def __init__(
        self,
        alpha: float = 0.1,
        gamma: float = 0.95,
        epsilon: float = 1.0,
        epsilon_min: float = 0.01,
        epsilon_decay: float = 0.995,
    ):
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

        # Initialize Q-table with small random values to break ties
        self.q_table = np.random.uniform(low=0.0, high=0.01,
                                         size=(TOTAL_STATES, NUM_ACTIONS))

    
    def choose_action(self, state: int) -> int:
        if np.random.random() < self.epsilon:
            return np.random.randint(0, NUM_ACTIONS)
        else:
            return int(np.argmax(self.q_table[state]))

    
    def update(self, state: int, action: int, reward: float,
               next_state: int, done: bool):
        """
        Apply the Q-learning update rule.

        Q(s, a) ← Q(s, a) + α [ r + γ max_a' Q(s', a') − Q(s, a) ]

        If the episode is done, the target is simply r (no future value).
        """
        best_next = np.max(self.q_table[next_state]) if not done else 0.0
        td_target = reward + self.gamma * best_next
        td_error = td_target - self.q_table[state, action]
        self.q_table[state, action] += self.alpha * td_error

    
    def decay_epsilon(self):
        """Decay epsilon after each episode."""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    
    def get_policy(self) -> dict:
        """
        Extract the learned greedy policy from the Q-table.

        Returns
        -------
        dict : {state_index: (action_index, action_name)}
        """
        policy = {}
        for s in range(TOTAL_STATES):
            best_action = int(np.argmax(self.q_table[s]))
            policy[s] = (best_action, ACTIONS[best_action])
        return policy

    def print_policy(self):
        """Print the learned policy in a readable format."""
        policy = self.get_policy()
        print("\n" + "=" * 80)
        print("LEARNED OPTIMAL POLICY")
        print("=" * 80)
        for s in range(TOTAL_STATES):
            action_idx, action_name = policy[s]
            desc = state_to_description(s)
            q_vals = self.q_table[s]
            print(f"  State {s:>2d}: {desc}")
            print(f"           → Action: {action_name}  "
                  f"(Q-values: {np.array2string(q_vals, precision=2)})")
        print("=" * 80)

    def print_policy_summary(self):
        """
        Print a concise summary showing which strategy is preferred
        for different customer types.
        """
        policy = self.get_policy()

        print("\n" + "=" * 70)
        print("POLICY SUMMARY — Preferred Marketing Strategy by Customer Type")
        print("=" * 70)

        from collections import Counter
        engagement_labels = ["Low", "Medium", "High"]
        churn_labels = ["Low", "Medium", "High"]

        for eng in range(3):
            for churn in range(3):
                actions_chosen = []
                for msp in range(3):
                    for freq in range(3):
                        from utils import encode_state
                        s = encode_state(eng, msp, freq, churn)
                        actions_chosen.append(policy[s][1])
                counter = Counter(actions_chosen)
                dominant = counter.most_common(1)[0]
                print(f"  Engagement={engagement_labels[eng]:>6s}, "
                      f"ChurnRisk={churn_labels[churn]:>6s}  →  "
                      f"{dominant[0]} ({dominant[1]}/9 states)")

        print("=" * 70)
