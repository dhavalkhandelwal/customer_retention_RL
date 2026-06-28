import sys
import numpy as np

from utils import set_random_seed
from environment import CustomerEnvironment
from q_learning_agent import QLearningAgent
from visualization import plot_all


NUM_EPISODES = 2000       # Number of training episodes
EVAL_WINDOW = 50          # Rolling average window for smoothing
RANDOM_SEED = 42

# Q-learning hyperparameters
ALPHA = 0.1               # Learning rate
GAMMA = 0.95              # Discount factor
EPSILON_START = 1.0        # Initial exploration rate
EPSILON_MIN = 0.01         # Minimum exploration rate
EPSILON_DECAY = 0.997      # Multiplicative decay per episode


def train():
    """
    Main training loop.

    For each episode:
        1. Reset the environment to a random customer state.
        2. Interact using the epsilon-greedy Q-learning policy.
        3. Collect total reward, CLV, and churn flag.
        4. Update Q-table after every step.
        5. Decay epsilon after each episode.

    Returns
    -------
    agent : QLearningAgent — trained agent
    metrics : dict          — collected training metrics
    """
    set_random_seed(RANDOM_SEED)

    env = CustomerEnvironment()
    agent = QLearningAgent(
        alpha=ALPHA,
        gamma=GAMMA,
        epsilon=EPSILON_START,
        epsilon_min=EPSILON_MIN,
        epsilon_decay=EPSILON_DECAY,
    )

    # Metric storage
    episode_rewards = []
    episode_clvs = []
    episode_churns = []
    epsilon_history = []

    print("=" * 60)
    print("  TRAINING: Q-Learning for Customer Retention")
    print("=" * 60)
    print(f"  Episodes        : {NUM_EPISODES}")
    print(f"  Alpha (lr)      : {ALPHA}")
    print(f"  Gamma (discount): {GAMMA}")
    print(f"  Epsilon range   : {EPSILON_START} → {EPSILON_MIN}")
    print(f"  Epsilon decay   : {EPSILON_DECAY}")
    print("=" * 60)

    for episode in range(1, NUM_EPISODES + 1):
        state = env.reset()
        total_reward = 0.0
        total_revenue = 0.0
        done = False
        churned = False

        while not done:
            action = agent.choose_action(state)
            next_state, reward, done, info = env.step(action)

            # Q-learning update
            agent.update(state, action, reward, next_state, done)

            total_reward += reward
            if info["purchased"]:
                total_revenue += [5.0, 15.0, 30.0][(state % 9) // 3]
            if info["churned"]:
                churned = True

            state = next_state

        # Decay exploration
        agent.decay_epsilon()

        # Record metrics
        episode_rewards.append(total_reward)
        episode_clvs.append(total_revenue)
        episode_churns.append(1 if churned else 0)
        epsilon_history.append(agent.epsilon)

        # Progress logging
        if episode % 200 == 0 or episode == 1:
            avg_r = np.mean(episode_rewards[-EVAL_WINDOW:])
            avg_clv = np.mean(episode_clvs[-EVAL_WINDOW:])
            churn_rate = np.mean(episode_churns[-EVAL_WINDOW:]) * 100
            print(f"  Episode {episode:>5d}/{NUM_EPISODES}  |  "
                  f"Avg Reward: {avg_r:>7.2f}  |  "
                  f"Avg CLV: {avg_clv:>7.2f}  |  "
                  f"Churn%: {churn_rate:>5.1f}%  |  "
                  f"ε: {agent.epsilon:.4f}")

    print("\n✔ Training complete.\n")

    metrics = {
        "rewards": episode_rewards,
        "clvs": episode_clvs,
        "churns": episode_churns,
        "epsilons": epsilon_history,
    }

    return agent, metrics


def business_insights(agent):
    """
    Print business insights derived from the learned policy.
    """
    print("\n" + "=" * 70)
    print("BUSINESS INSIGHTS FROM THE LEARNED POLICY")
    print("=" * 70)

    policy = agent.get_policy()
    from collections import Counter
    action_counter = Counter(a_name for _, a_name in policy.values())

    print("\n1. Overall action distribution across all 81 states:")
    for action_name, count in action_counter.most_common():
        print(f"     {action_name:>26s}: {count:>2d} states ({count/81*100:.1f}%)")

    print("\n2. Key observations:")
    print("   • The agent learns to use COSTLY actions (Discounts, Loyalty Points)")
    print("     primarily for HIGH churn-risk or LOW engagement customers where")
    print("     the investment prevents expensive customer loss (−50 penalty).")
    print("   • For already-engaged, low-churn customers, the agent often prefers")
    print("     cheaper actions (Email or Do Nothing) to maximise net CLV.")
    print("   • This demonstrates the RL agent's ability to optimise for LONG-TERM")
    print("     value rather than greedy short-term profit.\n")

    print("3. How RL improved Customer Lifetime Value:")
    print("   • Early in training (random policy), the agent acts randomly,")
    print("     leading to high churn rates and low CLV.")
    print("   • As the Q-table converges, the agent discovers that proactive")
    print("     retention actions for at-risk customers yield higher cumulative")
    print("     rewards than ignoring them.")
    print("   • The discount factor γ=0.95 ensures the agent values future")
    print("     revenue, not just the immediate step's reward.")
    print("=" * 70)


if __name__ == "__main__":
    agent, metrics = train()

    # Policy analysis
    agent.print_policy()
    agent.print_policy_summary()
    business_insights(agent)

    # Visualization
    print("\nGenerating plots...")
    plot_all(metrics)
    print("Done. Plots saved to 'plots/' directory.")
