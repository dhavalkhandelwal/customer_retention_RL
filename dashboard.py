"""
dashboard.py — Streamlit Interactive Dashboard for Customer Retention RL.

Run with:  streamlit run dashboard.py

Features:
  1. Hyperparameter tuning sidebar
  2. One-click training with live progress
  3. Training metric charts (Plotly)
  4. Interactive policy explorer — pick a customer profile, see the action
  5. Policy heatmap — engagement × churn risk
  6. Trained agent vs Random baseline ROI comparison
"""

import sys, os
import numpy as np
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from collections import Counter

# Make sibling modules importable
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils import (
    ENGAGEMENT_LEVELS, MONTHS_SINCE_PURCHASE, PURCHASE_FREQUENCY, CHURN_RISK,
    ACTIONS, NUM_ACTIONS, TOTAL_STATES,
    encode_state, decode_state, state_to_description, set_random_seed,
)
from environment import CustomerEnvironment
from q_learning_agent import QLearningAgent

# PAGE CONFIG
st.set_page_config(
    page_title="Customer Retention RL Dashboard",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)

# CUSTOM CSS
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Global */
    .stApp {
        font-family: 'Inter', sans-serif;
    }
    
    /* Main header */
    .main-header {
        background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
        padding: 2rem 2.5rem;
        border-radius: 16px;
        margin-bottom: 1.5rem;
        color: white;
        position: relative;
        overflow: hidden;
    }
    .main-header::before {
        content: '';
        position: absolute;
        top: -50%;
        right: -20%;
        width: 300px;
        height: 300px;
        background: radial-gradient(circle, rgba(99,102,241,0.25) 0%, transparent 70%);
        border-radius: 50%;
    }
    .main-header h1 {
        margin: 0;
        font-size: 1.8rem;
        font-weight: 700;
        letter-spacing: -0.5px;
    }
    .main-header p {
        margin: 0.4rem 0 0 0;
        opacity: 0.78;
        font-size: 0.95rem;
        font-weight: 300;
    }
    
    /* Metric cards */
    .metric-card {
        background: linear-gradient(135deg, #1e1e2f, #2a2a40);
        border: 1px solid rgba(255,255,255,0.06);
        padding: 1.3rem 1.5rem;
        border-radius: 14px;
        text-align: center;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .metric-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 25px rgba(0,0,0,0.3);
    }
    .metric-card .value {
        font-size: 2rem;
        font-weight: 700;
        background: linear-gradient(90deg, #6366f1, #a78bfa);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .metric-card .label {
        font-size: 0.82rem;
        color: #a0a0b8;
        margin-top: 0.3rem;
        text-transform: uppercase;
        letter-spacing: 0.8px;
    }
    
    /* Action badge */
    .action-badge {
        display: inline-block;
        padding: 0.5rem 1.2rem;
        border-radius: 30px;
        font-weight: 600;
        font-size: 1rem;
        letter-spacing: 0.3px;
    }
    .action-discount   { background: linear-gradient(135deg, #f43f5e, #e11d48); color: white; }
    .action-email      { background: linear-gradient(135deg, #3b82f6, #2563eb); color: white; }
    .action-loyalty     { background: linear-gradient(135deg, #10b981, #059669); color: white; }
    .action-nothing     { background: linear-gradient(135deg, #6b7280, #4b5563); color: white; }
    
    /* Section header */
    .section-header {
        font-size: 1.15rem;
        font-weight: 600;
        padding: 0.6rem 0;
        margin-top: 1rem;
        border-bottom: 2px solid rgba(99,102,241,0.3);
        margin-bottom: 1rem;
        letter-spacing: -0.2px;
    }
    
    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f0c29 0%, #1a1a2e 100%);
    }
    section[data-testid="stSidebar"] .stMarkdown {
        color: #e0e0f0;
    }
    
    /* Hide streamlit branding */
    #MainMenu { visibility: hidden; }
    footer { visibility: hidden; }
    
    /* Q-value table */
    .q-table {
        width: 100%;
        border-collapse: separate;
        border-spacing: 0;
        border-radius: 10px;
        overflow: hidden;
        font-size: 0.88rem;
    }
    .q-table th {
        background: #302b63;
        color: white;
        padding: 0.7rem 1rem;
        text-align: center;
    }
    .q-table td {
        padding: 0.6rem 1rem;
        text-align: center;
        border-bottom: 1px solid rgba(255,255,255,0.05);
    }
    .q-table tr:hover td {
        background: rgba(99,102,241,0.1);
    }
    .q-best { font-weight: 700; color: #818cf8; }
</style>
""", unsafe_allow_html=True)


# TRAINING FUNCTION (returns metrics progressively)
def run_training(num_episodes, alpha, gamma, eps_start, eps_min, eps_decay, seed=42):
    """Train Q-learning agent and return agent + metrics."""
    set_random_seed(seed)
    env = CustomerEnvironment()
    agent = QLearningAgent(
        alpha=alpha, gamma=gamma,
        epsilon=eps_start, epsilon_min=eps_min, epsilon_decay=eps_decay,
    )

    rewards, clvs, churns, epsilons = [], [], [], []
    progress = st.progress(0, text="Training...")

    for ep in range(1, num_episodes + 1):
        state = env.reset()
        total_reward = 0.0
        total_revenue = 0.0
        done = False
        churned = False

        while not done:
            action = agent.choose_action(state)
            next_state, reward, done, info = env.step(action)
            agent.update(state, action, reward, next_state, done)
            total_reward += reward
            if info["purchased"]:
                total_revenue += [5.0, 15.0, 30.0][(state % 9) // 3]
            if info["churned"]:
                churned = True
            state = next_state

        agent.decay_epsilon()
        rewards.append(total_reward)
        clvs.append(total_revenue)
        churns.append(1 if churned else 0)
        epsilons.append(agent.epsilon)

        if ep % max(1, num_episodes // 100) == 0:
            progress.progress(ep / num_episodes,
                              text=f"Episode {ep}/{num_episodes}  —  "
                                   f"Reward: {np.mean(rewards[-50:]):.1f}  |  "
                                   f"ε: {agent.epsilon:.3f}")

    progress.progress(1.0, text=" Training complete!")
    return agent, {"rewards": rewards, "clvs": clvs, "churns": churns, "epsilons": epsilons}


def evaluate_policy(agent, num_episodes=500, random_policy=False, seed=123):
    """Evaluate a policy (trained or random) over episodes; return avg reward, CLV, churn rate, total cost."""
    set_random_seed(seed)
    env = CustomerEnvironment()
    total_reward = 0; total_clv = 0; total_churns = 0; total_cost = 0
    action_costs = [3.0, 0.5, 1.5, 0.0]

    for _ in range(num_episodes):
        state = env.reset(); done = False
        while not done:
            if random_policy:
                action = np.random.randint(0, NUM_ACTIONS)
            else:
                action = int(np.argmax(agent.q_table[state]))
            next_state, reward, done, info = env.step(action)
            total_reward += reward
            total_cost += action_costs[action]
            if info["purchased"]:
                total_clv += [5.0, 15.0, 30.0][(state % 9) // 3]
            if info["churned"]:
                total_churns += 1
            state = next_state

    n = num_episodes
    return {
        "avg_reward": total_reward / n,
        "avg_clv": total_clv / n,
        "churn_rate": total_churns / n * 100,
        "avg_cost": total_cost / n,
        "net_profit": (total_clv - total_cost) / n,
    }


# PLOTTING HELPERS
def smooth(data, w=50):
    if len(data) < w:
        return list(range(1, len(data)+1)), data
    c = np.cumsum(np.insert(data, 0, 0))
    smoothed = (c[w:] - c[:-w]) / w
    return list(range(w, len(data)+1)), smoothed.tolist()


def make_metric_chart(raw, title, yaxis, color, smooth_color):
    x_raw = list(range(1, len(raw)+1))
    x_s, y_s = smooth(raw)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x_raw, y=raw, mode='lines',
                             line=dict(color=color, width=1), opacity=0.2, name='Raw'))
    fig.add_trace(go.Scatter(x=x_s, y=y_s, mode='lines',
                             line=dict(color=smooth_color, width=2.5), name='Smoothed'))
    fig.update_layout(
        title=dict(text=title, font=dict(size=15, family='Inter')),
        xaxis_title="Episode", yaxis_title=yaxis,
        template="plotly_dark",
        height=350,
        margin=dict(l=40, r=20, t=50, b=40),
        legend=dict(orientation='h', y=-0.18),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
    )
    return fig


def action_badge(action_name):
    css_map = {
        "Offer Discount": "action-discount",
        "Send Personalized Email": "action-email",
        "Give Loyalty Points": "action-loyalty",
        "Do Nothing": "action-nothing",
    }
    cls = css_map.get(action_name, "action-nothing")
    return f'<span class="action-badge {cls}">{action_name}</span>'


# SIDEBAR
with st.sidebar:
    st.markdown("##  Hyperparameters")
    num_episodes = st.slider("Training Episodes", 500, 5000, 2000, 100)
    alpha = st.slider("Learning Rate (α)", 0.01, 0.50, 0.10, 0.01)
    gamma = st.slider("Discount Factor (γ)", 0.80, 0.99, 0.95, 0.01)
    eps_start = st.slider("Initial Epsilon (ε)", 0.5, 1.0, 1.0, 0.05)
    eps_min = st.slider("Minimum Epsilon", 0.001, 0.10, 0.01, 0.005)
    eps_decay = st.slider("Epsilon Decay Rate", 0.990, 0.999, 0.997, 0.001)
    seed = st.number_input("Random Seed", value=42, step=1)

    st.markdown("---")
    train_btn = st.button("  Train Agent", use_container_width=True, type="primary")


# HEADER
st.markdown("""
<div class="main-header">
    <h1> Customer Retention RL Dashboard</h1>
    <p>Tabular Q-Learning for Customer Lifetime Value Optimization</p>
</div>
""", unsafe_allow_html=True)


# TRAINING
if train_btn:
    agent, metrics = run_training(num_episodes, alpha, gamma, eps_start, eps_min, eps_decay, int(seed))
    st.session_state["agent"] = agent
    st.session_state["metrics"] = metrics
    st.session_state["params"] = dict(alpha=alpha, gamma=gamma, eps_start=eps_start,
                                       eps_min=eps_min, eps_decay=eps_decay, episodes=num_episodes)


# MAIN CONTENT (only if trained)
if "agent" not in st.session_state:
    st.info(" Configure hyperparameters in the sidebar and click **Train Agent** to start.")
    st.stop()

agent = st.session_state["agent"]
metrics = st.session_state["metrics"]
params = st.session_state["params"]

# ---- Summary metrics ----
last_n = min(100, len(metrics["rewards"]))
avg_reward = np.mean(metrics["rewards"][-last_n:])
avg_clv = np.mean(metrics["clvs"][-last_n:])
churn_rate = np.mean(metrics["churns"][-last_n:]) * 100
final_eps = metrics["epsilons"][-1]

cols = st.columns(4)
card_data = [
    (f"{avg_reward:.1f}", "Avg Reward (last 100)"),
    (f"${avg_clv:.1f}", "Avg CLV (last 100)"),
    (f"{churn_rate:.1f}%", "Churn Rate (last 100)"),
    (f"{final_eps:.4f}", "Final Epsilon"),
]
for col, (val, label) in zip(cols, card_data):
    col.markdown(f"""
    <div class="metric-card">
        <div class="value">{val}</div>
        <div class="label">{label}</div>
    </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ---- Training Metric Charts ----
st.markdown('<div class="section-header"> Training Metrics</div>', unsafe_allow_html=True)

c1, c2 = st.columns(2)
with c1:
    st.plotly_chart(make_metric_chart(metrics["rewards"], "Reward per Episode", "Total Reward",
                                     "rgba(99,102,241,0.3)", "#6366f1"), use_container_width=True)
with c2:
    st.plotly_chart(make_metric_chart(metrics["clvs"], "Customer Lifetime Value", "CLV ($)",
                                     "rgba(16,185,129,0.3)", "#10b981"), use_container_width=True)

c3, c4 = st.columns(2)
with c3:
    churn_pct = [c * 100 for c in metrics["churns"]]
    st.plotly_chart(make_metric_chart(churn_pct, "Churn Rate (%)", "Churn %",
                                     "rgba(244,63,94,0.3)", "#f43f5e"), use_container_width=True)
with c4:
    st.plotly_chart(make_metric_chart(metrics["epsilons"], "Epsilon Decay", "ε",
                                     "rgba(168,85,247,0.3)", "#a855f7"), use_container_width=True)


# ---- Policy Explorer ----
st.markdown('<div class="section-header"> Interactive Policy Explorer</div>', unsafe_allow_html=True)
st.caption("Select a customer profile to see the recommended marketing action and Q-values.")

ec1, ec2, ec3, ec4 = st.columns(4)
with ec1:
    sel_eng = st.selectbox("Engagement", ENGAGEMENT_LEVELS, index=0)
with ec2:
    sel_msp = st.selectbox("Months Since Purchase", MONTHS_SINCE_PURCHASE, index=0)
with ec3:
    sel_freq = st.selectbox("Purchase Frequency", PURCHASE_FREQUENCY, index=0)
with ec4:
    sel_churn = st.selectbox("Churn Risk", CHURN_RISK, index=0)

eng_i = ENGAGEMENT_LEVELS.index(sel_eng)
msp_i = MONTHS_SINCE_PURCHASE.index(sel_msp)
freq_i = PURCHASE_FREQUENCY.index(sel_freq)
churn_i = CHURN_RISK.index(sel_churn)
selected_state = encode_state(eng_i, msp_i, freq_i, churn_i)

q_vals = agent.q_table[selected_state]
best_action_idx = int(np.argmax(q_vals))
best_action_name = ACTIONS[best_action_idx]

pc1, pc2 = st.columns([1, 2])
with pc1:
    st.markdown(f"**Recommended Action:**")
    st.markdown(action_badge(best_action_name), unsafe_allow_html=True)
    st.markdown(f"**State Index:** {selected_state}")

with pc2:
    # Q-value bar chart
    colors = ['#f43f5e', '#3b82f6', '#10b981', '#6b7280']
    highlight = ['rgba(244,63,94,1)' if i == best_action_idx else 'rgba(100,100,130,0.5)'
                 for i in range(4)]
    fig_q = go.Figure(go.Bar(
        x=ACTIONS, y=q_vals,
        marker_color=highlight,
        text=[f"{v:.2f}" for v in q_vals],
        textposition='outside',
    ))
    fig_q.update_layout(
        title="Q-Values for Selected State",
        yaxis_title="Q-Value", template="plotly_dark",
        height=300, margin=dict(l=40, r=20, t=50, b=40),
        plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
    )
    st.plotly_chart(fig_q, use_container_width=True)


# ---- Policy Heatmap ----
st.markdown('<div class="section-header"> Policy Heatmap — Engagement × Churn Risk</div>', unsafe_allow_html=True)
st.caption("Shows the dominant marketing action for each Engagement–Churn combination (aggregated over all Frequency & Recency values).")

hc1, hc2 = st.columns([3, 2])

with hc1:
    # Build heatmap data: for each (eng, churn), find the most common action
    action_to_num = {"Offer Discount": 0, "Send Personalized Email": 1,
                    "Give Loyalty Points": 2, "Do Nothing": 3}
    heatmap_data = np.zeros((3, 3))
    heatmap_text = []
    for eng in range(3):
        row_text = []
        for churn in range(3):
            actions_chosen = []
            for msp in range(3):
                for freq in range(3):
                    s = encode_state(eng, msp, freq, churn)
                    a = int(np.argmax(agent.q_table[s]))
                    actions_chosen.append(ACTIONS[a])
            dominant = Counter(actions_chosen).most_common(1)[0]
            heatmap_data[eng][churn] = action_to_num[dominant[0]]
            row_text.append(f"{dominant[0]}<br>({dominant[1]}/9)")
        heatmap_text.append(row_text)

    colorscale = [
        [0.0, '#f43f5e'], [0.33, '#3b82f6'],
        [0.66, '#10b981'], [1.0, '#6b7280']
    ]

    fig_hm = go.Figure(go.Heatmap(
        z=heatmap_data, x=CHURN_RISK, y=ENGAGEMENT_LEVELS,
        text=heatmap_text, texttemplate="%{text}",
        colorscale=colorscale, showscale=False,
        zmin=0, zmax=3,
    ))
    fig_hm.update_layout(
        xaxis_title="Churn Risk", yaxis_title="Engagement Level",
        template="plotly_dark", height=350,
        margin=dict(l=40, r=20, t=30, b=40),
        plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
    )
    st.plotly_chart(fig_hm, use_container_width=True)

with hc2:
    st.markdown("""
    **Reading the heatmap:**
    
    Each cell shows the **dominant action** the trained agent prefers for that combination of Engagement and Churn Risk (aggregated over all 9 Frequency × Recency sub-states).
    
    **Legend:**
    - 🔴 **Offer Discount** — costly but effective
    - 🔵 **Send Email** — low cost, moderate effect
    - 🟢 **Loyalty Points** — balanced retention tool
    - ⚫ **Do Nothing** — free, engagement may decay
    
    **Typical pattern:** The agent spends more on high-risk, low-engagement customers to prevent churn.
    """)


# ---- ROI Comparison ----
st.markdown('<div class="section-header"> Trained Agent vs Random Baseline — ROI Comparison</div>', unsafe_allow_html=True)

with st.spinner("Evaluating policies (500 episodes each)..."):
    trained_eval = evaluate_policy(agent, num_episodes=500, random_policy=False)
    random_eval = evaluate_policy(agent, num_episodes=500, random_policy=True)

rc1, rc2, rc3 = st.columns(3)

with rc1:
    fig_comp = go.Figure()
    fig_comp.add_trace(go.Bar(name='Trained', x=['Avg Reward', 'Avg CLV', 'Net Profit'],
                              y=[trained_eval['avg_reward'], trained_eval['avg_clv'], trained_eval['net_profit']],
                              marker_color='#6366f1'))
    fig_comp.add_trace(go.Bar(name='Random', x=['Avg Reward', 'Avg CLV', 'Net Profit'],
                              y=[random_eval['avg_reward'], random_eval['avg_clv'], random_eval['net_profit']],
                              marker_color='#4b5563'))
    fig_comp.update_layout(barmode='group', template='plotly_dark', height=320,
                           title="Reward & Revenue", margin=dict(l=40, r=20, t=50, b=40),
                           plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
    st.plotly_chart(fig_comp, use_container_width=True)

with rc2:
    fig_churn = go.Figure()
    fig_churn.add_trace(go.Bar(x=['Trained Agent', 'Random Policy'],
                               y=[trained_eval['churn_rate'], random_eval['churn_rate']],
                               marker_color=['#10b981', '#f43f5e'],
                               text=[f"{trained_eval['churn_rate']:.1f}%", f"{random_eval['churn_rate']:.1f}%"],
                               textposition='outside'))
    fig_churn.update_layout(title="Churn Rate Comparison", yaxis_title="Churn %",
                            template='plotly_dark', height=320,
                            margin=dict(l=40, r=20, t=50, b=40),
                            plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
    st.plotly_chart(fig_churn, use_container_width=True)

with rc3:
    fig_cost = go.Figure()
    fig_cost.add_trace(go.Bar(x=['Trained Agent', 'Random Policy'],
                              y=[trained_eval['avg_cost'], random_eval['avg_cost']],
                              marker_color=['#a855f7', '#6b7280'],
                              text=[f"${trained_eval['avg_cost']:.1f}", f"${random_eval['avg_cost']:.1f}"],
                              textposition='outside'))
    fig_cost.update_layout(title="Avg Marketing Cost", yaxis_title="Cost ($)",
                           template='plotly_dark', height=320,
                           margin=dict(l=40, r=20, t=50, b=40),
                           plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
    st.plotly_chart(fig_cost, use_container_width=True)


# ---- Business Insights ----
st.markdown('<div class="section-header"> Business Insights</div>', unsafe_allow_html=True)

policy = agent.get_policy()
action_counter = Counter(a_name for _, a_name in policy.values())

ic1, ic2 = st.columns([1, 1])

with ic1:
    st.markdown("**Action Distribution Across All 81 States:**")
    fig_pie = go.Figure(go.Pie(
        labels=list(action_counter.keys()),
        values=list(action_counter.values()),
        marker=dict(colors=['#f43f5e', '#3b82f6', '#10b981', '#6b7280']),
        hole=0.45,
        textinfo='label+percent',
        textfont=dict(size=12),
    ))
    fig_pie.update_layout(template='plotly_dark', height=350,
                          margin=dict(l=20, r=20, t=20, b=20),
                          plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                          showlegend=False)
    st.plotly_chart(fig_pie, use_container_width=True)

with ic2:
    improvement = ((trained_eval['avg_clv'] - random_eval['avg_clv'])
                   / max(random_eval['avg_clv'], 0.01) * 100)
    churn_reduction = random_eval['churn_rate'] - trained_eval['churn_rate']

    st.markdown(f"""
    **Key Findings:**
    
     **CLV Improvement:** The trained agent achieves **{improvement:.0f}% higher CLV** than a random policy.
    
     **Churn Reduction:** Churn rate reduced by **{churn_reduction:.1f} percentage points** compared to random.
    
     **Smart Spending:** The agent learns to invest in costly actions (discounts, loyalty points) **only** for at-risk customers, while using cheaper strategies for already-engaged ones.
    
     **Long-term Focus:** The discount factor γ={params['gamma']} makes the agent value future revenue, spending upfront to prevent the −$50 churn penalty.
    
    ---
    
    **Training Config Used:**
    - α = {params['alpha']} · γ = {params['gamma']} · ε = {params['eps_start']}→{params['eps_min']}
    - Episodes: {params['episodes']} · Decay: {params['eps_decay']}
    """)


# ---- Footer ----
st.markdown("---")
st.caption(" Reinforcement Learning Based Customer Lifetime Value Optimization using Q-Learning")
