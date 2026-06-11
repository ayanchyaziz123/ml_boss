# Reinforcement Learning — Interview Questions & Answers

---

## Small Dataset Example (Grid World)

```
4×4 Grid World:
S = Start, G = Goal (+10 reward), H = Hole (-10 reward), . = Empty (−0.1)

[ S  .  .  . ]
[ .  H  .  H ]
[ .  .  .  H ]
[ H  .  .  G ]

Actions: Up, Down, Left, Right
Discount factor γ = 0.9
Episode ends when agent reaches G or H
```

### Dataset Questions

**Q: The agent is at position (0,0)=S. It takes action Right and gets reward -0.1. Then moves Right again, reward -0.1. Then Down twice, and reaches G for +10. What is the total return (undiscounted)?**
> **A:** Return = -0.1 + (-0.1) + (-0.1) + (-0.1) + 10 = **9.6**

**Q: With γ=0.9, what is the discounted return for the same trajectory (4 steps then reward +10)?**
> **A:** G = -0.1 + 0.9(-0.1) + 0.9²(-0.1) + 0.9³(-0.1) + 0.9⁴(10) = -0.1 - 0.09 - 0.081 - 0.073 + 0.6561 = **6.561 − 0.344 = ~6.21**

**Q: Why do we use a discount factor γ < 1?**
> **A:** γ < 1 ensures the sum of infinite future rewards converges (mathematical necessity for infinite-horizon problems). It also encodes the preference for immediate rewards over distant future rewards — a reward now is more certain than a reward far in the future. Higher γ = more far-sighted agent.

---

## Questions & Answers

---

### Q1. What is Reinforcement Learning?

**A:** Reinforcement Learning (RL) is a machine learning paradigm where an **agent** learns to make decisions by interacting with an **environment** to maximize cumulative **reward**.

No labeled data — the agent learns from trial and error: take action → observe reward and next state → update policy. Used in games (AlphaGo), robotics, recommendation systems, and LLM alignment (RLHF).

---

### Q2. What is an agent and environment?

**A:**
- **Agent:** The learner/decision-maker that takes actions
- **Environment:** Everything outside the agent that it interacts with
- **State (sₜ):** Current situation the agent observes
- **Action (aₜ):** Decision taken by agent
- **Reward (rₜ):** Scalar feedback signal from environment
- **Next state (sₜ₊₁):** New state after action

The agent-environment loop: Agent observes state → takes action → environment returns reward + next state.

---

### Q3. What is a reward?

**A:** A reward is a scalar signal rₜ that tells the agent how good its last action was. The agent's goal is to maximize **cumulative discounted return**:

```
G_t = rₜ + γrₜ₊₁ + γ²rₜ₊₂ + ... = Σ γᵏrₜ₊ₖ
```

Reward design (reward shaping) is critical and tricky — the agent will find unexpected ways to maximize reward (Goodhart's Law).

---

### Q4. What is a policy?

**A:** A policy π maps states to actions:
- **Deterministic:** π(s) = a — one action per state
- **Stochastic:** π(a|s) = P(a|s) — probability distribution over actions

The goal is to find the **optimal policy** π* that maximizes expected return from any state.

---

### Q5. What is a value function?

**A:** The value function estimates the expected cumulative return from a state:

```
V^π(s) = E_π[G_t | sₜ = s]
```

The **Q-function (action-value function)** estimates return from a state-action pair:
```
Q^π(s, a) = E_π[G_t | sₜ = s, aₜ = a]
```

The optimal Q-function satisfies: `π*(s) = argmax_a Q*(s, a)`

---

### Q6. What is Q-learning?

**A:** Q-learning is a model-free, off-policy algorithm that learns the optimal Q-function:

```
Q(s, a) ← Q(s, a) + α[r + γ max_a' Q(s', a') − Q(s, a)]
```

- Current Q estimate is updated toward the **Bellman target**: `r + γ max Q(s', a')`
- **Off-policy:** Can learn from any experience, not just current policy
- Converges to Q* regardless of what policy generated the data (with enough exploration)

---

### Q7. What is the Bellman equation?

**A:** The Bellman equation is a recursive relationship for the optimal Q-function:

```
Q*(s, a) = E[r + γ max_a' Q*(s', a') | s, a]
```

"The value of (s,a) = immediate reward + discounted value of the best next action"

Q-learning minimizes the Bellman error (TD error):
```
TD error = r + γ max_a' Q(s', a') − Q(s, a)
```

---

### Q8. What is exploration vs exploitation?

**A:**
- **Exploitation:** Choose the action with highest current Q-value (greedy) — use what you know
- **Exploration:** Try random or new actions — discover potentially better strategies

Too much exploitation: stuck in suboptimal policy. Too much exploration: never exploits learned knowledge.

**Strategies:**
- **ε-greedy:** Explore with probability ε, exploit with 1−ε
- **UCB:** Choose action with upper confidence bound on reward
- **Thompson Sampling:** Sample from posterior reward estimate

---

### Q9. What is epsilon-greedy?

**A:**
```
With probability ε:  choose random action (exploration)
With probability 1−ε: choose argmax Q(s,a) (exploitation)
```

Typical schedule: Start with ε=1.0 (pure exploration), decay to ε=0.01 (mostly exploitation).

```python
if np.random.random() < epsilon:
    action = env.action_space.sample()  # explore
else:
    action = np.argmax(Q[state])        # exploit
```

---

### Q10. What is Deep Q-Network (DQN)?

**A:** DQN (Mnih et al., 2015) uses a neural network to approximate the Q-function:

```
Q(s, a; θ) ≈ Q*(s, a)
```

Key innovations:
1. **Experience Replay:** Store transitions (s,a,r,s') in buffer, sample random mini-batches → breaks correlation between sequential updates
2. **Target Network:** Separate frozen network for computing Bellman targets → stabilizes training
3. Trained using: `L = E[(r + γ max_a' Q(s',a'; θ⁻) − Q(s,a; θ))²]`

---

### Q11. What is policy gradient?

**A:** Policy gradient methods directly optimize the policy by maximizing expected return:

```
∇_θ J(π_θ) = E[∇_θ log π_θ(a|s) × G_t]
```

**REINFORCE algorithm:**
1. Run episode with current policy
2. Compute returns G_t at each step
3. Update: `θ = θ + α × ∇_θ log π_θ(aₜ|sₜ) × Gₜ`

High variance (full return used as signal). Baseline subtraction (advantage function) reduces variance.

---

### Q12. What is the actor-critic method?

**A:** Actor-Critic combines policy gradient (actor) with value function estimation (critic):

- **Actor:** Learns policy π_θ(a|s) — decides actions
- **Critic:** Learns value function V_w(s) — evaluates how good current state is

```
Advantage A(s, a) = Q(s, a) − V(s) ≈ r + γV(s') − V(s)  (TD error)
Actor update: θ = θ + α × ∇ log π(a|s) × A(s,a)
Critic update: w = w + β × (r + γV(s') − V(s)) × ∇V_w(s)
```

Reduces variance of REINFORCE while remaining unbiased.

---

### Q13. What is PPO (Proximal Policy Optimization)?

**A:** PPO is the most widely used policy gradient algorithm. It prevents destructively large policy updates:

```
L_clip = E[min(r_t(θ) × A_t, clip(r_t(θ), 1−ε, 1+ε) × A_t)]

r_t(θ) = π_θ(a|s) / π_θ_old(a|s)  (probability ratio)
```

- If the new policy is too different (ratio outside [1-ε, 1+ε]), clip the gradient
- ε = 0.2 is typical
- Used in ChatGPT training (RLHF stage), robotics, game playing

---

### Q14. What is model-based vs model-free RL?

**A:**
| | Model-Free | Model-Based |
|---|---|---|
| Learns | Policy/value function from interaction | A model of environment dynamics |
| Sample efficiency | Lower (many episodes needed) | Higher |
| Computation | Less (no planning) | More (simulate with model) |
| Accuracy | Exact environment | Model approximation error |
| Examples | Q-learning, PPO, SAC | AlphaZero, Dreamer, MuZero |

Model-based: learn p(s'|s,a) and r(s,a), then plan with it. Sample efficient but model errors compound.

---

### Q15. What is the discount factor γ?

**A:** γ ∈ [0, 1) discounts future rewards:
- γ = 0: Agent only cares about immediate reward (myopic)
- γ = 1: Agent cares equally about all future rewards (farsighted, may not converge)
- γ = 0.99: Common value — future reward 100 steps away worth 0.99^100 ≈ 0.37 of immediate

Mathematical purpose: ensures infinite-horizon returns converge: G = r/(1−γ) for constant r.

---

### Q16. What is a Markov Decision Process (MDP)?

**A:** An MDP formally defines the RL problem: (S, A, P, R, γ)

- **S:** State space
- **A:** Action space
- **P(s'|s,a):** Transition probability (environment dynamics)
- **R(s,a):** Reward function
- **γ:** Discount factor

**Markov property:** Future depends only on current state, not history: P(sₜ₊₁|sₜ, aₜ) = P(sₜ₊₁|s₀,...,sₜ, a₀,...,aₜ)

---

### Q17. What is on-policy vs off-policy?

**A:**
- **On-policy:** Learn value function for the policy currently being followed
  - SARSA, PPO — can only learn from current policy's experience
  - More stable but less sample efficient

- **Off-policy:** Learn value function for target policy different from behavior policy
  - Q-learning, DQN, SAC — can learn from any stored experience (experience replay)
  - More sample efficient, can reuse old data

---

### Q18. What is Monte Carlo vs Temporal Difference learning?

**A:**
| | Monte Carlo | Temporal Difference |
|---|---|---|
| Update timing | After episode ends | After each step |
| Uses | Full return Gₜ | Bootstrap estimate r + γV(s') |
| Variance | High | Lower |
| Bias | None | Some (bootstrapping) |
| Episodes | Must be finite | Works for continuous tasks |
| Example | REINFORCE | Q-learning, SARSA |

---

### Q19. What is reward shaping?

**A:** Reward shaping adds extra reward signals to guide learning when sparse rewards make training slow:

- **Sparse:** Only +1 for reaching goal (hard to learn)
- **Shaped:** +0.1 for moving closer to goal (easier to learn)

**Risk:** Agent may exploit the shaped reward in unintended ways. Potential-based shaping is provably safe (doesn't change optimal policy): `F(s,s') = γΦ(s') − Φ(s)`.

---

### Q20. What is the difference between SAC and PPO?

**A:**
| | SAC (Soft Actor-Critic) | PPO |
|---|---|---|
| Type | Off-policy, max-entropy RL | On-policy |
| Sample efficiency | Very high | Lower |
| Stability | Very stable | Very stable |
| Entropy | Maximizes entropy (exploration) | No explicit entropy bonus |
| Continuous actions | Excellent | Good |
| Discrete actions | Possible | Natural |
| Hyperparameters | More sensitive | Robust |
| Best for | Continuous control (robotics, MuJoCo) | General RL, RLHF |

---

## Quick Code Example (Q-Learning on Grid World)

```python
import numpy as np

# 4x4 GridWorld: 0=empty, -10=hole, +10=goal
grid = np.array([
    [ 0,  0,  0,  0],
    [ 0,-10,  0,-10],
    [ 0,  0,  0,-10],
    [-10,  0,  0, 10]
])
ROWS, COLS = 4, 4
ACTIONS = [(-1,0),(1,0),(0,-1),(0,1)]  # up,down,left,right

Q = np.zeros((ROWS*COLS, 4))  # Q-table: states × actions
alpha, gamma, epsilon = 0.1, 0.9, 1.0

def state(r, c): return r * COLS + c
def is_terminal(r, c): return grid[r][c] != 0

for episode in range(5000):
    r, c = 0, 0  # start
    epsilon = max(0.01, epsilon * 0.999)  # decay epsilon

    for _ in range(100):
        s = state(r, c)
        if np.random.random() < epsilon:
            a = np.random.randint(4)
        else:
            a = np.argmax(Q[s])

        dr, dc = ACTIONS[a]
        nr, nc = max(0, min(ROWS-1, r+dr)), max(0, min(COLS-1, c+dc))
        reward = grid[nr][nc] if grid[nr][nc] != 0 else -0.1
        ns = state(nr, nc)

        # Q-learning update
        Q[s, a] += alpha * (reward + gamma * np.max(Q[ns]) - Q[s, a])

        r, c = nr, nc
        if is_terminal(r, c):
            break

# Print optimal policy
arrows = ['↑','↓','←','→']
for r in range(ROWS):
    row = ""
    for c in range(COLS):
        if grid[r][c] == 10:  row += " G "
        elif grid[r][c] == -10: row += " H "
        else: row += f" {arrows[np.argmax(Q[state(r,c)])]} "
    print(row)
```
