# Workplan: GLE x MHSA Hybrid Learning for Visual Search

## Objective
To implement a biologically plausible, reinforcement learning-capable visual search agent by combining **Cortico-Thalamic Multi-Head Self-Attention (CT-MHSA)** with **Generalized Lagrangian Energy (GLE)** dynamics.

The core strategy is a **Hybrid Two-Phase Learning** approach:
1.  **Phase 1 (Development):** Supervised Backpropagation Pretraining on a convex "Fixation" task to establish perceptual representations and basic motor control.
2.  **Phase 2 (Learning):** Reward-Modulated Hebbian Learning (GLE 3-Factor Rule) on the complex "Visual Search" task to learn sequential policy and strategy.

---

## Phase 1: Supervised Pretraining (The "Developmental" Phase)

**Goal:** Train the Retina (ConvNet), Core Projections, and Saccade Head to map visual inputs to correct eye movements reliably.

### 1.1. Simplified Task: "Fixate"
*   **Environment:** Static images with a single salient target.
*   **Input:** Retinal patch centered at random location.
*   **Target:** A 2D vector $(dx, dy)$ pointing to the target center.
*   **Loss:** MSE between predicted saccade and target vector.

### 1.2. Implementation (JAX/Optax)
*   **Script:** `train_fixation.py`
*   **Model:** `VisualSearchParams` (Retina + CT-MHSA + Saccade Head).
*   **Training Loop:**
    *   Standard Gradient Descent (Adam).
    *   **Freeze:** Answer Head, Value Head (not needed yet).
    *   **Learn:** Retina Conv layers, Core ($W_Q, W_K, W_V, W_Y, C$), Saccade Head.
*   **Outcome:** A saved checkpoint `pretrained_fixation.msgpack` where the agent can reliably look at salient objects.

---

## Phase 2: GLE-RL Fine-tuning (The "Cognitive" Phase)

**Goal:** Train the agent to perform multi-step visual search (sequence of saccades) using biologically plausible local learning rules.

### 2.1. Task: "Visual Search"
*   **Environment:** Images with 1 Target and N Distractors.
*   **Loop:**
    1.  Observe patch.
    2.  Saccade (Agent moves window).
    3.  Repeat until Target found or Timeout.
*   **Reward:**
    *   $+1$ if Fixation lands on Target (Threshold distance).
    *   $-0.1$ per step (Time cost).
    *   $-1$ if Timeout.

### 2.2. The "Triad" Architecture (GLE-RL)
We introduce **Eligibility Traces** ($E$) for the learnable weights.

*   **Weight Sets to Train:**
    *   `Heads` (Saccade, Answer, Value).
    *   `Core` (Output Projection $W_Y$, Key/Query/Value Projections).
    *   *Note:* Retina layers remain frozen (or learn very slowly).

*   **Trace Dynamics (Fast Loop - "Action"):**
    During the episode (fixation sequence), we do *not* update weights. We accumulate activity correlations:
    $$ E_{ij}(t) = \alpha E_{ij}(t-1) + \text{Pre}_j(t) \times \text{ErrorPost}_i(t) $$
    *   For **Heads:** $\text{ErrorPost}$ is the localized activity (or proxy error from Value head).
    *   For **Core:** $\text{ErrorPost}$ is the GLE `prosp_v` (error potential) propagated from the heads.

*   **Weight Update (Slow Loop - "Plasticity"):**
    Upon Reward $R$ (or continuously via TD-error $\delta$):
    $$ \Delta W_{ij} = \eta \cdot (R - \bar{R}) \cdot E_{ij} $$

### 2.3. Implementation Steps

#### Step 1: Augment Model State
*   Modify `NetworkState` in `ct_mhsa.py` or wrap it to include `traces` for relevant parameters ($E_{W_Q}, E_{W_Y}, \dots$).

#### Step 2: Implement "Trace Step"
*   Create `gle_step()` function.
*   Instead of `grad()`, compute the **Hebbian Product** manually during the forward pass.
*   Store this in the `traces`.

#### Step 3: Implement "Reward Update"
*   Create `apply_reward()` function.
*   Takes `params`, `traces`, and `reward`.
*   Updates `params` using the 3-Factor rule.

#### Step 4: The Experiment Script (`train_search_gle.py`)
1.  Load `pretrained_fixation.msgpack`.
2.  Initialize Traces to zero.
3.  Run Episode:
    *   `agent_step_gle(...)` $\to$ Returns action + updates traces.
    *   Environment Step $\to$ Reward.
    *   `apply_reward(...)` $\to$ Updates weights.
4.  Log performance (Success Rate, Steps to Find).

---

## Phase 3: Evaluation & Analysis

*   **Compare:** GLE-RL Agent vs. PPO/Reinforce Agent (Backprop baseline).
*   **Metric:** Sample Efficiency (How many trials to learn?) and Robustness.
*   **Biological Validation:** Does the "Attention" (Saccades) mimic human scanpaths? Do the traces resemble dopamine-modulated plasticity?

---

## File Structure Proposal

```text
vbjax/
  app/
    visual_search/
      train_fixation.py       # Phase 1: Supervised Backprop
      train_search_gle.py     # Phase 2: GLE-RL
      gle_utils.py            # Helper functions for Traces/Hebbian rules
      model.py                # Existing model definition
```

```