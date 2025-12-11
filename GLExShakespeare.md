# Workplan: GLE-Driven Shakespeare (Language Modeling)

## Objective
To train a **Cortico-Thalamic Multi-Head Self-Attention (CT-MHSA)** model on the Shakespeare character prediction task using **biologically plausible, local Generalized Lagrangian Energy (GLE)** learning rules instead of Backpropagation Through Time (BPTT).

## Strategy
We will replace the standard JAX automatic differentiation (`jax.value_and_grad`) with explicit forward-mode error propagation and Hebbian gradient accumulation. This effectively turns the Transformer into a Deep Predictive Coding network.

---

## Phase 1: Core Implementation (`vbjax/ct_mhsa_gle.py`)

We need a new module that mirrors `ct_mhsa.py` but exposes the internal error dynamics.

### 1.1. Data Structures
*   Extend `NetworkState` to include `prosp_v` (error potential) and `prosp_u` (membrane potential) for all relevant nodes (Keys, Queries, Values, Output).
*   Create a `GLEGradients` struct to accumulate $\Delta W$ manually.

### 1.2. The `GLE_MHSA_Layer` Class
Instead of functional `mhsa_step`, we implement a class or closure-based structure that handles:
1.  **Forward Pass (Activity):** $u_{post} \leftarrow W u_{pre}$.
2.  **Error Injection (Feedback):** $v_{pre} \leftarrow W^T v_{post}$ (Implicit or Explicit).
3.  **Plasticity (Hebbian):** $\Delta W \propto v_{post} \otimes u_{pre}$.

**Key Challenge:** The "Fast Weights" ($M$) in MHSA are already dynamic. We must differentiate between:
*   **Learning $M$:** Done via the existing Delta Rule (Fast).
*   **Learning $W_{Q,K,V}$:** Done via GLE Error Propagation (Slow).

### 1.3. Dynamics Equations
For every projection $y = Wx$:
*   **Activity:** $\dot{u} = (-u + Wx + \gamma v) / \tau$
*   **Error:** $\dot{v} = (-v + e_{in}) / \tau$
*   **Update:** $\dot{W} = \eta \cdot v \cdot x^T$

---

## Phase 2: Experiment Script (`examples/shakespeare_ct_mhsa_gle.py`)

### 2.1. Architecture
*   **Topology:** Same 8-Region Diamond/Hierarchical topology as the original.
*   **Input:** Region 0.
*   **Output:** Region 7.

### 2.2. The Training Loop (Manual Scan)
Instead of `optax` and `loss_fn`:
1.  **Initialize:** Weights, State, and empty Gradients.
2.  **Sequence Loop (Scan):**
    *   **Input:** Inject character embedding at Region 0.
    *   **Target:** Calculate error vector at Region 7: $e = \text{OneHot}(char_{t+1}) - r_{out}$.
    *   **Step:** Run `gle_step`. This propagates signal forward and error backward *simultaneously* (or phased).
    *   **Accumulate:** Update `GLEGradients`.
3.  **Weight Update:** Apply gradients to $W$ matrices at end of sequence or batch.

### 2.3. Metrics
*   Track **Cross-Entropy Loss** (for monitoring only, not training).
*   Track **Gradient Norms** (to ensure signal isn't vanishing).

---

## Phase 3: Validation

### 3.1. Sanity Check
*   Train on a tiny sequence ("abcabc...") first.
*   Verify that error drops to zero.

### 3.2. Scaling
*   Train on full Shakespeare snippet.
*   Compare Loss curve vs. BPTT baseline (`shakespeare_ct_mhsa.py`).
*   **Expectation:** GLE will likely be slower to converge but should still learn valid structure.

---

## Roadmap

1.  [ ] Create `vbjax/ct_mhsa_gle.py` (The GLE-ified logic).
2.  [ ] Create `examples/shakespeare_ct_mhsa_gle.py` (The harness).
3.  [ ] Debug on toy data.
4.  [ ] Run full experiment.
