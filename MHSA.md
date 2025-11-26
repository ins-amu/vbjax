Here is a comprehensive work plan designed for a coding agent to implement the Cortico-Thalamic MHSA model in JAX.

### Project: Cortico-Thalamic MHSA Implementation in JAX

This plan targets **80% of the mathematical framework** defined in the provided PDF, specifically the recurrent key-value memory (L2/3), query retrieval (L5), and structural coupling between regions.

---

#### Phase 1: Data Structures & Initialization
**Goal:** Define the static configuration and mutable state containers using JAX/Flax patterns to handle the high-dimensionality ($N_{regions} \times N_{heads} \times D_{key} \times D_{value}$).

- [ ] **Define Hyperparameters Class**
    - [ ] Set dimensions: $N=84$ (regions), $H=8$ (heads), $D_k=16$, $D_v=16$ (feature dims).
    - [ ] [cite_start]Define $\lambda$ (forgetting factor) and $\overline{\lambda} = 1 - \lambda$[cite: 59, 65].
    - [ ] Define sequence length $T$ and batch size $B$.
- [ ] **Define Network State Class (`PyTree`)**
    - [ ] Initialize Memory $M$ (Layer 2/3) with shape $(B, N, H, D_v, D_k)$.
    - [ ] **Critical:** Initialize $M$ to zero or small variance to prevent early explosion (addressing the instability seen in the notebook).
- [ ] **Define Learnable Parameters (`PyTree`)**
    - [ ] [cite_start]Initialize $W_Q, W_K, W_V$ per head[cite: 54, 73].
    - [ ] [cite_start]Initialize $W_Y$ (Readout) per head[cite: 76].
    - [ ] Initialize Structural Connectivity $C$ (fixed or learnable mask).
- [ ] **Validation Step 1**
    - [ ] **Test:** Instantiate the classes and verify `jax.tree_util.tree_structure` outputs expected hierarchy.
    - [ ] **Test:** Verify VRAM usage for batch size=1 to ensure $M$ fits in memory (approx $84 \times 8 \times 16 \times 16$ floats per sample).

---

#### Phase 2: The Cortical Microcircuit (The "Head")
[cite_start]**Goal:** Implement the "Delta Rule" recurrence [cite: 67] and L2/3 vs L5 separation within a single time step.

- [ ] **Implement `compute_projections`**
    - [ ] [cite_start]Calculate $q_t, k_t, v_t$ using linear projections of input $x_t$[cite: 54, 73].
    - [ ] [cite_start]Apply activation $\phi$ (e.g., identity or ReLU) to $k$ and $q$[cite: 64].
- [ ] **Implement `update_memory_l23` (Delta Rule)**
    - [ ] [cite_start]Implement Eq 2: $M_t^h = M_{t-1}^h + \lambda^h (v_t^h \phi(k_t^h)^T - M_{t-1}^h)$[cite: 67].
    - [ ] *Constraint:* Ensure tensor contractions use `jax.numpy.einsum` for correct head broadcasting.
- [ ] **Implement `retrieve_query_l5`**
    - [ ] [cite_start]Implement Eq 6: $o_t^h = M_t^h \phi(q_t^h)$[cite: 112].
    - [ ] This represents the L5 pyramidal retrieval step.
- [ ] **Implement `aggregate_heads`**
    - [ ] [cite_start]Implement Eq 3: $y_t = \sum_h W_Y^h o_t^h$[cite: 76].
    - [ ] Sum across the $H$ dimension to flatten back to $(B, N, D_{out})$.
- [ ] **Validation Step 2**
    - [ ] **Test:** Check "Needle in a Haystack" logic on a single head:
        - Feed a distinct $(k, v)$ pair at $t=0$.
        - Feed empty inputs for $t=1..10$.
        - Query with $q \approx k$ at $t=11$.
        - Assert output matches $v$ (decayed by $\lambda^{10}$).
    - **Test:** Verify gradients flow through $M_t$ to $M_{t-1}$ (autodiff check).

---

#### Phase 3: The Connectome & Time Loop
**Goal:** Integrate the structural connectivity matrix and unroll the system over time using `jax.lax.scan`.

- [ ] **Implement `network_coupling`**
    - [ ] Define the transmission mechanism: $x_{t+1, i} = \sum_j C_{ij} y_{t, j}$ + External Input.
    - [ ] [cite_start]Ensure $C$ is treated as the "transcortical pathway"[cite: 228].
- [ ] **Implement `step_fn` for `jax.lax.scan`**
    - [ ] Bundle Phase 2 and Coupling into a single functional step: $(State_{t-1}, Input_t) \rightarrow (State_t, Output_t)$.
- [ ] **Implement `rollout`**
    - [ ] Use `jax.lax.scan` to process a sequence of inputs.
- [ ] **Validation Step 3**
    - [ ] **Test:** Conservation of Energy/Signal. Run for T=100 with identity $C$ and no decay. Check if values explode (NaNs) or vanish.
    - [ ] **Test:** Comparison against reference Python loop (for T=5). Ensure `jax.lax.scan` output matches numerically.

---

#### Phase 4: Training & Optimization
**Goal:** Implement the predictive coding loss and stabilize training.

- [ ] **Implement Loss Function (Predictive Coding)**
    - [ ] [cite_start]Implement Eq 27: $E_t = \frac{1}{2} || x_{t+1} - y_t ||^2$[cite: 611].
    - [ ] *Note:* Since this is a recurrence, we predict the *next* network input or a shifted target sequence.
- [ ] **Setup Optimizer (`optax`)**
    - [ ] Configure AdamW.
    - [ ] **Critical:** Implement a learning rate warmup schedule (crucial for linear attention stability).
    - [ ] Gradient clipping (global norm) to handle RNN gradients.
- [ ] **Implement Training Step**
    - [ ] `jax.value_and_grad` on the rollout function.
    - [ ] `optax.apply_updates`.
- [ ] **Validation Step 4**
    - [ ] **Test:** Overfitting check. Train on a single batch of random data. Loss should approach zero.
    - [ ] **Test:** Stability check. Train for 100 steps. Monitor $M$ statistics (mean/std). If diverging, adjust initialization variance of $W$ parameters.

---

#### Phase 5: "Tiny Shakespeare" Reproduction (Optional / Reach Goal)
**Goal:** replicate the validation task mentioned in the PDF (Fig 4) to prove capability.

- [ ] **Data Pipeline**
    - [ ] [cite_start]Create a simple char-level tokenizer/loader for Tiny Shakespeare text[cite: 284].
- [ ] **Experiment Runner**
    - [ ] Run the training loop on the text sequence.
    - [ ] Log training loss vs validation loss.
- [ ] **Validation Step 5**
    - [ ] **Qualitative:** Sample text generation. Does it output coherent character sequences?
    - [ ] [cite_start]**Quantitative:** Compare loss curve to [cite: 240] (Fig 4a).

---

### Critical Engineering Notes for the Agent
1.  **Einsum Notation:** Always use `einsum` for the linear algebra to keep dimensions $(B, N, H, D)$ clear.
    * *Example Memory Update:* `v_t (B, N, H, D_v), k_t (B, N, H, D_k) -> Outer Product (B, N, H, D_v, D_k)`.
2.  **JIT Compilation:** Ensure all functions in Phases 2-4 are pure and JIT-compatible.
3.  **Numerical Stability:** The PDF notes specific forgetting factors $\lambda$. [cite_start]If $\lambda=1$ (no decay), the memory integrates infinitely[cite: 78]. [cite_start]Ensure $\lambda < 1$ (e.g., 0.9 or 0.99) for stability[cite: 587].