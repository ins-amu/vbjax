Here is a comprehensive work plan designed for a coding agent to implement the Cortico-Thalamic MHSA model in JAX.

### Project: Cortico-Thalamic MHSA Implementation in JAX

This plan targets **80% of the mathematical framework** defined in the provided PDF, specifically the recurrent key-value memory (L2/3), query retrieval (L5), and structural coupling between regions.

---

#### Phase 1: Data Structures & Initialization
**Goal:** Define the static configuration and mutable state containers using JAX/Flax patterns to handle the high-dimensionality ($N_{regions} 	imes N_{heads} 	imes D_{key} 	imes D_{value}$). 

- [x] **Define Hyperparameters Class** (Implemented as `Hyperparameters` in `vbjax/ct_mhsa.py`)
    - [x] Set dimensions: $N=84$ (regions), $H=8$ (heads), $D_k=16$, $D_v=16$ (feature dims). Also added `d_model`.
    - [x] [cite_start]Define $\lambda$ (forgetting factor) and $\overline{\lambda} = 1 - \lambda$[cite: 59, 65].
    - [x] Define sequence length $T$ and batch size $B$. (Handled implicitly by `jax.lax.scan` inputs and `init_ct_mhsa` batch_size argument).
- [x] **Define Network State Class (`PyTree`)** (Implemented as `NetworkState` in `vbjax/ct_mhsa.py`)
    - [x] Initialize Memory $M$ (Layer 2/3) with shape $(B, N, H, D_v, D_k)$.
    - [x] **Critical:** Initialize $M$ to zero or small variance to prevent early explosion (addressed by `np.zeros`).
- [x] **Define Learnable Parameters (`PyTree`)** (Implemented as `CTMHSAParams` in `vbjax/ct_mhsa.py`)
    - [x] [cite_start]Initialize $W_Q, W_K, W_V$ per head[cite: 54, 73].
    - [x] [cite_start]Initialize $W_Y$ (Readout) per head[cite: 76].
    - [x] Initialize Structural Connectivity $C$ (fixed or learnable mask).
- [x] **Validation Step 1** (Tested in `vbjax/tests/test_ct_mhsa.py`)
    - [x] **Test:** Instantiate the classes and verify `jax.tree_util.tree_structure` outputs expected hierarchy.
    - [x] **Test:** Verify VRAM usage for batch size=1 to ensure $M$ fits in memory (approx $84 	imes 8 	imes 16 	imes 16$ floats per sample). (Conceptual check performed in test).

---

#### Phase 2: The Cortical Microcircuit (The "Head")
[cite_start]**Goal:** Implement the "Delta Rule" recurrence [cite: 67] and L2/3 vs L5 separation within a single time step. (Implemented in `vbjax/ct_mhsa.py`)

- [x] **Implement `compute_projections`**
    - [x] [cite_start]Calculate $q_t, k_t, v_t$ using linear projections of input $x_t$[cite: 54, 73].
    - [x] [cite_start]Apply activation $\phi$ (e.g., identity or ReLU) to $k$ and $q$[cite: 64]. (ReLU used).
- [x] **Implement `update_memory_l23` (Delta Rule)**
    - [x] [cite_start]Implement Eq 2: $M_t^h = M_{t-1}^h + \lambda^h (v_t^h \phi(k_t^h)^T - M_{t-1}^h)$[cite: 67].
    - [x] *Constraint:* Ensure tensor contractions use `jax.numpy.einsum` for correct head broadcasting.
- [x] **Implement `retrieve_query_l5`**
    - [x] [cite_start]Implement Eq 6: $o_t^h = M_t^h \phi(q_t^h)$[cite: 112].
    - [x] This represents the L5 pyramidal retrieval step.
- [x] **Implement `aggregate_heads`**
    - [x] [cite_start]Implement Eq 3: $y_t = \sum_h W_Y^h o_t^h$[cite: 76].
    - [x] Sum across the $H$ dimension to flatten back to $(B, N, D_{out})$.
- [x] **Validation Step 2** (Tested in `vbjax/tests/test_ct_mhsa.py`)
    - [x] **Test:** Check "Needle in a Haystack" logic on a single head.
    - [x] **Test:** Verify gradients flow through $M_t$ to $M_{t-1}$ (autodiff check).

---

#### Phase 3: The Connectome & Time Loop
**Goal:** Integrate the structural connectivity matrix and unroll the system over time using `jax.lax.scan`. (Implemented in `vbjax/ct_mhsa.py`)

- [x] **Implement `network_coupling`**
    - [x] Define the transmission mechanism: $x_{t+1, i} = 
sum_j C_{ij} y_{t, j}$ + External Input.
    - [x] [cite_start]Ensure $C$ is treated as the "transcortical pathway"[cite: 228].
- [x] **Implement `step_fn` for `jax.lax.scan`**
    - [x] Bundle Phase 2 and Coupling into a single functional step: $(State_{t-1}, Input_t) \rightarrow (State_t, Output_t)$.
- [x] **Implement `rollout`**
    - [x] Use `jax.lax.scan` to process a sequence of inputs. (Implemented as `scan_ct_mhsa`).
- [x] **Validation Step 3** (Tested in `vbjax/tests/test_ct_mhsa_rollout.py`)
    - [x] **Test:** Conservation of Energy/Signal. Run for T=100 with identity $C$ and no decay. Check if values explode (NaNs) or vanish.
    - [x] **Test:** Comparison against reference Python loop (for T=5). Ensure `jax.lax.scan` output matches numerically.

---

#### Phase 4: Training & Optimization
**Goal:** Implement the predictive coding loss and stabilize training. (Implemented in `vbjax/train_ct_mhsa.py`)

- [x] **Implement Loss Function (Predictive Coding)**
    - [x] [cite_start]Implement Eq 27: $E_t = \frac{1}{2} || x_{t+1} - y_t ||^2$[cite: 611].
    - [x] *Note:* Since this is a recurrence, we predict the *next* network input or a shifted target sequence.
- [x] **Setup Optimizer (`optax`)**
    - [x] Configure AdamW.
    - [x] **Critical:** Implement a learning rate warmup schedule (crucial for linear attention stability).
    - [x] Gradient clipping (global norm) to handle RNN gradients.
- [x] **Implement Training Step**
    - [x] `jax.value_and_grad` on the rollout function.
    - [x] `optax.apply_updates`.
- [x] **Validation Step 4** (Implemented in `vbjax/train_ct_mhsa.py`)
    - [x] **Test:** Overfitting check. Train on a single batch of random data. Loss should approach zero.
    - [x] **Test:** Stability check. Train for 100 steps. Monitor $M$ statistics (mean/std). If diverging, adjust initialization variance of $W$ parameters.

---

#### Phase 5: "Tiny Shakespeare" Reproduction (Optional / Reach Goal)
**Goal:** replicate the validation task mentioned in the PDF (Fig 4) to prove capability. (Implemented in `examples/shakespeare_ct_mhsa.py`)

- [x] **Data Pipeline**
    - [x] [cite_start]Create a simple char-level tokenizer/loader for Tiny Shakespeare text[cite: 284].
- [x] **Experiment Runner**
    - [x] Run the training loop on the text sequence.
    - [x] Log training loss vs validation loss.
- [x] **Validation Step 5**
    - [x] **Qualitative:** Sample text generation. Does it output coherent character sequences? (Basic generation implemented).
    - [x] **Quantitative:** Compare loss curve to [cite: 240] (Fig 4a). (Loss logging implemented).

---

#### Phase 6: Biological Realism & Observables
**Goal:** Bridge the gap between the current "Map-based" RNN and a biophysical Spatio-Temporal model, enabling "Virtual Brain" simulations.

- [x] **Implement Delayed Coupling (Space-Time)**
    - [x] **Constraint:** Signals require time $\tau_{ij} = D_{ij} / v$ to propagate.
    - [x] **Mechanism:** Maintain a History Buffer (`RingBuffer`) of shape $(L_{max}, N, D_{model})$ within the `lax.scan` loop.
    - [x] **Kernel:** Adapt `vbjax.coupling` to gather $y_{t-\tau_{ij}, j}$ for vector tokens.
- [x] **Structural Enhancements**
    - [x] **Skip Connections:** Add residual pathways (e.g., $x_{t+1} = x_t + \dots$) to improve gradient flow and model stability, mimicking biological preservation of state.
    - [x] **Layer Norm / Normalization:** Essential for deep/recurrent stability in linear transformers.
- [ ] **Task Stimuli (Inputs)**
    - [ ] Define standard injection protocols (Block design, Event-related).
    - [ ] Map scalar stimuli (e.g., visual contrast) to the vector input space $x_t$ (via projection matrix $W_{in}$).
- [ ] **Monitors (Observables)**
    - [ ] **EEG/MEG:** Define Lead Field projection $V(t) = L \cdot y(t)$ (or $L \cdot \text{aggregated activity}$).
    - [ ] **fMRI (BOLD):** Couple the neural output activity (e.g., $L1$ norm of $y_t$) to the Balloon-Windkessel hemodynamic model (existing in `vbjax`).

#### Phase 6.5: Anatomical Functional Mapping
**Goal:** Map abstract model components to specific brain regions using a realistic connectome (weights & lengths).

- [x] **Load Real Connectome:**
    - [x] Import specific Weights ($C$) and Tract Lengths ($L$) matrices (user to provide).
    - [x] Initialize model with delays based on $L$.
- [x] **Functional Region Assignment:**
    - [x] Identify indices for V1 (Visual Input), FEF (Saccade Control), and PFC/Frontal (Task Context).
    - [x] **V1:** Restrict visual feature injection to V1 indices (instead of global broadcast).
    - [x] **FEF:** Restrict saccade motor readout to FEF indices.
    - [x] **PFC:** Inject task context/rule vectors into Frontal regions.

---

#### Phase 7: Active Visual Search Task (The "Gym")
**Goal:** Implement a closed-loop environment where the model actively samples the world to solve a cognitive task.

- [x] **Define Environment (`ActiveVisionEnv`)**
    - [x] **State:** Current Image $I$, Current Eye Position $(x, y)$.
    - [x] **Observation:** Foveated patch $P_t$ extracted from $I$ at $(x, y)$.
    - [x] **Task:** "Feature Search" (e.g., "Find the color with the most objects" or "Count Red items").
    - [x] **Conditioning:** One-hot encoded task vector (e.g., Target Color).
- [x] **Augment Agent Architecture**
    - [x] **Retina Encoder:** A small CNN (e.g., 2-layer Conv) to project $P_t \rightarrow x_{vis} \in \mathbb{R}^{D}$.
    - [x] **Oculomotor Head:** Linear projection $y_t \rightarrow (\Delta x, \Delta y)$ to drive the fovea.
    - [x] **Decision Head:** Linear projection $y_t \rightarrow \text{Class/Count}$ (accumulated or final step).

#### Phase 7.5: Optimization & Anatomical Supervision
**Goal:** Stabilize Active RL using "White Box" supervision of functional regions and reward shaping.

- [x] **Anatomical Supervision:**
    - [x] **PCIP (Priority Map):** Regress `rPCIP` activity to target proximity ($1 - distance$).
    - [x] **PFCDL (Goal Maintenance):** Apply classification loss at *every* time step (not just final) to enforce stable working memory.
- [x] **Reward Shaping:**
    - [x] Implement "Potential-based" reward: $R_{shape} = \text{Dist}_{t-1} - \text{Dist}_t$ (reward approaching the target).
    - [x] Add Entropy regularization to prevent policy collapse. (Partially implemented via aux losses).
- [x] **Hyperparameter Tuning:**
    - [x] Run sweeps for Learning Rate and Auxiliary Weights to maximize "Active" phase stability. (Best: LR=1e-4, Aux=1.0 annealed, Term=5.0, Shape=2.0).

---

#### Phase 8: Pathology Analysis & Mitigation
**Goal:** Resolve the "Immediate Decision / Prior Bias" pathology identified in Phase 7.5 analysis, where the agent commits to a decision at Step 0 (based on priors) and fails to update its internal belief despite active sampling.

- [ ] **Pathology Analysis:**
    - [x] **Diagnosis:** Log analysis shows `stable_step` is consistently 0.0, and confidence remains static (~0.47) throughout the trial regardless of fixations.
    - [ ] **Hypothesis:** Dense classification loss at $t=0$ forces the model to output the "average" label immediately to minimize loss, learning a strong prior instead of evidence accumulation logic.
- [ ] **Mitigation Implementation:**
    - [ ] **Masked Classification Loss:** Apply classification loss *only* at steps $t > T_{warmup}$ (e.g., 5) or only at the final step, to force the model to wait for evidence.
    - [ ] **Evidence Accumulation Reward:** Introduce an intrinsic reward for "Information Gain" (KL Divergence between belief at $t$ and $t-1$) *if* it moves closer to the ground truth.
    - [ ] **Architecture Check:** Verify if the recurrence gating (Delta Rule) is too stiff or initialized poorly, preventing rapid updates.
- [ ] **Validation:**
    - [ ] Re-run `analyze_trials.py`. Success criteria: `stable_step > 0` (decisions change after fixations) and `Accuracy > 60%` on new seeds.

---

#### Phase 9: Neuro-Behavioral Link (Surprise as Signal)
**Goal:** Link the computational "internal state update" to biological "neural activity" observables.

- [ ] **Define Neural Signal Proxy**
    - [ ] [cite_start]Hypothesis: Neural metabolic cost is proportional to the "Bayesian Surprise" or Information Gain[cite: 611].
    - [ ] **Metric:** $NeuralAct_t = || M_t - M_{t-1} ||_F$ (Frobenius norm of the Memory update).
    - [ ] *Interpretation:* High update = high surprise = strong BOLD/EEG response.
- [ ] **Connect to Monitors**
    - [ ] Feed $NeuralAct_t$ into the `vbjax` Balloon-Windkessel model to generate synthetic fMRI.
    - [ ] Feed $NeuralAct_t$ into the Lead Field projection for synthetic ERP/EEG.

#### Phase 9.5: Critical Dynamics & Avalanche Analysis
**Goal:** Investigate if the "neural" surprise signals exhibit signatures of self-organized criticality (SOC), a hallmark of biological brain dynamics.

- [ ] **Avalanche Detection:**
    - [ ] Define an "event" threshold for the surprise signal $S(t)$.
    - [ ] Detect avalanches (contiguous time segments where $S(t) > \text{threshold}$).
- [ ] **Statistical Analysis:**
    - [ ] Compute distributions of Avalanche Size ($S$) and Duration ($D$).
    - [ ] Fit power-laws $P(S) \sim S^{-\alpha}$ and $P(D) \sim D^{-\beta}$.
    - [ ] Check for the scaling relation $\langle S \rangle \sim D^{\gamma}$.

#### Phase 10: Simulation & Empirical Fitting

**Goal:** Validate the model by comparing synthetic behavioral and physiological data against empirical baselines.

- [ ] **Pre-training (Behavioral)**
    - [ ] Train the agent (via RL or Differentiable Attention) to solve the Visual Search task with high accuracy.
    - [ ] **Behavioral Correlates:** Record Reaction Times (number of saccades to solution) and Scanpaths.
- [ ] **Synthetic Data Generation**
    - [ ] Run the pre-trained agent on a test set.
    - [ ] Record:
        1.  **Behavior:** Eye movement trajectories.
        2.  **Physiology:** "Surprise" time-series (simulated BOLD/EEG).
- [ ] **Empirical Fitting (Future)**
    - [ ] Compare synthetic Scanpaths to human eye-tracking datasets.
    - [ ] Correlate synthetic "Surprise" signals with human fMRI/EEG data collected during similar tasks.

---

### Critical Engineering Notes for the Agent
1.  **Einsum Notation:** Always use `einsum` for the linear algebra to keep dimensions $(B, N, H, D)$ clear.
    * *Example Memory Update:* `v_t (B, N, H, D_v), k_t (B, N, H, D_k) -> Outer Product (B, N, H, D_v, D_k)`. (Adhered to).
2.  **JIT Compilation:** Ensure all functions in Phases 2-4 are pure and JIT-compatible. (All core functions are `jax.jit`-able).
3.  **Numerical Stability:** The PDF notes specific forgetting factors $\lambda$. [cite_start]If $\lambda=1$ (no decay), the memory integrates infinitely[cite: 78]. [cite_start]Ensure $\lambda < 1$ (e.g., 0.9 or 0.99) for stability[cite: 587]. (Default `lam=0.9` used).

---

#### Phase 11: Debugging & Micro-Validation (Current Focus)
**Goal:** Diagnose the "Static Prior" pathology using gradient-based probes and micro-simulations to verify signal propagation and learning capability.

- [ ] **Probe 1: One-Step Overfitting (The "Sanity Check")**
    - [ ] **Concept:** Train the model on a *single* batch of fixed images.
    - [ ] **Expectation:** Accuracy must reach 100%. Failure indicates a broken architecture (disconnected graph, dimension bottleneck, or coding error).
- [ ] **Probe 2: Input Sensitivity (Gradient Check)**
    - [ ] **Concept:** Compute $\nabla_{Image} \text{Logits}$.
    - [ ] **Expectation:** Gradients should be non-zero and focused on task-relevant objects. Zero gradients imply the model is ignoring visual input.
- [ ] **Probe 3: Task Context Gradient**
    - [ ] **Concept:** Compute $\nabla_{Task} M_{t=30}$.
    - [ ] **Expectation:** Check if the Task Vector signal vanishes due to $\lambda$ decay over 30 steps.
- [ ] **Probe 4: Visual-Prior Conflict**
    - [ ] **Concept:** Present a "Perfect Stimulus" (e.g., massive target object) that contradicts the prior.
    - [ ] **Expectation:** Logits should shift significantly compared to a blank image.
- [ ] **Probe 5: Surprise Dynamics**
    - [ ] **Concept:** Feed alternating vs. static patches.
    - [ ] **Expectation:** Alternating patches should maintain high "Surprise" (update norm), static patches should decay to 0.
