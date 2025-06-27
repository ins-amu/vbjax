# **JAX-based DeltaNet Model for Pathfinder Task**

## **1\. Project Overview**

This project provides a JAX-based implementation of a recurrent neural network inspired by the DeltaNet architecture. The model is specifically designed to solve the Long Range Arena (LRA) Pathfinder task. This task, which requires determining if two points on a 64x64 grid are connected by a continuous path, serves as an ideal testbed. It directly challenges a model's ability to capture long-range spatial dependencies and route information selectively, capabilities that are central to the implemented architecture.  
The model's design is guided by a theoretical framework that views cognition not as a single, passive feed-forward pass, but as an active, iterative process of belief updating. This perspective is more aligned with biological cognition. The core of the model is a recurrent update rule that mathematically embodies this process of Bayesian belief propagation through cycles of prediction and error correction.

## **2\. Theoretical Foundations**

The model synthesizes several key concepts from computational neuroscience and machine learning to create a unified system.

* **Active Inference & Belief Updating**: Unlike standard Vision Transformers which process an entire image in one go, this model processes information sequentially, token-by-token. Each new token (an image patch) is used to update an internal belief state. This iterative refinement is computationally analogous to how an organism might scan a scene, gathering evidence over time. This approach has implications for biological plausibility and offers potential efficiency gains in complex tasks where only a subset of information is relevant at any given moment.  
* **The DeltaNet Update Rule**: The heart of the recurrence is the update to a state matrix S at each step t. This matrix represents the accumulated knowledge or beliefs of the model. The update rule has two equivalent mathematical forms:  
  1. Multiplicative Form: S\_t \= S\_{t-1} @ (I \- B \* k\_t @ k\_t.T) \+ v\_t @ k\_t.T  
     This form highlights how the prior belief S\_{t-1} is first adjusted by "explaining away" the predictable parts of the input, and then new information is added.  
  2. Prediction Error Form: S\_t \= S\_{t-1} \+ (v\_t \- B \* S\_{t-1} @ k\_t) @ k\_t.T  
     This is the form directly implemented in the code as it makes the learning signal explicit. It elegantly decomposes the update into intuitive computational steps:  
     * v\_pred \= B \* S\_{t-1} @ k\_t: The model uses its prior belief (S\_{t-1}) to predict the current value vector (v\_t) based on the current context or key vector (k\_t).  
     * error \= v\_t \- v\_pred: The model calculates the prediction error—the "surprise"—which quantifies how much the actual input deviates from the prediction. This error signal is the primary driver of learning and belief change.  
     * The outer product of the error and the k\_t vector creates the final update matrix dS, ensuring that the change to the belief state S is precisely targeted based on the context that produced the error.  
     * B is a learnable, per-head scalar parameter that modulates the strength of the prediction, allowing each head to learn how confidently to trust its own prior beliefs.  
* **Criticality & Precision Amplification**: The framework hypothesizes that effective cognitive systems operate near a critical state, where information can propagate without dying out or exploding. This is characterized by "avalanches" of neural activity that follow scale-free power laws. To model this, the system incorporates a precision amplification mechanism tied to prediction error:  
  * The magnitude of the prediction error ||error|| is calculated. A large error signifies a highly surprising event.  
  * This magnitude is used to compute a temperature T \= exp(-alpha \* ||error||). A larger error leads to a smaller temperature.  
  * This temperature dynamically sharpens the softmax output for the efferent (outgoing) signal: softmax((S\_t @ q\_t) / T).  
  * The consequence is that more "surprising" events (large errors) trigger higher-precision (lower temperature) outputs. This amplifies the signal, ensuring that significant, belief-violating information is propagated forcefully through the network, leading to rapid adaptation.  
* **Normalization**: Input tokens are z-scored before being projected into Q, K, and V vectors. This serves as a simple homeostatic mechanism to stabilize the network's internal dynamics. By ensuring each token has a mean of zero and unit variance, it prevents the recurrent updates from causing the signals to either explode or vanish over long sequences, keeping the network in a responsive and trainable regime.

## **3\. Model Architecture & Implementation**

### **3.1 Data Pipeline**

* **Input**:  
  * Images (ims.npy): A NumPy array of shape (N, 64, 64).  
  * Labels (labels.npy): A NumPy array of binary labels (N,).  
* **Preprocessing**:  
  * Each 64x64 image is divided into a sequence of non-overlapping 4x4 patches. This tokenization transforms the spatial problem into a sequential one.  
  * Each patch is flattened into a 16-dimensional vector (token), resulting in a sequence length of (64/4)^2 \= 256 tokens.  
  * To retain crucial spatial information lost during flattening, two types of embeddings are added to each token: positional embeddings for the patch's original (x, y) grid location and sequential embeddings for the token's index in the 1D sequence.

### **3.2 Core Logic (delta\_net\_layer\_forward)**

The model consists of several stacked DeltaNet layers. Within each layer, a jax.lax.scan function iterates through the sequence, which is the functional equivalent of a for loop in JAX and the engine for the temporal recurrence. For each token:

1. The input token is **z-scored** for normalization.  
2. The normalized token is projected into per-head Q, K, and V vectors.  
3. Q and K vectors are passed through a sigmoid nonlinearity to constrain their values.  
4. The per-head state matrix S is updated using the computationally explicit prediction error formulation of the DeltaNet rule.  
5. A precision-amplified softmax is applied to generate an efferent signal, with the temperature determined by the prediction error.  
6. The signals from all heads are concatenated, projected back to the embedding dimension, and added to the original input via a **skip connection**. This is critical for training deep networks as it provides a direct path for gradients to flow, mitigating the vanishing gradient problem.  
7. A final layer normalization is applied to stabilize the output of the entire layer before it is passed to the next.

### **3.3 JAX Implementation Details**

* **Framework**: The model is built entirely in JAX, using optax for the Adam optimizer and its rich ecosystem of optimizers and schedulers.  
* **Dataclasses**:  
  * TrainingConfig: Stores all global hyperparameters like learning rate, batch size, and model dimensions (embed\_dim, num\_heads, num\_layers).  
  * ModelParams & DeltaLayerParams: Flax-style dataclasses hold all trainable parameters. This is essential as it organizes the parameters into a nested structure that JAX recognizes as a pytree, allowing functions like grad and jit to work seamlessly on the entire model. The trainable B parameter, for instance, lives inside DeltaLayerParams with a shape of (num\_heads,).  
* **Vectorization**: vmap is used extensively to parallelize computations. It allows the same function to be efficiently run over the batch dimension and across the different attention heads within each layer, leading to clean and highly performant code.  
* **Compilation**: jit is used on the train\_step function. This compiles the entire device-side computation graph for the loss calculation, backpropagation, and optimizer update, resulting in significant speedups on accelerators like GPUs.

### **3.4 Code Structure**

* main.py: A self-contained script with all the code.  
  * **Dataclasses**: ModelParams, DeltaLayerParams, TrainingConfig.  
  * init\_model(): Initializes all model parameters with random values according to the specified configuration.  
  * delta\_net\_layer\_forward(): Contains the core recurrent logic for a single DeltaNet layer.  
  * forward\_pass(): Defines the full pass through all layers, from input embedding to final logit.  
  * loss\_fn(): Calculates the sigmoid binary cross-entropy loss from the logits and labels.  
  * train\_step(): A JIT-compiled function that executes one full training step.  
  * train(): The main training loop that handles data loading, batching, and orchestrates the training process.

## **4\. Next Steps & Future Directions**

* **Analysis**: Implement logging within the train\_step to record the average magnitude of the prediction error ||error|| for each batch. This data can be used to test the criticality hypothesis by plotting the distribution of error magnitudes over time and checking for power-law characteristics (e.g., via a log-log plot).  
* **Optimization**: Experiment with more advanced optax learning rate schedules, such as warmup\_cosine\_decay\_schedule, which can lead to more stable training and better final performance.  
* **Architecture**: Explore alternative architectural choices. For example, instead of averaging all final token outputs for the readout, one could use only the output of the very last token in the sequence. Different nonlinearities for the Q and K vectors (e.g., tanh) could also be investigated.