# Visual Search Model Diagrams

This document illustrates the architecture, task structure, training process, and verification probes for the Whole-Brain Visual Search Agent.

## 1. Whole-Brain Network Architecture

This diagram shows how visual information and task goals are injected into the brain network (Cortico-Thalamic MHSA) and how specific regions drive behavior.

```mermaid
graph TD
    subgraph Inputs
        Retina["Retina (ConvNet)"]
        Pos["Eye Position (x,y)"]
        Task["Task Goal (Color/Shape)"]
    end

    subgraph "Brain Network (38 Regions / Right Hemisphere)"
        direction TB
        
        subgraph "Injection Points"
            rV1["rV1 (Visual Cortex)"]
            rPFC["rPFCDL (Prefrontal)"]
        end
        
        subgraph "Core Dynamics"
            Connectome["Connectome (White Matter)"]
            MHSA["Self-Attention (Micro-Circuit)"]
        end
        
        subgraph "Readout Regions"
            rFEF["rFEF (Frontal Eye Field)"]
            rPCIP["rPCIP (Priority Map)"]
            rPFC_out["rPFCDL (Decision)"]
        end
    end

    subgraph Outputs
        Saccade["Saccade (dx, dy)"]
        Priority["Priority Map"]
        Class["Object Classification"]
        Value["State Value"]
    end

    %% Connections
    Retina -->|Patch Features| rV1
    Pos -->|Pos Embedding| rV1
    Task -->|Task Embedding| rPFC

    rV1 <==>|Coupling| Connectome
    rPFC <==>|Coupling| Connectome
    rFEF <==>|Coupling| Connectome
    rPCIP <==>|Coupling| Connectome
    
    Connectome <==> MHSA

    rFEF --> Saccade
    rPCIP --> Priority
    rPFC_out --> Class
    rPFC_out --> Value

    %% Styling
    style rV1 fill:#ffcccc,stroke:#333
    style rPFC fill:#ccffcc,stroke:#333
    style rFEF fill:#ccccff,stroke:#333
    style rPCIP fill:#ffffcc,stroke:#333
```

## 2. Task Structure (Active Vision Loop)

This diagram illustrates the interaction between the agent and the environment over discrete time steps (fixations).

```mermaid
sequenceDiagram
    participant Env as Environment (Image + Mask)
    participant Agent as Brain Model
    participant Eye as Eye Position

    Note over Eye: Initial Pos (0,0) or Random

    loop For T Steps (e.g. 30)
        Env->>Eye: Extract Visual Patch at (x,y)
        Eye->>Agent: Patch + Current Pos + Task
        
        activate Agent
        Note right of Agent: 1. Retina Processing<br/>2. Connectome Dynamics<br/>3. Head Readout
        Agent->>Agent: Update Internal State (Memory)
        Agent-->>Eye: Saccade Command (dx, dy)
        Agent-->>Env: Class Prediction & Value
        deactivate Agent

        Note over Eye: Update Pos:<br/>(x', y') = (x,y) + (dx,dy) + Noise
        
        Env->>Agent: Reward (Distance/Class)
    end
```

## 3. Training Curriculum

This diagram explains the multi-phase training strategy used to stabilize learning.

```mermaid
flowchart LR
    subgraph "Phase 1: Warm-Up"
        direction TB
        Target["Target Location (Oracle)"]
        FEF_Pre["FEF Policy"]
        Loss_S["Supervised Saccade Loss"]
        
        Target --> Loss_S
        FEF_Pre --> Loss_S
        Loss_S -->|Gradient| FEF_Pre
    end

    subgraph "Phase 2: Active Search (RL + Aux)"
        direction TB
        
        subgraph "RL (PPO)"
            Reward["Reward Signal"]
            Value["Value Head"]
            Policy["Saccade Policy"]
            
            Reward --> Advantage["GAE Calculation"]
            Value --> Advantage
            Advantage --> Loss_Pol["Policy Loss"]
            Advantage --> Loss_Val["Value Loss"]
        end
        
        subgraph "Auxiliary Supervision"
            Oracle["Oracle Saccade"]
            Dist["Distance to Target"]
            
            Oracle --> Loss_AuxS["Aux Saccade Loss"]
            Dist --> Loss_Pri["Priority Map Loss"]
        end
        
        subgraph "Classification"
            Label["Ground Truth"]
            ClassHead["Class Head"]
            Label --> Loss_Cls["Cross Entropy"]
        end
        
        %% Connections
        Loss_Pol & Loss_Val & Loss_AuxS & Loss_Pri & Loss_Cls --> TotalLoss
        TotalLoss -->|Optimizer| ModelParams
    end

    Phase1[Phase 1: FEF Warm-Up] ==> Phase2[Phase 2: Active RL]
    style Phase1 fill:#e6f3ff,stroke:#333
    style Phase2 fill:#fff0e6,stroke:#333
```

## 4. Verification Probes

This diagram details the probes used to verify the internal mechanics of the trained model.

```mermaid
graph TB
    subgraph "Probe A: Belief Update"
        P1_In["Forced Scanpath:\nEmpty -> Object"]
        Model_A["Model"]
        P1_Out["Logit Trace"]
        
        P1_In --> Model_A
        Model_A --> P1_Out
        P1_Out --> Check1{Confidence Spike?}
        Check1 -->|Yes| Pass1[Verify Perception]
    end

    subgraph "Probe B: FEF Vector Field"
        P2_In["Grid of Eye Positions\n(Static Object at Center)"]
        Model_B["Model"]
        P2_Out["Saccade Vectors (dx, dy)"]
        
        P2_In --> Model_B
        Model_B --> P2_Out
        P2_Out --> Check2{Vectors point to Object?}
        Check2 -->|Yes| Pass2[Verify Attention/Motor]
    end
    
    subgraph "Debug Probes"
        D1[Overfitting Check]
        D2[Gradient Sensitivity]
        D3[Signal Trace]
    end

    style Pass1 fill:#ccffcc,stroke:#090
    style Pass2 fill:#ccffcc,stroke:#090
```
