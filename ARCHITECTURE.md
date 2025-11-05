# CEGE Architecture Flow

## Complete Pipeline

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          CEGE: COMPLETE ARCHITECTURE                        │
└─────────────────────────────────────────────────────────────────────────────┘

INPUT: TextCNN Utterance Encodings (100-dim)
   │
   │   u_1, u_2, ..., u_N  ∈ R^100
   │
   ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│  STEP 1: Sequential Context Encoding (DialogueGCN-aligned)                  │
│  ──────────────────────────────────────────────────────────────────         │
│  ┌──────────────────────────────────────────────────────────┐               │
│  │  2-Layer Bidirectional GRU                                │               │
│  │  - Input: 100-dim                                         │               │
│  │  - Hidden: 100-dim per direction                          │               │
│  │  - Output: 200-dim (bidirectional)                        │               │
│  │                                                            │               │
│  │  Forward:  u_1 → u_2 → ... → u_N                         │               │
│  │  Backward: u_N → ... → u_2 → u_1                         │               │
│  │                                                            │               │
│  │  MatchingAttention (general2)                             │               │
│  │  - Attention over entire conversation                     │               │
│  │  - Weighted context aggregation                           │               │
│  └──────────────────────────────────────────────────────────┘               │
│                                                                               │
│  Output: g_i ∈ R^200 (context-aware embeddings)                             │
└──────────────────────────────────────────────────────────────────────────────┘
   │
   │   g_1, g_2, ..., g_N  ∈ R^200
   │
   ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│  STEP 2: State Tracking & Temporal Memory                                   │
│  ──────────────────────────────────────────────────────────────────         │
│  For each utterance i at time t:                                            │
│                                                                               │
│  ┌──────────────────────────┐  ┌──────────────────────────┐                │
│  │ Speaker State Tracker    │  │ Conversation State       │                │
│  │ ──────────────────────   │  │ ──────────────────────   │                │
│  │ Q_s^t = GRU(Q_s^{t-1},  │  │ C^t = GRU(C^{t-1}, g_i) │                │
│  │              g_i)         │  │                          │                │
│  │                           │  │                          │                │
│  │ - Separate state per      │  │ - Global conversation    │                │
│  │   speaker (F, M)          │  │   context                │                │
│  │ - 150-dim state           │  │ - 150-dim state          │                │
│  └──────────────────────────┘  └──────────────────────────┘                │
│                 │                           │                                 │
│                 └─────────┬─────────────────┘                                │
│                           ▼                                                  │
│  ┌────────────────────────────────────────────────────────┐                │
│  │ Temporal Memory Module (LSTM)                          │                │
│  │ ─────────────────────────────────────────────────────  │                │
│  │ Input: [g_i | Q_s^t | C^t]  (500-dim)                 │                │
│  │                                                         │                │
│  │ m_i^t, cell_i^t = LSTM(m_i^{t-1}, cell_i^{t-1},       │                │
│  │                         [g_i | Q_s^t | C^t])           │                │
│  │                                                         │                │
│  │ Output: m_i ∈ R^200 (temporal memory state)            │                │
│  └────────────────────────────────────────────────────────┘                │
└──────────────────────────────────────────────────────────────────────────────┘
   │
   │   For each utterance: (g_i, m_i)
   │
   ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│  STEP 3: Dynamic Graph Construction                                         │
│  ──────────────────────────────────────────────────────────────────         │
│  ┌────────────────────────────────────────────────────────┐                │
│  │ Edge Weight Prediction Network                          │                │
│  │ ─────────────────────────────────────────────────────  │                │
│  │ For each pair (i, j):                                   │                │
│  │                                                          │                │
│  │ Input features:                                          │                │
│  │   - g_i, g_j (node embeddings)                          │                │
│  │   - Same speaker? (binary)                              │                │
│  │   - Temporal distance: |t_i - t_j|                      │                │
│  │   - Temporal decay: exp(-λ * |t_i - t_j|)              │                │
│  │                                                          │                │
│  │ alpha_ij = σ(MLP([g_i | g_j | speaker | temporal]))    │                │
│  │                                                          │                │
│  │ Edge Pruning: Keep if alpha_ij >= tau_remove (0.1)      │                │
│  │ Edge Creation: Add if alpha_ij > tau_create (0.7)       │                │
│  └────────────────────────────────────────────────────────┘                │
│                                                                               │
│  ┌────────────────────────────────────────────────────────┐                │
│  │ Relation Type Assignment                                │                │
│  │ ─────────────────────────────────────────────────────  │                │
│  │ rel_type = speaker_i * 2 * n_speakers +                 │                │
│  │            speaker_j * 2 +                               │                │
│  │            temporal_direction                            │                │
│  │                                                          │                │
│  │ Example (2 speakers):                                    │                │
│  │   F → F (past):  0    F → F (future): 1                │                │
│  │   F → M (past):  2    F → M (future): 3                │                │
│  │   M → F (past):  4    M → F (future): 5                │                │
│  │   M → M (past):  6    M → M (future): 7                │                │
│  └────────────────────────────────────────────────────────┘                │
│                                                                               │
│  Output: G = (V, E, R, W)                                                   │
│          - V: nodes (utterances)                                             │
│          - E: edge_index (2, num_edges)                                      │
│          - R: edge_type (num_edges,)                                         │
│          - W: edge_weights (num_edges,)                                      │
└──────────────────────────────────────────────────────────────────────────────┘
   │
   │   Graph with dynamic edges
   │
   ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│  STEP 4: Temporal Graph Convolution (2 layers)                              │
│  ──────────────────────────────────────────────────────────────────         │
│  ┌────────────────────────────────────────────────────────┐                │
│  │ Layer 1: Relation-specific GCN (RGCN)                  │                │
│  │ ─────────────────────────────────────────────────────  │                │
│  │ Input: [g_i | m_i] ∈ R^400 (concatenated)              │                │
│  │                                                          │                │
│  │ h_i^(1) = ReLU( Σ_r Σ_{j∈N_i^r}                        │                │
│  │                 α_ij/c_ir * W_r [g_j | m_j] +          │                │
│  │                 W_0 [g_i | m_i] )                       │                │
│  │                                                          │                │
│  │ - Relation-specific transformations W_r                 │                │
│  │ - Edge weight normalization α_ij                        │                │
│  │ - Self-connection W_0                                    │                │
│  │                                                          │                │
│  │ Output: h^(1) ∈ R^200                                   │                │
│  └────────────────────────────────────────────────────────┘                │
│                           │                                                  │
│                           ▼                                                  │
│  ┌────────────────────────────────────────────────────────┐                │
│  │ Layer 2: Standard Graph Convolution                     │                │
│  │ ─────────────────────────────────────────────────────  │                │
│  │ h_i^(2) = ReLU( Σ_j W h_j^(1) + W_0 h_i^(1) )          │                │
│  │                                                          │                │
│  │ Output: h^(2) ∈ R^200                                   │                │
│  └────────────────────────────────────────────────────────┘                │
└──────────────────────────────────────────────────────────────────────────────┘
   │
   │   h_i ∈ R^200
   │
   ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│  STEP 5: Temporal Attention over Memory                                     │
│  ──────────────────────────────────────────────────────────────────         │
│  For each utterance i at time t:                                            │
│                                                                               │
│  ┌────────────────────────────────────────────────────────┐                │
│  │ Attention Mechanism                                      │                │
│  │ ─────────────────────────────────────────────────────  │                │
│  │ scores_ik = (h_i^(2))^T W_β m_k  for k < i             │                │
│  │                                                          │                │
│  │ Temporal Decay:                                          │                │
│  │   scores_ik *= exp(-λ_decay * (t - k))                 │                │
│  │                                                          │                │
│  │ Attention Weights:                                       │                │
│  │   β_ik = softmax(scores_ik)                             │                │
│  │                                                          │                │
│  │ Context Vector:                                          │                │
│  │   context_i = Σ_k β_ik * m_k                           │                │
│  │                                                          │                │
│  │ - Recent memories weighted more                          │                │
│  │ - Focuses on relevant history                            │                │
│  └────────────────────────────────────────────────────────┘                │
│                                                                               │
│  Output: context_i ∈ R^200                                                  │
└──────────────────────────────────────────────────────────────────────────────┘
   │
   │   [h_i | context_i] ∈ R^400
   │
   ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│  STEP 6: Classification                                                      │
│  ──────────────────────────────────────────────────────────────────         │
│  ┌────────────────────────────────────────────────────────┐                │
│  │ Multi-Layer Perceptron                                  │                │
│  │ ─────────────────────────────────────────────────────  │                │
│  │ Input: [h_i | context_i] ∈ R^400                        │                │
│  │                                                          │                │
│  │ hidden = ReLU(Linear_400→100([h_i | context_i]))        │                │
│  │ hidden = Dropout(hidden)                                 │                │
│  │ logits = Linear_100→6(hidden)                           │                │
│  │                                                          │                │
│  │ Output: log_prob = LogSoftmax(logits)                   │                │
│  └────────────────────────────────────────────────────────┘                │
│                                                                               │
│  Output: Emotion predictions for 6 classes                                  │
│          [happy, sad, neutral, angry, excited, frustrated]                  │
└──────────────────────────────────────────────────────────────────────────────┘
   │
   │   log P(y_i | conversation context)
   │
   ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│  TRAINING LOSS                                                               │
│  ──────────────────────────────────────────────────────────────────         │
│  L = L_CE + λ_reg * ||θ||_2^2 + λ_temp * L_temporal                        │
│                                                                               │
│  - L_CE: Cross-entropy loss for emotion classification                      │
│  - L_reg: L2 regularization (weight decay)                                  │
│  - L_temporal: Temporal consistency (smooth memory evolution)               │
└──────────────────────────────────────────────────────────────────────────────┘

```

## Key Innovations

1. **DialogueGCN-Aligned Sequential Encoding**
   - 2-layer BiGRU (not 1)
   - MatchingAttention
   - 200-dim output (not projected to 100)
   - Supervised training

2. **Explicit Temporal Memory**
   - LSTM-based memory for each node
   - Integrates speaker states and conversation state
   - Maintains history across utterances

3. **Dynamic Graph Evolution**
   - Learned edge weights (not hand-crafted)
   - Dynamic edge pruning/addition
   - Relation types capture speaker interactions

4. **Temporal Attention with Decay**
   - Focuses on relevant history
   - Recent memories weighted more
   - Smooth integration of context

5. **Multi-level State Tracking**
   - Speaker states (Q_s)
   - Conversation state (C)
   - Node memory (m_i)
   - All updated dynamically

## Dimension Flow

```
100 → 200 → 400 → 200 → 400 → 100 → 6
 ↑     ↑     ↑     ↑     ↑     ↑    ↑
Text  BiG   [g|m] GCN  [h|c] MLP  Classes
CNN   RU          2-layer     
```

Total Parameters: **4,103,814**
