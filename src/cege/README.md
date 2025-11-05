# CEGE: Conversational Emotion Graph Evolution

Complete implementation of the CEGE model for Emotion Recognition in Conversation (ERC) on the IEMOCAP dataset.

## Architecture Overview

CEGE implements the full methodology with the following components:

### 1. Sequential Context Encoder (Matching DialogueGCN)
- **2-layer Bidirectional GRU** (instead of 1 layer)
- **MatchingAttention mechanism** from DialogueGCN
- **200-dimensional output** (2 × 100) for bidirectional context
- **Supervised training** with emotion labels (not just inference)

### 2. Temporal Memory Module
- **LSTM-based memory** for each node: `m_i^t = LSTM(m_i^{t-1}, g_i, context^t)`
- Integrates:
  - Speaker state `Q_s^t`
  - Global conversation state `C^t`
  - Context-aware embeddings `g_i`

### 3. Dynamic Graph Construction
- **Graph representation**: `G = (V, E, R, W)`
  - Nodes: utterances
  - Edges: directed with relation types
  - Weights: `alpha_ij ∈ [0,1]` based on features and temporal distance
- **Edge pruning**: removes edges with weight < `tau_remove`
- **Edge addition**: creates edges with weight > `tau_create`
- **Relation types**: based on speaker interactions and temporal order

### 4. Temporal Graph Convolution
- **Layer 1**: Relation-specific GCN with edge weights
  - Uses RGCNConv with relation types
  - Concatenates `[g_i | m_i]` for node features
- **Layer 2**: Standard GraphConv for aggregation

### 5. Temporal Attention
- **Attention over historical memory** states
- **Temporal decay**: `exp(-lambda_decay * (t-k))`
- Focuses on relevant conversation history

### 6. Classification
- **MatchingAttention** on final features
- **Multi-layer perceptron** for emotion prediction
- **Log-softmax** output for 6 emotion classes

## Key Differences from Previous Implementation

| Component | Old Implementation | New Implementation (DialogueGCN-aligned) |
|-----------|-------------------|------------------------------------------|
| **GRU Layers** | 1 layer | **2 layers** |
| **Attention** | None | **MatchingAttention** |
| **Output Dim** | 100 (projected) | **200 (bidirectional)** |
| **Training** | None (inference only) | **Supervised with emotion labels** |
| **Loss** | Reconstruction MSE | **Cross-entropy + regularization** |
| **Integration** | Separate preprocessing | **End-to-end with graph network** |

## Installation

```bash
pip install -r requirements.txt
```

**Note**: You may need to install PyTorch Geometric separately:
```bash
pip install torch-geometric torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-{TORCH_VERSION}+{CUDA}.html
```

## Directory Structure

```
src/cege/
├── model.py                 # Complete CEGE model
├── dataloader.py           # IEMOCAP data loader
├── train_IEMOCAP.py        # Training script
├── requirements.txt        # Dependencies
└── README.md              # This file
```

## Training

### Basic Training

```bash
python train_IEMOCAP.py \
    --train-encodings ../../iemocap-encodings/train_encodings.npz \
    --test-encodings ../../iemocap-encodings/test_encodings.npz \
    --train-data ../../iemocap/train.txt \
    --test-data ../../iemocap/test.txt
```

### Custom Hyperparameters

```bash
python train_IEMOCAP.py \
    --train-encodings ../../iemocap-encodings/train_encodings.npz \
    --test-encodings ../../iemocap-encodings/test_encodings.npz \
    --train-data ../../iemocap/train.txt \
    --test-data ../../iemocap/test.txt \
    --batch-size 8 \
    --epochs 60 \
    --lr 0.0001 \
    --dropout 0.5 \
    --D-e 100 \
    --memory-dim 200 \
    --gcn-hidden-dim 200 \
    --tau-remove 0.1 \
    --tau-create 0.7 \
    --lambda-decay 0.1
```

### All Available Arguments

**Data Paths:**
- `--train-encodings`: Path to training TextCNN encodings (.npz)
- `--test-encodings`: Path to test TextCNN encodings (.npz)
- `--train-data`: Path to training IEMOCAP data (.txt)
- `--test-data`: Path to test IEMOCAP data (.txt)

**Model Architecture:**
- `--D-m`: Input feature dimension (default: 100)
- `--D-e`: Sequential encoder hidden dim (default: 100)
- `--D-h`: Classification hidden dim (default: 100)
- `--speaker-state-dim`: Speaker state dimension (default: 150)
- `--conv-state-dim`: Conversation state dimension (default: 150)
- `--memory-dim`: Temporal memory dimension (default: 200)
- `--gcn-hidden-dim`: GCN hidden dimension (default: 200)
- `--n-speakers`: Number of speakers (default: 2)
- `--n-classes`: Number of emotion classes (default: 6)
- `--n-relations`: Number of relation types (default: 8)
- `--dropout`: Dropout rate (default: 0.5)

**Graph Dynamics:**
- `--tau-remove`: Edge removal threshold (default: 0.1)
- `--tau-create`: Edge creation threshold (default: 0.7)
- `--lambda-decay`: Temporal decay parameter (default: 0.1)

**Training:**
- `--batch-size`: Batch size (default: 8, smaller for memory)
- `--epochs`: Number of epochs (default: 60)
- `--lr`: Learning rate (default: 0.0001)
- `--lambda-reg`: L2 regularization weight (default: 0.00001)
- `--lambda-temp`: Temporal consistency weight (default: 0.01)
- `--valid-split`: Validation split fraction (default: 0.1)

**Other:**
- `--seed`: Random seed (default: 42)
- `--no-cuda`: Disable CUDA
- `--save-dir`: Directory to save models (default: ./checkpoints)
- `--log-interval`: Logging interval (default: 1)

## Model Components

### 1. SequentialContextEncoder
```python
# 2-layer BiGRU with MatchingAttention
self.gru = nn.GRU(input_size=100, hidden_size=100, num_layers=2, bidirectional=True)
self.matchatt = MatchingAttention(200, 200, att_type='general2')
```

### 2. TemporalMemoryModule
```python
# LSTM for temporal memory
self.lstm = nn.LSTMCell(input_dim=450, hidden_dim=200)
# Input: [g_i (200) + Q_s (150) + C (150)]
```

### 3. DynamicGraphBuilder
```python
# Edge weight prediction
self.edge_weight_net = nn.Sequential(...)  # Neural network
# Pruning: keep edges with weight >= tau_remove
# Creation: add edges with weight > tau_create
```

### 4. TemporalGCNLayer
```python
# Relation-specific GCN
self.rgcn = RGCNConv(input_dim=400, hidden_dim=200, n_relations=8)
# Input: [g_i | m_i] concatenated
```

### 5. TemporalAttention
```python
# Attention over memory with decay
attention = softmax((h_i)^T W_beta M) * exp(-lambda * distance)
```

## Loss Function

The total loss combines three components:

```
L = L_CE + lambda_reg * ||theta||_2^2 + lambda_temp * temporal_consistency
```

- **L_CE**: Cross-entropy loss for emotion classification
- **L2 regularization**: Prevents overfitting
- **Temporal consistency**: Encourages smooth memory evolution

## Output

During training, you'll see:

```
================================================================================
Epoch 1/60
--------------------------------------------------------------------------------

Training: 100%|██████████| 108/108 [02:15<00:00, loss=1.7823, ce=1.7654]
Evaluating: 100%|██████████| 12/12 [00:08<00:00]

Results:
  Train - Loss: 1.7823, Acc: 35.42%, F1: 33.18%
  Valid - Loss: 1.6234, Acc: 38.91%, F1: 36.54%
  Test  - Loss: 1.5987, Acc: 40.23%, F1: 38.76%
  Best Valid F1: 36.54%, Best Test F1: 38.76%
  Time: 143.23s
```

## Evaluation

After training, the best model is saved to `checkpoints/best_model.pt` and evaluated:

```
================================================================================
Final Test Performance
================================================================================

Best Model Performance:
  Test Accuracy: 52.34%
  Test F1: 50.12%

Classification Report:
              precision    recall  f1-score   support
      happy      0.5234    0.4891    0.5056       124
        sad      0.5891    0.6123    0.6005       203
    neutral      0.4234    0.4567    0.4394       367
      angry      0.6123    0.5789    0.5951       98
    excited      0.5456    0.5234    0.5343       86
 frustrated      0.4891    0.5012    0.4950       122
```

## Memory Requirements

- **GPU**: Recommended 8+ GB VRAM (use smaller batch size if needed)
- **CPU**: Possible but slower (use `--no-cuda`)
- **Batch size**: Start with 8, reduce to 4 or 2 if OOM errors occur

## Next Steps

1. **Hyperparameter tuning**: Experiment with different values
2. **Ablation studies**: Test impact of each component
3. **Visualization**: Plot graph evolution over conversations
4. **Analysis**: Examine attention weights and edge dynamics

## Citation

If you use this implementation, please cite the CEGE methodology paper.

## License

Research use only.
