# Sequential Context Encoding

This module implements the **Sequential Context Encoding** step of the CEGE (Conversational Emotion Graph Evolution) model.

## Overview

Following the methodology in the paper, this step processes utterance-level features through a **Bidirectional GRU** to capture sequential context:

$$g_i = \overleftrightarrow{\mathrm{GRU}}_S(g_{i\pm1}, u_i), \quad \text{for } i = 1,2,\ldots,N$$

where:
- $u_i$ is the context-independent utterance representation (from TextCNN, 100-dim)
- $g_i$ is the sequential context-aware utterance representation (output, 100-dim)

## What This Step Does

1. **Loads pre-computed utterance encodings** from the TextCNN encoder (100-dim features)
2. **Groups utterances into conversations** by parsing the conversation structure from raw IEMOCAP data
3. **Processes each conversation** through a Bidirectional GRU to capture temporal dependencies
4. **Outputs context-aware representations** for each utterance that incorporate information from the entire conversation sequence
5. **Preserves conversation metadata** including speaker information and conversation IDs for the next stage

## Key Features

- **Bidirectional processing**: Captures both past and future context for each utterance
- **Conversation-aware**: Processes utterances within their conversational context
- **Speaker tracking**: Maintains speaker information for graph construction
- **Comprehensive logging**: Uses tqdm and detailed logging to track progress

## Usage

```bash
python sequential_context_encoder.py \
    --train_encodings ../../iemocap-encodings/train_encodings.npz \
    --test_encodings ../../iemocap-encodings/test_encodings.npz \
    --train_data ../../iemocap/train.txt \
    --test_data ../../iemocap/test.txt \
    --output_dir ../../iemocap-context-encodings \
    --hidden_dim 100 \
    --num_layers 1 \
    --device cuda
```

### Arguments

**Input paths:**
- `--train_encodings`: Path to train utterance encodings (.npz file from TextCNN)
- `--test_encodings`: Path to test utterance encodings (.npz file from TextCNN)
- `--train_data`: Path to raw train data file (for conversation structure)
- `--test_data`: Path to raw test data file (for conversation structure)

**Output:**
- `--output_dir`: Directory to save context-encoded features (default: `./iemocap-context-encodings`)

**Model parameters:**
- `--hidden_dim`: Hidden dimension for BiGRU (default: 100)
- `--num_layers`: Number of BiGRU layers (default: 1)
- `--dropout`: Dropout rate (default: 0.1, only used if num_layers > 1)
- `--device`: Device to use - `cuda` or `cpu` (default: cuda)

## Output Files

The script generates three files in the output directory:

1. **`train_context_encodings.npz`**: Context-aware features for training set
   - `conv_ids`: Conversation IDs for each utterance
   - `utt_ids`: Utterance IDs
   - `context_features`: Context-aware representations (N × 100)
   - `labels`: Emotion labels
   - `speakers`: Speaker identifiers

2. **`test_context_encodings.npz`**: Context-aware features for test set (same structure)

3. **`sequential_context_encoder.pth`**: Model checkpoint containing:
   - `model_state_dict`: BiGRU weights
   - `hidden_dim`: Hidden dimension
   - `num_layers`: Number of layers
   - `input_dim`: Input feature dimension

## Architecture Details

### SequentialContextEncoder

```
Input: (seq_len, 100) - TextCNN utterance features
  ↓
Bidirectional GRU (100 hidden units, 1 layer)
  ↓
Linear Projection (200 → 100)
  ↓
Output: (seq_len, 100) - Context-aware features
```

### ConversationDataset

- Parses IEMOCAP data format: `utterance_id\tlabel\ttext`
- Extracts conversation IDs from utterance IDs (e.g., `Ses01F_impro01_F000` → `Ses01F_impro01`)
- Groups utterances by conversation, maintaining temporal order
- Links utterances to their pre-computed TextCNN features

## Important Notes

1. **No training at this stage**: The BiGRU is initialized with random weights. It will be trained end-to-end with the full CEGE model in the next stages.

2. **Speaker-agnostic encoding**: As mentioned in the methodology, this encoding is speaker-agnostic. Speaker information is captured later in the graph construction phase.

3. **Conversation boundaries**: The model processes each conversation independently, resetting the hidden state between conversations.

4. **Variable-length sequences**: Each conversation can have a different number of utterances. The model handles this naturally through sequential processing.

## Next Steps

After running this script, the next implementation steps are:

1. **Graph Representation and Temporal Memory**: Build the dynamic temporal graph structure
2. **Temporal Graph Convolution**: Implement the multi-relational graph convolution layers
3. **Full CEGE Training**: Train the complete model end-to-end

## Example Output

```
================================================================================
SEQUENTIAL CONTEXT ENCODING
================================================================================

Configuration:
  Train encodings: ../../iemocap-encodings/train_encodings.npz
  Test encodings: ../../iemocap-encodings/test_encodings.npz
  Train data: ../../iemocap/train.txt
  Test data: ../../iemocap/test.txt
  Output directory: ../../iemocap-context-encodings
  Hidden dim: 100
  Num layers: 1
  Device: cuda
================================================================================

Using device: cuda

================================================================================
LOADING TRAIN DATASET
================================================================================
Loading encodings from ../../iemocap-encodings/train_encodings.npz...
Loaded 5747 utterances with 100-dim features
Parsing conversation structure from ../../iemocap/train.txt...
Found 120 conversations
After filtering: 120 conversations with features

...

Encoded 5747 utterances
Context features shape: (5747, 100)

================================================================================
SEQUENTIAL CONTEXT ENCODING COMPLETE!
================================================================================
```
