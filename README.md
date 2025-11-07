# CEGE: Conversational Emotion Graph Evolution

Emotion Recognition in Conversation (ERC) is key to empathetic dialogue systems. Existing models use static graphs or sequences, missing the evolving nature of emotions. **CEGE** introduces a **dynamic temporal graph neural network** that updates utterance representations and influence edges as conversations progress.



## Features

* Dynamic graph evolution with temporal memory.
* CNN utterance encoding + GRU sequential context.
* Attention-based emotion classification.
* Baselines: DialogueRNN, DialogueGCN.

## Instructions To Run

```bash
# with temporal features
python train.py \
  --base-model LSTM \
  --graph-model \
  --temporal-model \
  --nodal-attention \
  --class-weight \
  --dropout 0.4 \
  --lr 0.0003 \
  --batch-size 32 \
  --epochs 60 \
  --l2 0.0 \
  --lambda-temp 0.1 \
  --no-cuda

# without temporal features
python train.py \
  --base-model LSTM \
  --graph-model \
  --nodal-attention \
  --class-weight \
  --dropout 0.4 \
  --lr 0.0003 \
  --batch-size 32 \
  --epochs 60 \
  --l2 0.0 \
  --no-cuda


# sample a random test conversation
python3 sample.py --model bestmodel.pth --no-cuda

# or sample a specific conversation index from train split
python3 sample.py --model bestmodel.pth --split train --idx 5

# sample a specific utterance
python3 sample.py --model bestmodel.pth --track-idx 10
```

## Contributors

Bhavya V · Farzana S · Rudra Choudhary 

📧 {bhavyaram.v, farzana.s, rudra.choudhary}@research.iiit.ac.in



