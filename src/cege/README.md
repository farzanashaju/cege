# CEGE

```bash
# with temporal features
python train.py \
  --base-model GRU \
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
  --base-model GRU \
  --graph-model \
  --nodal-attention \
  --class-weight \
  --dropout 0.4 \
  --lr 0.0003 \
  --batch-size 32 \
  --epochs 60 \
  --l2 0.0 \
  --no-cuda


# sample a random test conversation, using CUDA if available
python3 sample.py --model bestmodel.pth

# or sample a specific conversation index from train split and force CPU
python3 sample.py --model bestmodel.pth --split train --idx 5 --no-cuda

python3 sample.py --model bestmodel.pth --track-idx 10 --outdir sample_plots --show

```
