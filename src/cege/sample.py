import argparse
import random
import torch
import numpy as np
from dataloader import IEMOCAPDataset
from model import DialogueGCNTemporalModel

# plotting libs
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.decomposition import PCA
    HAS_PLOTTING = True
except Exception:
    HAS_PLOTTING = False

# label names consistent with train.py
EMOTION_LABELS = ['Neutral', 'Happy', 'Sad', 'Angry', 'Excited', 'Frustrated']


def load_checkpoint(path, device):
    try:
        ckpt = torch.load(path, map_location=device)
    except Exception as e:
        print('Initial torch.load failed with:', e)
        # Try to reload allowing full pickle semantics (weights_only=False).
        # This may be required for older checkpoints that contain non-weight objects
        # (only do this for trusted checkpoints).
        try:
            # If the newer torch provides add_safe_globals, allow numpy scalar class
            if hasattr(torch.serialization, 'add_safe_globals'):
                try:
                    with torch.serialization.add_safe_globals([np.core.multiarray.scalar]):
                        ckpt = torch.load(path, map_location=device, weights_only=False)
                except Exception:
                    # Some numpy builds may not expose multiarray.scalar; fall back
                    ckpt = torch.load(path, map_location=device, weights_only=False)
            else:
                ckpt = torch.load(path, map_location=device, weights_only=False)
            print('Loaded checkpoint with weights_only=False.')
        except Exception as e2:
            print('Retry with weights_only=False failed with:', e2)
            raise

    # try common checkpoint formats
    if isinstance(ckpt, dict):
        # common keys: 'state_dict', 'model_state', 'model', 'model_state_dict'
        for key in ('state_dict', 'model_state', 'model', 'model_state_dict'):
            if key in ckpt:
                return ckpt[key]
        # otherwise assume ckpt itself is state_dict-like
        return ckpt
    else:
        # assume it's a state_dict
        return ckpt


def build_model(device, base_model='LSTM'):
    # hyperparameters used in training (from train.py)
    D_m = 100
    D_g = 150
    D_p = 150
    D_e = 100
    D_h = 100
    D_a = 100
    graph_h = 100

    model = DialogueGCNTemporalModel(base_model,
                                     D_m, D_g, D_p, D_e, D_h, D_a, graph_h,
                                     n_speakers=2,
                                     max_seq_len=110,
                                     window_past=10,
                                     window_future=10,
                                     n_classes=len(EMOTION_LABELS),
                                     listener_state=False,
                                     context_attention='general',
                                     dropout=0.5,
                                     nodal_attention=True,
                                     no_cuda=(device.type != 'cuda'))

    model.to(device)
    model.eval()
    return model


def collate_single_sample(dataset, idx):
    sample = dataset[idx]
    batch = dataset.collate_fn([sample])
    return batch


def safe_tensor_to(device, t):
    if isinstance(t, torch.Tensor):
        return t.to(device)
    return t


def plot_conversation(vid, sentences, true_labels, preds, mem_arr, outdir, show, log_prob=None, qmask=None):
    """Create and save comprehensive visualization plots."""
    if not HAS_PLOTTING:
        print('matplotlib/seaborn not available. Skipping plots. Install with: pip install matplotlib seaborn')
        return

    import os
    os.makedirs(outdir, exist_ok=True)

    seq_len = len(true_labels)
    x = np.arange(seq_len)

    # Set style for prettier plots
    sns.set_style("whitegrid")
    plt.rcParams['font.size'] = 10
    plt.rcParams['axes.labelsize'] = 11
    plt.rcParams['axes.titlesize'] = 12
    plt.rcParams['legend.fontsize'] = 9

    # label indices
    true_idx = np.array(true_labels)
    pred_idx = np.array(preds[:seq_len])
    
    # Color palette for emotions
    emotion_colors = {
        0: '#95a5a6',  # Neutral - gray
        1: '#f39c12',  # Happy - orange
        2: '#3498db',  # Sad - blue
        3: '#e74c3c',  # Angry - red
        4: '#9b59b6',  # Excited - purple
        5: '#e67e22',  # Frustrated - dark orange
    }

    # 1) Enhanced label timeline with color coding and error highlighting
    fig, ax = plt.subplots(figsize=(14, 4))
    
    # Plot background for correct/incorrect predictions
    for i in range(seq_len):
        if true_idx[i] == pred_idx[i]:
            ax.axvspan(i-0.4, i+0.4, alpha=0.1, color='green')
        else:
            ax.axvspan(i-0.4, i+0.4, alpha=0.1, color='red')
    
    # Plot lines with markers
    ax.plot(x, true_idx, marker='o', markersize=8, label='Ground Truth', 
            linestyle='-', linewidth=2, color='#2c3e50', alpha=0.8)
    ax.plot(x, pred_idx, marker='x', markersize=10, label='Prediction', 
            linestyle='--', linewidth=2, color='#e74c3c', alpha=0.8)
    
    ax.set_yticks(range(len(EMOTION_LABELS)))
    ax.set_yticklabels(EMOTION_LABELS)
    ax.set_xlabel('Utterance Index', fontweight='bold')
    ax.set_ylabel('Emotion', fontweight='bold')
    ax.set_title('Emotion Recognition Timeline\n(Green=Correct, Red=Incorrect)', 
                 fontweight='bold', pad=15)
    ax.legend(loc='upper right', framealpha=0.9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fname1 = os.path.join(outdir, f'{vid}_labels.png')
    plt.savefig(fname1, dpi=150, bbox_inches='tight')
    if show:
        plt.show()
    plt.close()

    # 2) Confusion scatter plot
    fig, ax = plt.subplots(figsize=(8, 8))
    for i in range(seq_len):
        color = emotion_colors.get(true_idx[i], '#95a5a6')
        marker = 'o' if true_idx[i] == pred_idx[i] else 'x'
        if marker == 'o':
            ax.scatter(true_idx[i], pred_idx[i], c=color, marker=marker, s=100, alpha=0.6, edgecolors='black')
        else:
            ax.scatter(true_idx[i], pred_idx[i], c=color, marker=marker, s=100, alpha=0.6)
    
    ax.plot([-0.5, len(EMOTION_LABELS)-0.5], [-0.5, len(EMOTION_LABELS)-0.5], 
            'k--', alpha=0.3, linewidth=2, label='Perfect Prediction')
    ax.set_xticks(range(len(EMOTION_LABELS)))
    ax.set_yticks(range(len(EMOTION_LABELS)))
    ax.set_xticklabels(EMOTION_LABELS, rotation=45, ha='right')
    ax.set_yticklabels(EMOTION_LABELS)
    ax.set_xlabel('True Emotion', fontweight='bold')
    ax.set_ylabel('Predicted Emotion', fontweight='bold')
    ax.set_title('Prediction Scatter\n(○=Correct, ×=Incorrect)', 
                 fontweight='bold', pad=15)
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()
    fname_scatter = os.path.join(outdir, f'{vid}_scatter.png')
    plt.savefig(fname_scatter, dpi=150, bbox_inches='tight')
    if show:
        plt.show()
    plt.close()

    # 3) Enhanced memory norm with moving average
    if mem_arr is not None:
        norms = np.linalg.norm(mem_arr, axis=1)
        
        fig, ax = plt.subplots(figsize=(14, 4))
        ax.plot(x, norms, marker='o', markersize=6, linewidth=2, 
                color='#3498db', alpha=0.7, label='Memory Norm')
        
        # Add moving average if sequence is long enough
        if len(norms) > 5:
            window = min(5, len(norms) // 3)
            moving_avg = np.convolve(norms, np.ones(window)/window, mode='valid')
            x_ma = x[window-1:]
            ax.plot(x_ma, moving_avg, linewidth=3, color='#e74c3c', 
                   alpha=0.8, label=f'{window}-point Moving Avg')
        
        # Highlight high memory points
        threshold = np.percentile(norms, 75)
        high_mem = norms > threshold
        ax.scatter(x[high_mem], norms[high_mem], s=100, c='red', 
                  marker='^', alpha=0.6, label='High Memory', zorder=5)
        
        ax.set_xlabel('Utterance Index', fontweight='bold')
        ax.set_ylabel('Memory Norm (L2)', fontweight='bold')
        ax.set_title('Temporal Memory Strength', 
                    fontweight='bold', pad=15)
        ax.legend(loc='best', framealpha=0.9)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        fname2 = os.path.join(outdir, f'{vid}_memnorm.png')
        plt.savefig(fname2, dpi=150, bbox_inches='tight')
        if show:
            plt.show()
        plt.close()

        # 4) Memory PCA visualization (first 2 components)
        if mem_arr.shape[1] >= 2 and seq_len > 3:
            from sklearn.decomposition import PCA
            
            pca = PCA(n_components=min(2, mem_arr.shape[1]))
            mem_pca = pca.fit_transform(mem_arr)
            
            fig, ax = plt.subplots(figsize=(10, 8))
            
            # Color by emotion
            for i in range(seq_len):
                color = emotion_colors.get(true_idx[i], '#95a5a6')
                ax.scatter(mem_pca[i, 0], mem_pca[i, 1], c=color, s=150, 
                          alpha=0.7, edgecolors='black', linewidth=1.5)
                ax.annotate(str(i), (mem_pca[i, 0], mem_pca[i, 1]), 
                           fontsize=8, ha='center', va='center')
            
            # Draw trajectory
            ax.plot(mem_pca[:, 0], mem_pca[:, 1], 'k-', alpha=0.2, linewidth=1)
            
            ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} var)', fontweight='bold')
            ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} var)', fontweight='bold')
            ax.set_title('Memory State Trajectory (PCA)', 
                        fontweight='bold', pad=15)
            
            # Add legend for emotions
            from matplotlib.patches import Patch
            legend_elements = [Patch(facecolor=emotion_colors[i], label=EMOTION_LABELS[i]) 
                             for i in range(len(EMOTION_LABELS))]
            ax.legend(handles=legend_elements, loc='best', framealpha=0.9)
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            fname_pca = os.path.join(outdir, f'{vid}_memory_pca.png')
            plt.savefig(fname_pca, dpi=150, bbox_inches='tight')
            if show:
                plt.show()
            plt.close()

    # 5) Prediction confidence over time (if log_prob available)
    if log_prob is not None:
        probs = np.exp(log_prob[:seq_len])
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
        
        # Top: Stacked area chart of all class probabilities
        ax1.stackplot(x, *[probs[:, i] for i in range(len(EMOTION_LABELS))],
                     labels=EMOTION_LABELS,
                     colors=[emotion_colors[i] for i in range(len(EMOTION_LABELS))],
                     alpha=0.7)
        ax1.set_ylabel('Probability', fontweight='bold')
        ax1.set_title('Model Confidence Distribution', 
                     fontweight='bold', pad=15)
        ax1.legend(loc='upper left', bbox_to_anchor=(1, 1), framealpha=0.9)
        ax1.set_ylim([0, 1])
        ax1.grid(True, alpha=0.3, axis='y')
        
        # Bottom: Prediction confidence (max probability)
        max_probs = np.max(probs, axis=1)
        pred_confidence = max_probs
        correct = (true_idx == pred_idx)
        
        colors_conf = ['green' if c else 'red' for c in correct]
        bars = ax2.bar(x, pred_confidence, color=colors_conf, alpha=0.6, edgecolor='black')
        ax2.axhline(y=np.mean(pred_confidence), color='blue', linestyle='--', 
                   linewidth=2, alpha=0.7, label=f'Mean: {np.mean(pred_confidence):.3f}')
        ax2.set_xlabel('Utterance Index', fontweight='bold')
        ax2.set_ylabel('Confidence (Max Prob)', fontweight='bold')
        ax2.set_title('Prediction Confidence (Green=Correct, Red=Incorrect)', 
                     fontweight='bold', pad=15)
        ax2.legend(loc='best', framealpha=0.9)
        ax2.set_ylim([0, 1])
        ax2.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        fname_conf = os.path.join(outdir, f'{vid}_confidence.png')
        plt.savefig(fname_conf, dpi=150, bbox_inches='tight')
        if show:
            plt.show()
        plt.close()

    saved_files = [fname1, fname_scatter]
    if mem_arr is not None:
        saved_files.extend([fname2, fname_pca])
    if log_prob is not None:
        saved_files.append(fname_conf)
    
    print(f'\n✓ Plots saved to {outdir}/')
    for f in saved_files:
        print(f'  - {os.path.basename(f)}')


def normalize_mem_array(mem_arr, seq_len):
    """Try to coerce mem_arr into shape (seq_len, dim) or return None if impossible."""
    if mem_arr is None:
        return None
    try:
        arr = np.array(mem_arr)
    except Exception:
        return None

    # if already 2D and first dim equals seq_len, ok
    if arr.ndim == 2 and arr.shape[0] == seq_len:
        return arr

    # if 2D and second dim equals seq_len, transpose
    if arr.ndim == 2 and arr.shape[1] == seq_len:
        return arr.T

    # if 1D and length divisible by seq_len, reshape to (seq_len, -1)
    if arr.ndim == 1 and seq_len > 0 and arr.size % seq_len == 0:
        mem_dim = arr.size // seq_len
        return arr.reshape(seq_len, mem_dim)

    # if 1D but not divisible, try to take first seq_len elements if they form a reasonable chunk
    # e_t might be concatenated node features, so try to extract per-node features
    if arr.ndim == 1 and seq_len > 0:
        # Check if we can reasonably split this - try common feature dimensions
        for possible_dim in [100, 150, 200, 128, 256, 64, 50]:
            if arr.size >= seq_len * possible_dim:
                # Take the first seq_len * possible_dim elements
                truncated = arr[:seq_len * possible_dim]
                if truncated.size % seq_len == 0:
                    return truncated.reshape(seq_len, -1)
        
        # Last resort: if array is longer than seq_len, assume first dimension could be squeezed
        if arr.size > seq_len:
            # Try to extract seq_len evenly-spaced samples
            mem_dim = arr.size // seq_len
            if mem_dim > 0:
                return arr[:seq_len * mem_dim].reshape(seq_len, mem_dim)

    # if higher-dim, try to flatten trailing dims and match seq_len
    if arr.ndim > 2:
        # collapse all dims after the first into feature dim
        # try first axis
        if arr.shape[0] == seq_len:
            return arr.reshape(seq_len, -1)
        # try last axis
        if arr.shape[-1] == seq_len:
            new = np.moveaxis(arr, -1, 0)
            return new.reshape(seq_len, -1)

    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='bestmodel.pth', help='path to model checkpoint')
    parser.add_argument('--split', choices=['train', 'test'], default='test', help='which split to sample from')
    parser.add_argument('--idx', type=int, default=None, help='index of conversation to sample (optional)')
    parser.add_argument('--seed', type=int, default=300, help='random seed for sampling')
    parser.add_argument('--no-cuda', action='store_true', default=False, help='disable CUDA')
    parser.add_argument('--outdir', default='sample_plots', help='directory to save plots')
    parser.add_argument('--track-idx', type=int, default=None, help='utterance index to track temporal evolution')
    parser.add_argument('--show', action='store_true', help='show plots interactively')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() and not args.no_cuda else 'cpu')

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # load dataset
    ds = IEMOCAPDataset(train=(args.split == 'train'))

    if args.idx is None:
        idx = random.randrange(len(ds))
    else:
        idx = args.idx

    batch = collate_single_sample(ds, idx)

    # build model and load checkpoint
    model = build_model(device)
    state = load_checkpoint(args.model, device)

    # try to adapt loaded state to model's load_state_dict
    try:
        model.load_state_dict(state)
    except RuntimeError:
        # try stripping 'module.' prefixes
        new_state = {}
        for k, v in state.items():
            new_k = k.replace('module.', '')
            new_state[new_k] = v
        try:
            model.load_state_dict(new_state)
        except Exception as e:
            print('Failed to load checkpoint into model:', e)
            print('Checkpoint keys (sample):', list(state.keys())[:10])
            return

    # prepare inputs (same unpacking as train_or_eval_graph_model)
    # batch structure from collate_fn: [textf, visuf, acouf, qmask, umask, label, vid_list]
    textf, visuf, acouf, qmask, umask, label, vid_list = batch

    # move tensors to device
    textf = safe_tensor_to(device, textf)
    qmask = safe_tensor_to(device, qmask)
    umask = safe_tensor_to(device, umask)
    label = safe_tensor_to(device, label)

    # compute lengths same as training
    batch_size = qmask.shape[1]
    lengths = [(umask[j] == 1).nonzero().tolist()[-1][0] + 1 for j in range(batch_size)]

    # forward pass
    with torch.no_grad():
        try:
            out = model(textf, qmask, umask, lengths, compute_temporal_loss=True)
        except TypeError:
            out = model(textf, qmask, umask, lengths)

    # parse outputs. Training expected: log_prob, e_i, e_n, e_t, e_l, temporal_loss
    log_prob = None
    e_i = e_n = e_t = e_l = temporal_loss = None
    if isinstance(out, (tuple, list)):
        if len(out) == 6:
            log_prob, e_i, e_n, e_t, e_l, temporal_loss = out
        elif len(out) == 5:
            log_prob, e_i, e_n, e_t, e_l = out
        else:
            log_prob = out[0]
    else:
        log_prob = out

    if log_prob is None:
        print('Model did not return log_prob. Exiting.')
        return

    preds = torch.argmax(log_prob, dim=1).cpu().numpy()

    # get vid to fetch utterance text if available
    vid = vid_list[0]
    sentences = ds.videoSentence[vid]
    true_labels = ds.videoLabels[vid]

    print('\nConversation id:', vid)
    print('Num utterances:', len(true_labels))
    if temporal_loss is not None:
        try:
            tl = temporal_loss.item() if isinstance(temporal_loss, torch.Tensor) else float(temporal_loss)
            print('Temporal loss (model):', tl)
        except Exception:
            print('Temporal loss (model):', temporal_loss)

    print('\nPer-utterance predictions and diagnostics:')
    print('-'*120)
    header = '{:>3} | {:<60} | {:<10} | {:<10} | {:<8} | {:s}'.format('Idx','Utterance (truncated)','True','Pred','MemNorm','Mem[0:6]')
    print(header)
    print('-'*120)

    mem_arr = None
    if isinstance(e_t, torch.Tensor):
        raw_mem = e_t.detach().cpu().numpy()
        mem_arr = normalize_mem_array(raw_mem, len(true_labels))
        if mem_arr is None:
            # couldn't coerce shape; skip memory diagnostics
            print('Warning: could not interpret temporal memory tensor shape:', raw_mem.shape)

    for i in range(len(true_labels)):
        sent = sentences[i] if i < len(sentences) else ''
        sent_trunc = (sent[:57] + '...') if len(sent) > 60 else sent
        true_name = EMOTION_LABELS[true_labels[i]] if true_labels[i] < len(EMOTION_LABELS) else str(true_labels[i])
        pred_idx = int(preds[i]) if i < len(preds) else -1
        pred_name = EMOTION_LABELS[pred_idx] if 0 <= pred_idx < len(EMOTION_LABELS) else str(pred_idx)

        mem_norm = ''
        mem_first = ''
        if mem_arr is not None and i < mem_arr.shape[0]:
            vec = mem_arr[i]
            mem_norm = f'{np.linalg.norm(vec):.4f}'
            mem_first = ','.join([f'{x:.3f}' for x in vec.flatten()[:6]])
        else:
            mem_norm = '-'
            mem_first = '-'

        line = '{:>3} | {:<60} | {:<10} | {:<10} | {:<8} | {:s}'.format(i, sent_trunc, true_name, pred_name, mem_norm, mem_first)
        print(line)

    print('-'*120)

    if isinstance(e_n, torch.Tensor):
        try:
            print('\nNode features `e_n` shape:', tuple(e_n.detach().cpu().shape))
        except Exception:
            pass
    if isinstance(e_i, torch.Tensor):
        try:
            print('Node indices `e_i` shape:', tuple(e_i.detach().cpu().shape))
        except Exception:
            pass

    # Generate main conversation plots
    print('\nGenerating conversation visualizations...')
    plot_conversation(vid, sentences, true_labels, preds, mem_arr, args.outdir, args.show, 
                     log_prob=log_prob.detach().cpu().numpy() if log_prob is not None else None,
                     qmask=qmask)

    # if user requested tracking of a specific utterance, run prefixes and record evolution
    if args.track_idx is not None:
        track_idx = int(args.track_idx)
        if track_idx < 0 or track_idx >= len(true_labels):
            print(f'--track-idx {track_idx} out of range for conversation length {len(true_labels)}')
        else:
            def run_prefix(prefix_len):
                # Create a fresh batch for this prefix by re-collating from dataset
                # Get the original sample and modify it to only include prefix utterances
                sample = ds[idx]
                # sample is a tuple: (textf, visuf, acouf, qmask, umask, label, vid)
                # Slice each to prefix_len
                prefix_sample = (
                    sample[0][:prefix_len],  # textf
                    sample[1][:prefix_len],  # visuf
                    sample[2][:prefix_len],  # acouf
                    sample[3][:prefix_len],  # qmask
                    sample[4][:prefix_len],  # umask
                    sample[5][:prefix_len],  # label
                    sample[6]  # vid (unchanged)
                )
                # Re-collate to get proper batch structure
                prefix_batch = ds.collate_fn([prefix_sample])
                
                # Unpack and move to device
                tf_p, visuf_p, acouf_p, qm_p, um_p, label_p, vid_list_p = prefix_batch
                tf_p = safe_tensor_to(device, tf_p)
                qm_p = safe_tensor_to(device, qm_p)
                um_p = safe_tensor_to(device, um_p)
                
                # Compute lengths
                batch_size_p = qm_p.shape[1]
                lengths_p = [(um_p[j] == 1).nonzero().tolist()[-1][0] + 1 for j in range(batch_size_p)]
                
                with torch.no_grad():
                    try:
                        outp = model(tf_p, qm_p, um_p, lengths_p, compute_temporal_loss=False)
                    except TypeError:
                        outp = model(tf_p, qm_p, um_p, lengths_p)
                return outp

            n_classes = len(EMOTION_LABELS)
            probs_over_time = []
            memvecs = []
            prefix_range = list(range(1, len(true_labels)+1))
            for p in prefix_range:
                outp = run_prefix(p)
                # parse output
                if isinstance(outp, (tuple, list)):
                    # prefer log_prob at index 0
                    lp = outp[0]
                    etp = outp[3] if len(outp) > 3 else None
                else:
                    lp = outp
                    etp = None

                # lp is log_prob over utterances in the truncated prefix
                lp_np = lp.detach().cpu().numpy()
                # probabilities
                probs = np.exp(lp_np)
                # utt available only if track_idx < p
                if track_idx < p:
                    utt_prob = probs[track_idx]
                    probs_over_time.append(utt_prob)
                    # collect memory vector for this utterance if present
                    if isinstance(etp, torch.Tensor):
                        rawm = etp.detach().cpu().numpy()
                        mem = normalize_mem_array(rawm, p)
                        if mem is not None and track_idx < mem.shape[0]:
                            memvecs.append(mem[track_idx])
                        else:
                            memvecs.append(None)
                    else:
                        memvecs.append(None)
                else:
                    # pad with NaNs for consistency
                    probs_over_time.append(np.full(n_classes, np.nan))
                    memvecs.append(None)

            # convert to arrays
            probs_over_time = np.stack(probs_over_time, axis=0)  # time x classes

            # plot class probability evolution
            if HAS_PLOTTING:
                import os
                os.makedirs(args.outdir, exist_ok=True)
                
                # Enhanced probability evolution plot
                fig, ax = plt.subplots(figsize=(12, 6))
                
                # Color palette matching emotion colors
                emotion_colors = ['#95a5a6', '#f39c12', '#3498db', '#e74c3c', '#9b59b6', '#e67e22']
                
                for c in range(n_classes):
                    ax.plot(prefix_range, probs_over_time[:, c], label=EMOTION_LABELS[c],
                           linewidth=2.5, marker='o', markersize=4, alpha=0.8,
                           color=emotion_colors[c])
                
                # Highlight the true label
                true_label = true_labels[track_idx]
                ax.axhline(y=1.0, color='green', linestyle=':', alpha=0.3, linewidth=2,
                          label=f'Target (True={EMOTION_LABELS[true_label]})')
                
                # Mark when utterance first becomes available (track_idx+1)
                ax.axvline(x=track_idx+1, color='red', linestyle='--', alpha=0.5, linewidth=2,
                          label=f'Utterance {track_idx} available')
                
                ax.set_xlabel('Prefix Length (utterances seen)', fontweight='bold', fontsize=12)
                ax.set_ylabel('Predicted Probability', fontweight='bold', fontsize=12)
                ax.set_title(f'Utterance {track_idx} - Emotion Prediction Evolution\n' +
                           f'True Label: {EMOTION_LABELS[true_label]}',
                           fontweight='bold', fontsize=13, pad=15)
                ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', framealpha=0.9)
                ax.grid(True, alpha=0.3)
                ax.set_ylim([-0.05, 1.05])
                plt.tight_layout()
                fname = f'{vid}_utt{track_idx}_evolution.png'
                fpath = os.path.join(args.outdir, fname)
                plt.savefig(fpath, dpi=150, bbox_inches='tight')
                if args.show:
                    plt.show()
                plt.close()

                # Enhanced memory norm plot with context
                memnorms = [np.linalg.norm(m) if (m is not None) else np.nan for m in memvecs]
                
                fig, ax = plt.subplots(figsize=(12, 5))
                ax.plot(prefix_range, memnorms, marker='o', markersize=6, linewidth=2.5,
                       color='#3498db', alpha=0.8, label='Memory Norm')
                
                # Add moving average if enough data
                if len(memnorms) > 5:
                    valid_norms = [n for n in memnorms if not np.isnan(n)]
                    if len(valid_norms) > 5:
                        window = min(5, len(valid_norms) // 3)
                        valid_arr = np.array(valid_norms)
                        moving_avg = np.convolve(valid_arr, np.ones(window)/window, mode='valid')
                        x_ma = np.arange(len(moving_avg)) + window - 1 + (track_idx + 1)
                        ax.plot(x_ma, moving_avg, linewidth=3, color='#e74c3c',
                               alpha=0.7, label=f'{window}-point Moving Avg')
                
                # Mark when utterance first becomes available
                ax.axvline(x=track_idx+1, color='red', linestyle='--', alpha=0.5, linewidth=2,
                          label=f'Utterance {track_idx} available')
                
                # Highlight final value
                if not np.isnan(memnorms[-1]):
                    ax.scatter([prefix_range[-1]], [memnorms[-1]], s=200, c='red',
                             marker='*', zorder=5, label='Final State', edgecolors='black', linewidth=1.5)
                
                ax.set_xlabel('Prefix Length (utterances seen)', fontweight='bold', fontsize=12)
                ax.set_ylabel('Memory Norm (L2)', fontweight='bold', fontsize=12)
                ax.set_title(f'Utterance {track_idx} - Temporal Memory Evolution',
                           fontweight='bold', fontsize=13, pad=15)
                ax.legend(loc='best', framealpha=0.9)
                ax.grid(True, alpha=0.3)
                plt.tight_layout()
                fname2 = f'{vid}_utt{track_idx}_memnorm.png'
                fpath2 = os.path.join(args.outdir, fname2)
                plt.savefig(fpath2, dpi=150, bbox_inches='tight')
                if args.show:
                    plt.show()
                plt.close()

                # New: Prediction stability plot
                # Show how prediction changes over context windows
                pred_changes = []
                prev_pred = None
                for p_idx in range(len(probs_over_time)):
                    curr_pred = np.argmax(probs_over_time[p_idx])
                    if prev_pred is not None and not np.isnan(probs_over_time[p_idx]).any():
                        pred_changes.append(1 if curr_pred != prev_pred else 0)
                    else:
                        pred_changes.append(0)
                    prev_pred = curr_pred if not np.isnan(probs_over_time[p_idx]).any() else prev_pred
                
                fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
                
                # Top: Entropy (uncertainty)
                valid_probs = probs_over_time[~np.isnan(probs_over_time).any(axis=1)]
                valid_x = [prefix_range[i] for i in range(len(probs_over_time)) 
                          if not np.isnan(probs_over_time[i]).any()]
                
                if len(valid_probs) > 0:
                    entropy = -np.sum(valid_probs * np.log(valid_probs + 1e-10), axis=1)
                    ax1.plot(valid_x, entropy, marker='o', markersize=6, linewidth=2.5,
                            color='#9b59b6', alpha=0.8)
                    ax1.axhline(y=np.log(n_classes), color='red', linestyle='--', alpha=0.5,
                               label=f'Max Entropy (uniform dist.)')
                    ax1.axvline(x=track_idx+1, color='red', linestyle='--', alpha=0.5, linewidth=2)
                
                ax1.set_ylabel('Prediction Entropy\n(Uncertainty)', fontweight='bold', fontsize=11)
                ax1.set_title(f'Utterance {track_idx} - Prediction Stability Analysis', 
                            fontweight='bold', fontsize=13, pad=15)
                ax1.legend(loc='best', framealpha=0.9)
                ax1.grid(True, alpha=0.3)
                
                # Bottom: Prediction changes
                ax2.bar(prefix_range[1:], pred_changes[1:], color='#e74c3c', alpha=0.7,
                       edgecolor='black')
                ax2.axvline(x=track_idx+1, color='red', linestyle='--', alpha=0.5, linewidth=2,
                           label=f'Utterance {track_idx} available')
                ax2.set_xlabel('Prefix Length (utterances seen)', fontweight='bold', fontsize=12)
                ax2.set_ylabel('Prediction Changed', fontweight='bold', fontsize=11)
                ax2.set_ylim([-0.1, 1.5])
                ax2.set_yticks([0, 1])
                ax2.set_yticklabels(['Stable', 'Changed'])
                ax2.legend(loc='best', framealpha=0.9)
                ax2.grid(True, alpha=0.3, axis='x')
                
                plt.tight_layout()
                fname3 = f'{vid}_utt{track_idx}_stability.png'
                fpath3 = os.path.join(args.outdir, fname3)
                plt.savefig(fpath3, dpi=150, bbox_inches='tight')
                if args.show:
                    plt.show()
                plt.close()

                print(f'\n✓ Tracking plots saved to {args.outdir}/')
                print(f'  - {fname}')
                print(f'  - {fname2}')
                print(f'  - {fname3}')
            else:
                print('Plotting libraries not available; skipping evolution plots.')

    print('\nDone.')


if __name__ == '__main__':
    main()
