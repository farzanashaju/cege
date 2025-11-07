import numpy as np
import argparse
import time
import pickle
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from dataloader import IEMOCAPDataset
from model import MaskedNLLLoss, LSTMModel, GRUModel, DialogueGCNModel, DialogueGCNTemporalModel
from sklearn.metrics import (f1_score, accuracy_score, precision_recall_fscore_support)

EMOTION_LABELS = ['Neutral', 'Happy', 'Sad', 'Angry', 'Excited', 'Frustrated']

seed = 100

def seed_everything(seed=seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

# split the training set into training and validation sets
def get_train_valid_sampler(trainset, valid=0.1):
    size = len(trainset)
    idx = list(range(size))
    split = int(valid*size)
    return SubsetRandomSampler(idx[split:]), SubsetRandomSampler(idx[:split])

# dataset loaders
def get_IEMOCAP_loaders(batch_size=32, valid=0.1, num_workers=0, pin_memory=False):
    trainset = IEMOCAPDataset()
    train_sampler, valid_sampler = get_train_valid_sampler(trainset, valid)

    train_loader = DataLoader(trainset,
                              batch_size=batch_size,
                              sampler=train_sampler,
                              collate_fn=trainset.collate_fn,
                              num_workers=num_workers,
                              pin_memory=pin_memory)

    valid_loader = DataLoader(trainset,
                              batch_size=batch_size,
                              sampler=valid_sampler,
                              collate_fn=trainset.collate_fn,
                              num_workers=num_workers,
                              pin_memory=pin_memory)

    testset = IEMOCAPDataset(train=False)
    test_loader = DataLoader(testset,
                             batch_size=batch_size,
                             collate_fn=testset.collate_fn,
                             num_workers=num_workers,
                             pin_memory=pin_memory)

    return train_loader, valid_loader, test_loader


# print metrics for each class
def print_per_class_metrics(labels, preds, phase='Test'):
    print(f"\n{'='*70}")
    print(f"{phase} Results - Per-Class Metrics")
    print(f"{'='*70}")
    
    # per-class precision, recall, f1
    precision, recall, f1, support = precision_recall_fscore_support(
        labels, preds, labels=range(len(EMOTION_LABELS)), zero_division=0
    )
    
    # print header
    print(f"{'Class':<15} {'Accuracy':<12} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Support':<10}")
    print("-" * 70)
    
    # print per-class metrics
    for i, label in enumerate(EMOTION_LABELS):
        # per-class accuracy
        mask = labels == i
        class_acc = accuracy_score(labels[mask], preds[mask]) * 100 if mask.sum() > 0 else 0.0
        
        print(f"{label:<15} {class_acc:>10.2f}% {precision[i]:>11.4f} {recall[i]:>11.4f} {f1[i]:>11.4f} {int(support[i]):>9}")
    
    # print weighted averages
    print("-" * 70)
    weighted_precision = np.average(precision, weights=support)
    weighted_recall = np.average(recall, weights=support)
    weighted_f1 = np.average(f1, weights=support)
    macro_f1 = np.mean(f1)
    
    print(f"{'Weighted Avg':<15} {accuracy_score(labels, preds)*100:>10.2f}% {weighted_precision:>11.4f} {weighted_recall:>11.4f} {weighted_f1:>11.4f} {int(support.sum()):>9}")
    print(f"{'Macro Avg':<15} {'-':>10} {np.mean(precision):>11.4f} {np.mean(recall):>11.4f} {macro_f1:>11.4f} {'-':>9}")
    print(f"{'='*70}\n")
    
    return {
        'accuracy': accuracy_score(labels, preds) * 100,
        'weighted_f1': weighted_f1 * 100,
        'macro_f1': macro_f1 * 100,
        'per_class_f1': f1 * 100,
        'per_class_acc': [accuracy_score(labels[labels == i], preds[labels == i]) * 100 
                         if (labels == i).sum() > 0 else 0.0 for i in range(len(EMOTION_LABELS))]
    }

# main training / evaluation loop
# forward pass, compute cross-entropy + temporal consistency loss, backward pass
def train_or_eval_graph_model(model, loss_function, dataloader, epoch, cuda, optimizer=None, 
                             train=False, lambda_temp=0.1):
    losses, preds, labels = [], [], []
    scores, vids = [], []
    temporal_losses = []

    ei, et, en, el = torch.empty(0).type(torch.LongTensor), torch.empty(0).type(torch.LongTensor), torch.empty(0), []

    if cuda:
        ei, et, en = ei.cuda(), et.cuda(), en.cuda()

    assert not train or optimizer!=None
    if train:
        model.train()
    else:
        model.eval()

    seed_everything()
    for data in dataloader:
        if train:
            optimizer.zero_grad()
        
        textf, visuf, acouf, qmask, umask, label = [d.cuda() for d in data[:-1]] if cuda else data[:-1]

        lengths = [(umask[j] == 1).nonzero().tolist()[-1][0] + 1 for j in range(len(umask))]

        # check if model supports temporal loss computation
        if hasattr(model, '__class__') and 'Temporal' in model.__class__.__name__:
            output = model(textf, qmask, umask, lengths, compute_temporal_loss=True)
            if len(output) == 6:
                log_prob, e_i, e_n, e_t, e_l, temporal_loss = output
                temporal_losses.append(temporal_loss.item() if isinstance(temporal_loss, torch.Tensor) else temporal_loss)
            else:
                log_prob, e_i, e_n, e_t, e_l = output
                temporal_loss = torch.tensor(0.0)
        else:
            log_prob, e_i, e_n, e_t, e_l = model(textf, qmask, umask, lengths)
            temporal_loss = torch.tensor(0.0)
        
        label = torch.cat([label[j][:lengths[j]] for j in range(len(label))])
        ce_loss = loss_function(log_prob, label)
        
        # combined loss: cross-entropy loss + temporal consistency penalty
        total_loss = ce_loss + lambda_temp * temporal_loss if isinstance(temporal_loss, torch.Tensor) and temporal_loss.item() != 0 else ce_loss

        ei = torch.cat([ei, e_i], dim=1)
        et = torch.cat([et, e_t])
        en = torch.cat([en, e_n])
        el += e_l

        preds.append(torch.argmax(log_prob, 1).cpu().numpy())
        labels.append(label.cpu().numpy())
        losses.append(total_loss.item())

        if train:
            total_loss.backward()
            if hasattr(args, 'tensorboard') and args.tensorboard:
                for name, param in model.named_parameters():
                    # Skip if gradient is None (e.g., frozen params or unused)
                    if param.grad is None:
                        continue
                    # Move grad to CPU for safe logging
                    try:
                        grad_to_log = param.grad.detach().cpu()
                    except Exception:
                        grad_to_log = param.grad
                    writer.add_histogram(name, grad_to_log, epoch)
            optimizer.step()

    if preds!=[]:
        preds  = np.concatenate(preds)
        labels = np.concatenate(labels)
    else:
        return float('nan'), float('nan'), [], [], float('nan'), [], [], [], [], [], 0.0

    vids += data[-1]
    ei = ei.data.cpu().numpy()
    et = et.data.cpu().numpy()
    en = en.data.cpu().numpy()
    el = np.array(el)
    labels = np.array(labels)
    preds = np.array(preds)
    vids = np.array(vids)

    avg_loss = round(np.sum(losses)/len(losses), 4)
    avg_accuracy = round(accuracy_score(labels, preds)*100, 2)
    avg_fscore = round(f1_score(labels,preds, average='weighted')*100, 2)
    avg_temporal_loss = round(np.mean(temporal_losses), 4) if temporal_losses else 0.0

    return avg_loss, avg_accuracy, labels, preds, avg_fscore, vids, ei, et, en, el, avg_temporal_loss


if __name__ == '__main__':

    path = './saved/IEMOCAP/'

    parser = argparse.ArgumentParser()

    parser.add_argument('--no-cuda', action='store_true', default=False, help='does not use GPU')
    parser.add_argument('--base-model', default='LSTM', help='base recurrent model, must be LSTM/GRU')
    parser.add_argument('--graph-model', action='store_true', default=False, help='whether to use graph model after recurrent encoding')
    parser.add_argument('--nodal-attention', action='store_true', default=False, help='whether to use nodal attention in graph model')
    parser.add_argument('--windowp', type=int, default=10, help='context window size for constructing edges in graph model for past utterances')
    parser.add_argument('--windowf', type=int, default=10, help='context window size for constructing edges in graph model for future utterances')
    parser.add_argument('--lr', type=float, default=0.0001, metavar='LR', help='learning rate')
    parser.add_argument('--l2', type=float, default=0.00001, metavar='L2', help='L2 regularization weight')
    parser.add_argument('--rec-dropout', type=float, default=0.1, metavar='rec_dropout', help='rec_dropout rate')
    parser.add_argument('--dropout', type=float, default=0.5, metavar='dropout', help='dropout rate')
    parser.add_argument('--batch-size', type=int, default=32, metavar='BS', help='batch size')
    parser.add_argument('--epochs', type=int, default=60, metavar='E', help='number of epochs')
    parser.add_argument('--class-weight', action='store_true', default=False, help='use class weights')
    parser.add_argument('--active-listener', action='store_true', default=False, help='active listener')
    parser.add_argument('--attention', default='general', help='attention type')
    parser.add_argument('--tensorboard', action='store_true', default=False, help='Enables tensorboard log')

    # temporal memory arguments
    parser.add_argument('--temporal-model', action='store_true', default=False, help='whether to use temporal memory model')
    parser.add_argument('--lambda-temp', type=float, default=0.1, metavar='lambda_temp', help='weight for temporal consistency loss')
    parser.add_argument('--temporal-decay', type=float, default=0.1, metavar='temporal_decay', help='temporal decay rate for attention')
    parser.add_argument('--edge-prune-threshold', type=float, default=0.1, metavar='tau_remove', help='threshold for edge pruning')

    args = parser.parse_args()
    print(args)

    args.cuda = torch.cuda.is_available() and not args.no_cuda
    if args.cuda:
        print('Running on GPU')
    else:
        print('Running on CPU')

    if args.tensorboard:
        from tensorboardX import SummaryWriter
        writer = SummaryWriter()

    n_classes  = 6
    cuda       = args.cuda
    n_epochs   = args.epochs
    batch_size = args.batch_size

    D_m = 100
    D_g = 150
    D_p = 150
    D_e = 100
    D_h = 100
    D_a = 100
    graph_h = 100

    if args.graph_model:
        seed_everything()
        if args.temporal_model:
            # use temporal memory-enhanced model
            model = DialogueGCNTemporalModel(args.base_model,
                                     D_m, D_g, D_p, D_e, D_h, D_a, graph_h,
                                     n_speakers=2,
                                     max_seq_len=110,
                                     window_past=args.windowp,
                                     window_future=args.windowf,
                                     n_classes=n_classes,
                                     listener_state=args.active_listener,
                                     context_attention=args.attention,
                                     dropout=args.dropout,
                                     nodal_attention=args.nodal_attention,
                                     temporal_decay=args.temporal_decay,
                                     edge_prune_threshold=args.edge_prune_threshold,
                                     no_cuda=args.no_cuda)
            print('Graph NN with Temporal Memory and', args.base_model, 'as base model.')
            name = 'GraphTemporal'
        else:
            model = DialogueGCNModel(args.base_model,
                                 D_m, D_g, D_p, D_e, D_h, D_a, graph_h,
                                 n_speakers=2,
                                 max_seq_len=110,
                                 window_past=args.windowp,
                                 window_future=args.windowf,
                                 n_classes=n_classes,
                                 listener_state=args.active_listener,
                                 context_attention=args.attention,
                                 dropout=args.dropout,
                                 nodal_attention=args.nodal_attention,
                                 no_cuda=args.no_cuda)

        print('Graph NN with', args.base_model, 'as base model.')
        name = 'Graph'

    else:
        if args.base_model == 'GRU':
            model = GRUModel(D_m, D_e, D_h, 
                              n_classes=n_classes, 
                              dropout=args.dropout)
            print('Basic GRU Model.')

        elif args.base_model == 'LSTM':
            model = LSTMModel(D_m, D_e, D_h, 
                              n_classes=n_classes, 
                              dropout=args.dropout)
            print('Basic LSTM Model.')

        else:
            print('Base model must be LSTM / GRU')
            raise NotImplementedError

        name = 'Base'

    if cuda:
        model.cuda()

    # class weights for iemocap
    loss_weights = torch.FloatTensor([1/0.086747,
                                      1/0.144406,
                                      1/0.227883,
                                      1/0.160585,
                                      1/0.127711,
                                      1/0.252668])
    

    if args.class_weight:
        if args.graph_model:
            loss_function  = nn.NLLLoss(loss_weights.cuda() if cuda else loss_weights)
        else:
            loss_function  = MaskedNLLLoss(loss_weights.cuda() if cuda else loss_weights)
    else:
        if args.graph_model:
            loss_function = nn.NLLLoss()
        else:
            loss_function = MaskedNLLLoss()

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.l2)

    train_loader, valid_loader, test_loader = get_IEMOCAP_loaders(valid=0.0,
                                                                  batch_size=batch_size,
                                                                  num_workers=0)

    best_fscore, best_loss, best_label, best_pred, best_mask = None, None, None, None, None
    all_fscore, all_acc, all_loss = [], [], []
    per_epoch_metrics = []

    print(f"\n{'='*70}")
    print(f"Training Configuration")
    print(f"{'='*70}")
    print(f"Model: {name} {args.base_model}")
    print(f"Graph Model: {args.graph_model}")
    print(f"Temporal Model: {args.temporal_model}")
    print(f"Lambda Temp: {args.lambda_temp}")
    print(f"Batch Size: {batch_size}")
    print(f"Epochs: {n_epochs}")
    print(f"Learning Rate: {args.lr}")
    print(f"L2 Regularization: {args.l2}")
    print(f"{'='*70}\n")

    for e in range(n_epochs):
        start_time = time.time()

        if args.graph_model:
            train_loss, train_acc, train_labels, train_preds, train_fscore, _, _, _, _, _, train_temp_loss = train_or_eval_graph_model(
                model, loss_function, train_loader, e, cuda, optimizer, True, args.lambda_temp)
            valid_loss, valid_acc, _, _, valid_fscore, _, _, _, _, _, valid_temp_loss = train_or_eval_graph_model(
                model, loss_function, valid_loader, e, cuda, lambda_temp=args.lambda_temp)
            test_loss, test_acc, test_labels, test_preds, test_fscore, _, _, _, _, _, test_temp_loss = train_or_eval_graph_model(
                model, loss_function, test_loader, e, cuda, lambda_temp=args.lambda_temp)
            all_fscore.append(test_fscore)
            # record per-epoch summary metrics
            per_epoch_metrics.append(print_per_class_metrics(test_labels, test_preds, phase='Test'))
        else:
            raise NotImplementedError("Only graph models. Use --graph-model flag.")

        if args.tensorboard:
            writer.add_scalar('test: accuracy/loss', test_acc/test_loss, e)
            writer.add_scalar('train: accuracy/loss', train_acc/train_loss, e)

        # print epoch summary
        print(f"\nEpoch {e+1}/{n_epochs}")
        print(f"  Train - Loss: {train_loss:.4f}, Acc: {train_acc:.2f}%, F1: {train_fscore:.2f}%")
        print(f"  Valid - Loss: {valid_loss:.4f}, Acc: {valid_acc:.2f}%, F1: {valid_fscore:.2f}%")
        print(f"  Test  - Loss: {test_loss:.4f}, Acc: {test_acc:.2f}%, F1: {test_fscore:.2f}%")
        if args.temporal_model:
            print(f"  Temporal Loss - Train: {train_temp_loss:.4f}, Valid: {valid_temp_loss:.4f}, Test: {test_temp_loss:.4f}")

        # print per-class metrics for test set
        # already printed and stored when using graph_model; in case not stored, call again
        if not per_epoch_metrics:
            final_metrics = print_per_class_metrics(test_labels, test_preds, phase='Test')
            per_epoch_metrics.append(final_metrics)
        else:
            # print last appended metrics for visibility (function already prints)
            pass

        print(f"Time for epoch: {round(time.time()-start_time, 2)} sec")

    if args.tensorboard:
        writer.close()

    print(f"\n{'='*70}")
    print(f'Final Results')
    print(f"{'='*70}")
    # best epoch by weighted F1
    if len(all_fscore) > 0:
        best_idx = int(np.argmax(all_fscore))
        print(f'Best Weighted F1: {all_fscore[best_idx]:.2f}%')
        print(f'Best Epoch: {best_idx + 1}')
        # print corresponding weighted accuracy if available
        if best_idx < len(per_epoch_metrics):
            best_metrics = per_epoch_metrics[best_idx]
            print(f"Best Epoch - Weighted Accuracy: {best_metrics['accuracy']:.2f}%")
            print(f"Best Epoch - Weighted F1: {best_metrics['weighted_f1']:.2f}%")
    else:
        print('No epochs were run.')

    # final epoch metrics (last epoch)
    if len(per_epoch_metrics) > 0:
        final_metrics = per_epoch_metrics[-1]
        print(f"Final Epoch - Weighted Accuracy: {final_metrics['accuracy']:.2f}%")
        print(f"Final Epoch - Weighted F1: {final_metrics['weighted_f1']:.2f}%")
    print(f"{'='*70}\n")
