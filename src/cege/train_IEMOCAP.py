"""
Training script for CEGE model on IEMOCAP dataset.
"""

import os
import argparse
import time
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from sklearn.metrics import f1_score, accuracy_score, classification_report

from model import CEGEModel, MaskedNLLLoss
from dataloader import get_IEMOCAP_loaders


def seed_everything(seed=42):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train_epoch(model, dataloader, optimizer, criterion, device, lambda_reg=0.00001, lambda_temp=0.01):
    """
    Train for one epoch.
    
    Loss = L_CE + lambda_reg * ||theta||_2^2 + lambda_temp * temporal_consistency
    """
    model.train()
    
    total_loss = 0
    total_ce_loss = 0
    total_reg_loss = 0
    total_temp_loss = 0
    all_preds = []
    all_labels = []
    all_masks = []
    
    pbar = tqdm(dataloader, desc="Training", leave=False)
    
    for batch_idx, (features, qmask, labels, umask, seq_lengths, conv_ids) in enumerate(pbar):
        # Move to device
        features = features.to(device)  # (seq_len, batch, 100)
        qmask = qmask.to(device)  # (seq_len, batch, 2)
        labels = labels.to(device)  # (seq_len, batch)
        umask = umask.to(device)  # (batch, seq_len)
        
        optimizer.zero_grad()
        
        # Forward pass
        try:
            log_prob, edge_indices, edge_weights = model(features, qmask, umask, seq_lengths)
            
            # Prepare labels for loss
            labels_flat = []
            for b in range(len(seq_lengths)):
                labels_flat.extend(labels[:seq_lengths[b], b].tolist())
            labels_flat = torch.LongTensor(labels_flat).to(device)
            
            # Cross-entropy loss
            ce_loss = criterion(log_prob, labels_flat)
            
            # L2 regularization
            reg_loss = 0
            for param in model.parameters():
                reg_loss += torch.norm(param, p=2)
            reg_loss = lambda_reg * reg_loss
            
            # Temporal consistency loss (placeholder - would need memory tracking)
            temp_loss = torch.tensor(0.0).to(device)
            
            # Total loss
            loss = ce_loss + reg_loss + temp_loss
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            
            # Track metrics
            total_loss += loss.item()
            total_ce_loss += ce_loss.item()
            total_reg_loss += reg_loss.item()
            total_temp_loss += temp_loss.item()
            
            # Predictions
            preds = torch.argmax(log_prob, dim=1).cpu().numpy()
            labels_np = labels_flat.cpu().numpy()
            
            all_preds.extend(preds)
            all_labels.extend(labels_np)
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'ce': f'{ce_loss.item():.4f}'
            })
            
        except Exception as e:
            print(f"\nError in batch {batch_idx}: {e}")
            print(f"Seq lengths: {seq_lengths}")
            print(f"Features shape: {features.shape}")
            continue
    
    # Calculate metrics
    avg_loss = total_loss / len(dataloader)
    avg_ce_loss = total_ce_loss / len(dataloader)
    avg_reg_loss = total_reg_loss / len(dataloader)
    
    accuracy = accuracy_score(all_labels, all_preds) * 100
    f1 = f1_score(all_labels, all_preds, average='weighted') * 100
    
    return avg_loss, avg_ce_loss, avg_reg_loss, accuracy, f1


def evaluate(model, dataloader, criterion, device):
    """Evaluate model on validation/test set."""
    model.eval()
    
    total_loss = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        pbar = tqdm(dataloader, desc="Evaluating", leave=False)
        
        for batch_idx, (features, qmask, labels, umask, seq_lengths, conv_ids) in enumerate(pbar):
            # Move to device
            features = features.to(device)
            qmask = qmask.to(device)
            labels = labels.to(device)
            umask = umask.to(device)
            
            try:
                # Forward pass
                log_prob, edge_indices, edge_weights = model(features, qmask, umask, seq_lengths)
                
                # Prepare labels
                labels_flat = []
                for b in range(len(seq_lengths)):
                    labels_flat.extend(labels[:seq_lengths[b], b].tolist())
                labels_flat = torch.LongTensor(labels_flat).to(device)
                
                # Loss
                loss = criterion(log_prob, labels_flat)
                total_loss += loss.item()
                
                # Predictions
                preds = torch.argmax(log_prob, dim=1).cpu().numpy()
                labels_np = labels_flat.cpu().numpy()
                
                all_preds.extend(preds)
                all_labels.extend(labels_np)
                
            except Exception as e:
                print(f"\nError in eval batch {batch_idx}: {e}")
                continue
    
    # Calculate metrics
    avg_loss = total_loss / len(dataloader)
    accuracy = accuracy_score(all_labels, all_preds) * 100
    f1 = f1_score(all_labels, all_preds, average='weighted') * 100
    
    return avg_loss, accuracy, f1, all_labels, all_preds


def main():
    parser = argparse.ArgumentParser(description='Train CEGE model on IEMOCAP')
    
    # Data paths
    parser.add_argument('--train-encodings', type=str, 
                       default='../../iemocap-encodings/train_encodings.npz',
                       help='Path to training TextCNN encodings')
    parser.add_argument('--test-encodings', type=str,
                       default='../../iemocap-encodings/test_encodings.npz',
                       help='Path to test TextCNN encodings')
    parser.add_argument('--train-data', type=str,
                       default='../../iemocap/train.txt',
                       help='Path to training IEMOCAP data')
    parser.add_argument('--test-data', type=str,
                       default='../../iemocap/test.txt',
                       help='Path to test IEMOCAP data')
    
    # Model hyperparameters
    parser.add_argument('--D-m', type=int, default=100, help='Input feature dimension')
    parser.add_argument('--D-e', type=int, default=100, help='Sequential encoder hidden dim')
    parser.add_argument('--D-h', type=int, default=100, help='Classification hidden dim')
    parser.add_argument('--speaker-state-dim', type=int, default=150, help='Speaker state dimension')
    parser.add_argument('--conv-state-dim', type=int, default=150, help='Conversation state dimension')
    parser.add_argument('--memory-dim', type=int, default=200, help='Temporal memory dimension')
    parser.add_argument('--gcn-hidden-dim', type=int, default=200, help='GCN hidden dimension')
    parser.add_argument('--n-speakers', type=int, default=2, help='Number of speakers')
    parser.add_argument('--n-classes', type=int, default=6, help='Number of emotion classes')
    parser.add_argument('--n-relations', type=int, default=8, help='Number of relation types')
    parser.add_argument('--dropout', type=float, default=0.5, help='Dropout rate')
    parser.add_argument('--tau-remove', type=float, default=0.1, help='Edge removal threshold')
    parser.add_argument('--tau-create', type=float, default=0.7, help='Edge creation threshold')
    parser.add_argument('--lambda-decay', type=float, default=0.1, help='Temporal decay parameter')
    
    # Training hyperparameters
    parser.add_argument('--batch-size', type=int, default=8, help='Batch size (smaller for memory)')
    parser.add_argument('--epochs', type=int, default=60, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate')
    parser.add_argument('--lambda-reg', type=float, default=0.00001, help='L2 regularization weight')
    parser.add_argument('--lambda-temp', type=float, default=0.01, help='Temporal consistency weight')
    parser.add_argument('--valid-split', type=float, default=0.1, help='Validation split fraction')
    
    # Other
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--no-cuda', action='store_true', help='Disable CUDA')
    parser.add_argument('--save-dir', type=str, default='./checkpoints', help='Directory to save models')
    parser.add_argument('--log-interval', type=int, default=1, help='Logging interval')
    
    args = parser.parse_args()
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() and not args.no_cuda else 'cpu')
    print(f"\n{'='*80}")
    print(f"CEGE: Conversational Emotion Graph Evolution")
    print(f"{'='*80}")
    print(f"Device: {device}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.lr}")
    print(f"Epochs: {args.epochs}")
    
    # Set seed
    seed_everything(args.seed)
    
    # Create save directory
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Load data
    print(f"\n{'='*80}")
    print("Loading Data")
    print(f"{'='*80}")
    train_loader, valid_loader, test_loader = get_IEMOCAP_loaders(
        train_encodings=args.train_encodings,
        test_encodings=args.test_encodings,
        train_data=args.train_data,
        test_data=args.test_data,
        batch_size=args.batch_size,
        num_workers=0,
        valid_split=args.valid_split
    )
    
    # Create model
    print(f"\n{'='*80}")
    print("Initializing CEGE Model")
    print(f"{'='*80}")
    model = CEGEModel(
        D_m=args.D_m,
        D_e=args.D_e,
        D_h=args.D_h,
        speaker_state_dim=args.speaker_state_dim,
        conv_state_dim=args.conv_state_dim,
        memory_dim=args.memory_dim,
        gcn_hidden_dim=args.gcn_hidden_dim,
        n_speakers=args.n_speakers,
        n_classes=args.n_classes,
        n_relations=args.n_relations,
        dropout=args.dropout,
        tau_remove=args.tau_remove,
        tau_create=args.tau_create,
        lambda_decay=args.lambda_decay
    ).to(device)
    
    # Loss and optimizer
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.lambda_reg)
    
    # Training loop
    print(f"\n{'='*80}")
    print("Training")
    print(f"{'='*80}")
    
    best_valid_f1 = 0
    best_test_f1 = 0
    train_losses = []
    valid_f1s = []
    test_f1s = []
    
    for epoch in range(1, args.epochs + 1):
        start_time = time.time()
        
        print(f"\nEpoch {epoch}/{args.epochs}")
        print("-" * 80)
        
        # Train
        train_loss, train_ce, train_reg, train_acc, train_f1 = train_epoch(
            model, train_loader, optimizer, criterion, device,
            lambda_reg=args.lambda_reg, lambda_temp=args.lambda_temp
        )
        
        # Validate
        valid_loss, valid_acc, valid_f1, _, _ = evaluate(model, valid_loader, criterion, device)
        
        # Test
        test_loss, test_acc, test_f1, test_labels, test_preds = evaluate(
            model, test_loader, criterion, device
        )
        
        # Track best
        if valid_f1 > best_valid_f1:
            best_valid_f1 = valid_f1
            # Save best model
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'valid_f1': valid_f1,
                'test_f1': test_f1,
                'args': args
            }, os.path.join(args.save_dir, 'best_model.pt'))
        
        if test_f1 > best_test_f1:
            best_test_f1 = test_f1
        
        # Log
        elapsed = time.time() - start_time
        print(f"\nResults:")
        print(f"  Train - Loss: {train_loss:.4f}, Acc: {train_acc:.2f}%, F1: {train_f1:.2f}%")
        print(f"  Valid - Loss: {valid_loss:.4f}, Acc: {valid_acc:.2f}%, F1: {valid_f1:.2f}%")
        print(f"  Test  - Loss: {test_loss:.4f}, Acc: {test_acc:.2f}%, F1: {test_f1:.2f}%")
        print(f"  Best Valid F1: {best_valid_f1:.2f}%, Best Test F1: {best_test_f1:.2f}%")
        print(f"  Time: {elapsed:.2f}s")
        
        train_losses.append(train_loss)
        valid_f1s.append(valid_f1)
        test_f1s.append(test_f1)
        
        # Save checkpoint
        if epoch % 10 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'valid_f1': valid_f1,
                'test_f1': test_f1
            }, os.path.join(args.save_dir, f'checkpoint_epoch_{epoch}.pt'))
    
    # Final evaluation
    print(f"\n{'='*80}")
    print("Final Test Performance")
    print(f"{'='*80}")
    
    # Load best model
    checkpoint = torch.load(os.path.join(args.save_dir, 'best_model.pt'), weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    test_loss, test_acc, test_f1, test_labels, test_preds = evaluate(
        model, test_loader, criterion, device
    )
    
    print(f"\nBest Model Performance:")
    print(f"  Test Accuracy: {test_acc:.2f}%")
    print(f"  Test F1: {test_f1:.2f}%")
    
    # Classification report
    label_names = ['happy', 'sad', 'neutral', 'angry', 'excited', 'frustrated']
    print(f"\nClassification Report:")
    print(classification_report(test_labels, test_preds, target_names=label_names, digits=4))
    
    print(f"\n{'='*80}")
    print("Training Complete!")
    print(f"{'='*80}")
    print(f"Best model saved to: {os.path.join(args.save_dir, 'best_model.pt')}")


if __name__ == '__main__':
    main()
