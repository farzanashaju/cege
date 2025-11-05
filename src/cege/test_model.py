"""
Quick test script to verify CEGE model components work correctly.
"""

import torch
import numpy as np
from model import CEGEModel
from dataloader import IEMOCAPDataset

def test_model_initialization():
    """Test if model initializes correctly."""
    print("="*80)
    print("Testing Model Initialization")
    print("="*80)
    
    try:
        model = CEGEModel(
            D_m=100,
            D_e=100,
            D_h=100,
            speaker_state_dim=150,
            conv_state_dim=150,
            memory_dim=200,
            gcn_hidden_dim=200,
            n_speakers=2,
            n_classes=6,
            n_relations=8,
            dropout=0.5
        )
        print("\n✓ Model initialized successfully!")
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"\nTotal parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        
        return True
    except Exception as e:
        print(f"\n✗ Model initialization failed: {e}")
        return False


def test_dataloader():
    """Test if dataloader works."""
    print("\n" + "="*80)
    print("Testing DataLoader")
    print("="*80)
    
    try:
        dataset = IEMOCAPDataset(
            encodings_path='../../iemocap-encodings/train_encodings.npz',
            data_path='../../iemocap/train.txt',
            train=True
        )
        print(f"\n✓ Dataset loaded successfully!")
        print(f"Number of conversations: {len(dataset)}")
        
        # Test getting one sample
        features, qmask, labels, umask, conv_id = dataset[0]
        print(f"\nSample conversation: {conv_id}")
        print(f"  Features shape: {features.shape}")
        print(f"  Qmask shape: {qmask.shape}")
        print(f"  Labels shape: {labels.shape}")
        print(f"  Umask shape: {umask.shape}")
        
        return True
    except Exception as e:
        print(f"\n✗ DataLoader test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_forward_pass():
    """Test a forward pass through the model."""
    print("\n" + "="*80)
    print("Testing Forward Pass")
    print("="*80)
    
    try:
        # Create small dummy data
        seq_len = 5
        batch_size = 2
        D_m = 100
        n_speakers = 2
        
        # Dummy inputs
        U = torch.randn(seq_len, batch_size, D_m)
        qmask = torch.zeros(seq_len, batch_size, n_speakers)
        # Set random speakers
        for t in range(seq_len):
            for b in range(batch_size):
                speaker = np.random.randint(0, n_speakers)
                qmask[t, b, speaker] = 1.0
        
        umask = torch.ones(batch_size, seq_len)
        seq_lengths = [seq_len, seq_len]
        
        # Create model
        model = CEGEModel(
            D_m=D_m,
            D_e=100,
            D_h=100,
            speaker_state_dim=150,
            conv_state_dim=150,
            memory_dim=200,
            gcn_hidden_dim=200,
            n_speakers=n_speakers,
            n_classes=6,
            n_relations=8,
            dropout=0.5
        )
        
        model.eval()
        
        print("\nRunning forward pass...")
        with torch.no_grad():
            log_prob, edge_indices, edge_weights = model(U, qmask, umask, seq_lengths)
        
        print(f"\n✓ Forward pass successful!")
        print(f"Output shape: {log_prob.shape}")
        print(f"Expected shape: ({sum(seq_lengths)}, 6)")
        print(f"Number of graphs: {len(edge_indices)}")
        
        return True
    except Exception as e:
        print(f"\n✗ Forward pass failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    print("\n" + "="*80)
    print("CEGE MODEL TESTING")
    print("="*80)
    
    results = []
    
    # Test 1: Model initialization
    results.append(("Model Initialization", test_model_initialization()))
    
    # Test 2: DataLoader
    results.append(("DataLoader", test_dataloader()))
    
    # Test 3: Forward pass
    results.append(("Forward Pass", test_forward_pass()))
    
    # Summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    
    for test_name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{test_name:.<50} {status}")
    
    all_passed = all(result[1] for result in results)
    
    if all_passed:
        print("\n" + "="*80)
        print("ALL TESTS PASSED! ✓")
        print("="*80)
        print("\nYou can now run the full training:")
        print("python train_IEMOCAP.py --train-encodings ../../iemocap-encodings/train_encodings.npz --test-encodings ../../iemocap-encodings/test_encodings.npz --train-data ../../iemocap/train.txt --test-data ../../iemocap/test.txt")
    else:
        print("\n" + "="*80)
        print("SOME TESTS FAILED ✗")
        print("="*80)
        print("\nPlease check the errors above and fix them before training.")


if __name__ == '__main__':
    main()
