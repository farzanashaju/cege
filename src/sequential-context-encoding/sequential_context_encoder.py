import os
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from collections import defaultdict
from tqdm import tqdm

class ConversationDataset(Dataset):
    """
    Dataset that groups utterances into conversations and loads their pre-computed features.
    Each sample is a full conversation with all its utterances in order.
    """
    def __init__(self, encodings_path, raw_data_path):
        """
        Args:
            encodings_path: Path to .npz file with utterance encodings (from TextCNN)
            raw_data_path: Path to raw IEMOCAP data file (to get conversation structure)
        """
        print(f"Loading encodings from {encodings_path}...")
        # Load pre-computed utterance features
        data = np.load(encodings_path)
        self.utt_ids = data['utt_ids']
        self.features = data['feats']  # Shape: (N, 100)
        self.labels = data['labels']
        
        print(f"Loaded {len(self.utt_ids)} utterances with {self.features.shape[1]}-dim features")
        
        # Build mapping from utterance ID to its index
        self.utt_id_to_idx = {utt_id: idx for idx, utt_id in enumerate(self.utt_ids)}
        
        # Parse raw data to get conversation structure and speaker info
        print(f"Parsing conversation structure from {raw_data_path}...")
        self.conversations, self.conv_to_speakers = self._parse_conversations(raw_data_path)
        
        print(f"Found {len(self.conversations)} conversations")
        
        # Filter conversations to only include utterances we have features for
        self._filter_conversations()
        
        print(f"After filtering: {len(self.conversations)} conversations with features")
        
    def _parse_conversations(self, raw_data_path):
        """
        Parse the raw IEMOCAP data to group utterances into conversations.
        Returns:
            conversations: dict mapping conv_id to list of (utt_id, label, speaker)
            conv_to_speakers: dict mapping conv_id to set of speakers
        """
        conversations = defaultdict(list)
        conv_to_speakers = defaultdict(set)
        
        with open(raw_data_path, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) < 3:
                    continue
                    
                utt_id, label, text = parts[0], parts[1], parts[2]
                
                # Extract conversation ID and speaker from utterance ID
                # Format: Ses01F_impro01_F000 -> conv_id: Ses01F_impro01, speaker: F
                parts_id = utt_id.rsplit('_', 1)
                conv_id = parts_id[0]
                speaker = parts_id[1][0]  # First char is speaker (F/M)
                
                conversations[conv_id].append((utt_id, label, speaker))
                conv_to_speakers[conv_id].add(speaker)
        
        return dict(conversations), dict(conv_to_speakers)
    
    def _filter_conversations(self):
        """
        Filter conversations to only include utterances we have features for.
        """
        filtered_conversations = {}
        for conv_id, utterances in self.conversations.items():
            filtered_utts = [
                (utt_id, label, speaker) 
                for utt_id, label, speaker in utterances 
                if utt_id in self.utt_id_to_idx
            ]
            if filtered_utts:
                filtered_conversations[conv_id] = filtered_utts
        
        self.conversations = filtered_conversations
        self.conv_ids = sorted(self.conversations.keys())
    
    def __len__(self):
        return len(self.conv_ids)
    
    def __getitem__(self, idx):
        """
        Returns a single conversation with all its utterances.
        
        Returns:
            conv_id: conversation identifier
            utt_ids: list of utterance IDs in temporal order
            features: tensor of shape (seq_len, feature_dim)
            labels: tensor of shape (seq_len,)
            speakers: list of speaker identifiers
        """
        conv_id = self.conv_ids[idx]
        utterances = self.conversations[conv_id]
        
        utt_ids = []
        features = []
        labels = []
        speakers = []
        
        for utt_id, label, speaker in utterances:
            idx = self.utt_id_to_idx[utt_id]
            utt_ids.append(utt_id)
            features.append(self.features[idx])
            labels.append(self.labels[idx])
            speakers.append(speaker)
        
        features = torch.tensor(np.array(features), dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.long)
        
        return conv_id, utt_ids, features, labels, speakers


def collate_fn(batch):
    """
    Collate function for DataLoader.
    Since conversations have variable lengths, we return lists.
    Processing will be done one conversation at a time.
    """
    return batch


class SequentialContextEncoder(nn.Module):
    """
    Bidirectional GRU that processes utterance sequences to produce
    context-aware representations.
    
    Following the methodology:
    g_i = BiGRU_S(g_{i±1}, u_i) for i = 1,2,...,N
    
    where u_i is the context-independent utterance representation (from TextCNN)
    and g_i is the sequential context-aware representation.
    """
    def __init__(self, input_dim=100, hidden_dim=100, num_layers=1, dropout=0.1):
        super(SequentialContextEncoder, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Bidirectional GRU
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Project bidirectional output back to desired dimension
        # BiGRU outputs 2*hidden_dim, we project to hidden_dim
        self.projection = nn.Linear(2 * hidden_dim, hidden_dim)
        
        print(f"Initialized SequentialContextEncoder:")
        print(f"  Input dim: {input_dim}")
        print(f"  Hidden dim: {hidden_dim}")
        print(f"  Num layers: {num_layers}")
        print(f"  Bidirectional: True")
        print(f"  Output dim: {hidden_dim}")
    
    def forward(self, utterance_features):
        """
        Process a single conversation through the BiGRU.
        
        Args:
            utterance_features: tensor of shape (seq_len, input_dim)
                               context-independent features from TextCNN
        
        Returns:
            context_features: tensor of shape (seq_len, hidden_dim)
                            context-aware representations
        """
        # Add batch dimension: (seq_len, input_dim) -> (1, seq_len, input_dim)
        x = utterance_features.unsqueeze(0)
        
        # Pass through BiGRU
        # Output shape: (1, seq_len, 2*hidden_dim)
        gru_out, _ = self.gru(x)
        
        # Project to desired dimension
        # Shape: (1, seq_len, hidden_dim)
        context_features = self.projection(gru_out)
        
        # Remove batch dimension: (1, seq_len, hidden_dim) -> (seq_len, hidden_dim)
        context_features = context_features.squeeze(0)
        
        return context_features


def encode_dataset(model, dataset, device, output_path):
    """
    Encode all conversations in the dataset using the Sequential Context Encoder.
    
    Args:
        model: trained SequentialContextEncoder
        dataset: ConversationDataset
        device: torch device
        output_path: path to save encoded features
    """
    model.eval()
    
    all_utt_ids = []
    all_context_features = []
    all_labels = []
    all_speakers = []
    all_conv_ids = []
    
    print(f"\nEncoding {len(dataset)} conversations...")
    
    with torch.no_grad():
        for conv_id, utt_ids, features, labels, speakers in tqdm(dataset, desc="Processing conversations"):
            # Move features to device
            features = features.to(device)
            
            # Encode conversation
            context_features = model(features)
            
            # Store results
            all_conv_ids.extend([conv_id] * len(utt_ids))
            all_utt_ids.extend(utt_ids)
            all_context_features.append(context_features.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_speakers.extend(speakers)
    
    # Concatenate all features
    all_context_features = np.vstack(all_context_features)
    all_labels = np.array(all_labels)
    
    print(f"\nEncoded {len(all_utt_ids)} utterances")
    print(f"Context features shape: {all_context_features.shape}")
    
    # Save to file
    print(f"Saving to {output_path}...")
    np.savez(
        output_path,
        conv_ids=all_conv_ids,
        utt_ids=all_utt_ids,
        context_features=all_context_features,
        labels=all_labels,
        speakers=all_speakers
    )
    print("Saved successfully!")


def main(args):
    print("="*80)
    print("SEQUENTIAL CONTEXT ENCODING")
    print("="*80)
    print(f"\nConfiguration:")
    print(f"  Train encodings: {args.train_encodings}")
    print(f"  Test encodings: {args.test_encodings}")
    print(f"  Train data: {args.train_data}")
    print(f"  Test data: {args.test_data}")
    print(f"  Output directory: {args.output_dir}")
    print(f"  Hidden dim: {args.hidden_dim}")
    print(f"  Num layers: {args.num_layers}")
    print(f"  Device: {args.device}")
    print("="*80)
    
    # Set device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    
    # Load datasets
    print("\n" + "="*80)
    print("LOADING TRAIN DATASET")
    print("="*80)
    train_dataset = ConversationDataset(args.train_encodings, args.train_data)
    
    print("\n" + "="*80)
    print("LOADING TEST DATASET")
    print("="*80)
    test_dataset = ConversationDataset(args.test_encodings, args.test_data)
    
    # Get feature dimension from dataset
    feature_dim = train_dataset.features.shape[1]
    print(f"\nFeature dimension: {feature_dim}")
    
    # Initialize model
    print("\n" + "="*80)
    print("INITIALIZING MODEL")
    print("="*80)
    model = SequentialContextEncoder(
        input_dim=feature_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout
    )
    model.to(device)
    
    # Note: In this implementation, we're not training the BiGRU separately.
    # It will be trained end-to-end with the full CEGE model later.
    # For now, we're just setting up the architecture and encoding with random initialization.
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Encode train set
    print("\n" + "="*80)
    print("ENCODING TRAIN SET")
    print("="*80)
    train_output = os.path.join(args.output_dir, "train_context_encodings.npz")
    encode_dataset(model, train_dataset, device, train_output)
    
    # Encode test set
    print("\n" + "="*80)
    print("ENCODING TEST SET")
    print("="*80)
    test_output = os.path.join(args.output_dir, "test_context_encodings.npz")
    encode_dataset(model, test_dataset, device, test_output)
    
    # Save model architecture (will be trained later in full pipeline)
    model_path = os.path.join(args.output_dir, "sequential_context_encoder.pth")
    print(f"\nSaving model to {model_path}...")
    torch.save({
        'model_state_dict': model.state_dict(),
        'hidden_dim': args.hidden_dim,
        'num_layers': args.num_layers,
        'input_dim': feature_dim,
    }, model_path)
    
    print("\n" + "="*80)
    print("SEQUENTIAL CONTEXT ENCODING COMPLETE!")
    print("="*80)
    print(f"\nOutputs saved to: {args.output_dir}")
    print(f"  - {train_output}")
    print(f"  - {test_output}")
    print(f"  - {model_path}")
    print("\nNext step: Graph Representation and Temporal Memory")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Sequential Context Encoding using Bidirectional GRU"
    )
    
    # Input paths
    parser.add_argument(
        '--train_encodings',
        type=str,
        required=True,
        help='Path to train utterance encodings (.npz file from TextCNN)'
    )
    parser.add_argument(
        '--test_encodings',
        type=str,
        required=True,
        help='Path to test utterance encodings (.npz file from TextCNN)'
    )
    parser.add_argument(
        '--train_data',
        type=str,
        required=True,
        help='Path to raw train data file (to get conversation structure)'
    )
    parser.add_argument(
        '--test_data',
        type=str,
        required=True,
        help='Path to raw test data file (to get conversation structure)'
    )
    
    # Output path
    parser.add_argument(
        '--output_dir',
        type=str,
        default='./iemocap-context-encodings',
        help='Directory to save context-encoded features'
    )
    
    # Model hyperparameters
    parser.add_argument(
        '--hidden_dim',
        type=int,
        default=100,
        help='Hidden dimension for BiGRU'
    )
    parser.add_argument(
        '--num_layers',
        type=int,
        default=1,
        help='Number of BiGRU layers'
    )
    parser.add_argument(
        '--dropout',
        type=float,
        default=0.1,
        help='Dropout rate (only used if num_layers > 1)'
    )
    
    # Device
    parser.add_argument(
        '--device',
        type=str,
        default='cuda',
        help='Device to use (cuda or cpu)'
    )
    
    args = parser.parse_args()
    main(args)
