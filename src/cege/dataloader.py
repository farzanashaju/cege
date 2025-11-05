"""
Data loader for CEGE model on IEMOCAP dataset.
"""

import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import numpy as np
from collections import defaultdict


class IEMOCAPDataset(Dataset):
    """
    IEMOCAP Dataset for CEGE model.
    
    Loads:
    - Pre-computed TextCNN utterance encodings
    - Emotion labels
    - Speaker information
    - Conversation structure
    """
    def __init__(self, encodings_path, data_path, train=True):
        """
        Args:
            encodings_path: Path to .npz file with TextCNN encodings
            data_path: Path to IEMOCAP text file (train.txt or test.txt)
            train: Whether this is training data
        """
        self.train = train
        
        # Load encodings
        print(f"Loading encodings from {encodings_path}...")
        data = np.load(encodings_path, allow_pickle=True)
        self.utt_ids = data['utt_ids']
        self.features = data['feats']  # (N, 100)
        self.labels_array = data['labels']
        
        # Create mapping
        self.utt_to_idx = {uid: idx for idx, uid in enumerate(self.utt_ids)}
        
        # Label mapping
        self.label_to_idx = {
            'hap': 0, 'sad': 1, 'neu': 2, 'ang': 3, 'exc': 4, 'fru': 5,
            'happy': 0, 'neutral': 2, 'angry': 3, 'excited': 4, 'frustrated': 5,
            'disgust': 6, 'fear': 7
        }
        # Map 'hap' and 'exc' to same class (happy emotions)
        self.label_to_idx['exc'] = 0  # Map excited to happy
        
        # Parse conversation structure
        print(f"Parsing conversations from {data_path}...")
        self.conversations = self._parse_conversations(data_path)
        
        # Get conversation IDs
        self.conv_ids = sorted(self.conversations.keys())
        
        print(f"Loaded {len(self.utt_ids)} utterances from {len(self.conv_ids)} conversations")
        
    def _parse_conversations(self, data_path):
        """
        Parse IEMOCAP data file into conversations.
        
        Returns:
            dict: {conv_id: [(utt_id, label, speaker), ...]}
        """
        conversations = defaultdict(list)
        
        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) < 3:
                    continue
                
                utt_id, label, text = parts[0], parts[1], parts[2]
                
                # Extract conversation ID and speaker
                # Format: Ses01F_impro01_F000
                parts_id = utt_id.rsplit('_', 1)
                conv_id = parts_id[0]  # Ses01F_impro01
                speaker = parts_id[1][0]  # F or M
                
                # Only include utterances we have features for
                if utt_id in self.utt_to_idx:
                    conversations[conv_id].append((utt_id, label, speaker))
        
        return dict(conversations)
    
    def __len__(self):
        return len(self.conv_ids)
    
    def __getitem__(self, idx):
        """
        Returns a conversation with all its utterances.
        
        Returns:
            features: (seq_len, 100) - TextCNN features
            qmask: (seq_len, n_speakers) - speaker one-hot encoding
            labels: (seq_len,) - emotion labels
            umask: (seq_len,) - utterance mask (all 1s for real data)
            conv_id: str - conversation ID
        """
        conv_id = self.conv_ids[idx]
        utterances = self.conversations[conv_id]
        
        seq_len = len(utterances)
        features_list = []
        labels_list = []
        speakers_list = []
        
        for utt_id, label, speaker in utterances:
            # Get features
            feat_idx = self.utt_to_idx[utt_id]
            features_list.append(self.features[feat_idx])
            
            # Get label
            label_idx = self.label_to_idx.get(label.lower(), 2)  # Default to neutral
            labels_list.append(label_idx)
            
            # Get speaker (0 for F, 1 for M)
            speaker_id = 0 if speaker == 'F' else 1
            speakers_list.append(speaker_id)
        
        # Convert to tensors
        features = torch.FloatTensor(np.array(features_list))  # (seq_len, 100)
        labels = torch.LongTensor(labels_list)  # (seq_len,)
        
        # Create speaker one-hot encoding (qmask)
        qmask = torch.zeros(seq_len, 2)  # 2 speakers: F, M
        for i, speaker_id in enumerate(speakers_list):
            qmask[i, speaker_id] = 1.0
        
        # Utterance mask (all 1s, no padding yet)
        umask = torch.ones(seq_len)
        
        return features, qmask, labels, umask, conv_id
    
    @staticmethod
    def collate_fn(batch):
        """
        Collate function for batching multiple conversations.
        Pads to the longest sequence in the batch.
        
        Args:
            batch: list of (features, qmask, labels, umask, conv_id)
        
        Returns:
            features_pad: (max_seq_len, batch, 100)
            qmask_pad: (max_seq_len, batch, n_speakers)
            labels_pad: (max_seq_len, batch)
            umask_pad: (batch, max_seq_len)
            seq_lengths: list of actual lengths
            conv_ids: list of conversation IDs
        """
        features_list, qmask_list, labels_list, umask_list, conv_ids = zip(*batch)
        
        # Get sequence lengths
        seq_lengths = [f.size(0) for f in features_list]
        
        # Pad sequences
        features_pad = pad_sequence(features_list, batch_first=False, padding_value=0.0)  # (max_len, batch, 100)
        qmask_pad = pad_sequence(qmask_list, batch_first=False, padding_value=0.0)  # (max_len, batch, 2)
        labels_pad = pad_sequence(labels_list, batch_first=False, padding_value=-1)  # (max_len, batch)
        umask_pad = pad_sequence(umask_list, batch_first=True, padding_value=0.0)  # (batch, max_len)
        
        return features_pad, qmask_pad, labels_pad, umask_pad, seq_lengths, list(conv_ids)


def get_IEMOCAP_loaders(train_encodings, test_encodings, train_data, test_data, 
                         batch_size=32, num_workers=0, valid_split=0.1):
    """
    Create data loaders for IEMOCAP dataset.
    
    Args:
        train_encodings: Path to train encodings .npz
        test_encodings: Path to test encodings .npz
        train_data: Path to train.txt
        test_data: Path to test.txt
        batch_size: Batch size
        num_workers: Number of worker processes
        valid_split: Fraction of training data to use for validation
    
    Returns:
        train_loader, valid_loader, test_loader
    """
    from torch.utils.data import DataLoader, SubsetRandomSampler
    
    # Create datasets
    train_dataset = IEMOCAPDataset(train_encodings, train_data, train=True)
    test_dataset = IEMOCAPDataset(test_encodings, test_data, train=False)
    
    # Split training into train/valid
    train_size = len(train_dataset)
    indices = list(range(train_size))
    split = int(np.floor(valid_split * train_size))
    
    # Shuffle indices
    np.random.seed(42)
    np.random.shuffle(indices)
    
    train_indices, valid_indices = indices[split:], indices[:split]
    
    # Create samplers
    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(valid_indices)
    
    # Create loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        collate_fn=IEMOCAPDataset.collate_fn,
        num_workers=num_workers,
        pin_memory=True
    )
    
    valid_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=valid_sampler,
        collate_fn=IEMOCAPDataset.collate_fn,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=IEMOCAPDataset.collate_fn,
        num_workers=num_workers,
        pin_memory=True
    )
    
    print(f"\nDataLoaders created:")
    print(f"  Train: {len(train_indices)} conversations")
    print(f"  Valid: {len(valid_indices)} conversations")
    print(f"  Test: {len(test_dataset)} conversations")
    
    return train_loader, valid_loader, test_loader
