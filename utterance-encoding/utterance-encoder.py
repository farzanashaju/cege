import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm

class IEMOCAPDataset(Dataset):
    def __init__(self, file_path, vocab, max_len=50):
        # samples: (utt_id, label, text)
        self.samples = []
        # vocab: dict mapping tokens to indices
        self.vocab = vocab
        self.max_len = max_len

        # iterate over each line
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                # split on tab, remove newline
                parts = line.strip().split('\t')
                utt_id, label, text = parts[0], parts[1], parts[2]
                self.samples.append((utt_id, label, text))

        # build label to index mapping
        self.label2idx = {lab: i for i, lab in enumerate(sorted(set([s[1] for s in self.samples])))}
        # invert mapping to get index to label mapping
        self.idx2label = {i: lab for lab, i in self.label2idx.items()}

    def __len__(self):
        return len(self.samples)

    def encode_text(self, text):
        # whitespace tokenizer
        tokens = text.lower().split()
        # map each token to vocab id
        ids = [self.vocab.get(tok, self.vocab['<unk>']) for tok in tokens]
        # truncate to max_len
        if len(ids) > self.max_len:
            ids = ids[:self.max_len]
        # return tensor of token ids
        return torch.tensor(ids, dtype=torch.long)
    
    # returns a triple: utt_id (string), x (tensor of token ids), y (label index)
    def __getitem__(self, idx):
        utt_id, label, text = self.samples[idx]
        x = self.encode_text(text)
        y = self.label2idx[label]
        return utt_id, x, y

# pad batches
def collate_fn(batch):
    utt_ids, xs, ys = zip(*batch)
    # pad variable length sequences to the length of the longest in the batch
    xs_pad = pad_sequence(xs, batch_first=True, padding_value=0)
    ys = torch.tensor(ys, dtype=torch.long)
    return utt_ids, xs_pad, ys


class TextCNN(nn.Module):
    def __init__(self, vocab_size, embed_dim, embed_matrix, num_classes, num_filters=50, filter_sizes=[3,4,5], hidden_dim=100):
        super(TextCNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        if embed_matrix is not None:
            self.embedding.weight.data.copy_(torch.from_numpy(embed_matrix))
            self.embedding.weight.requires_grad = False

        # cnn layers with filter sizes [3,4,5]
        self.convs = nn.ModuleList([
            nn.Conv1d(in_channels=embed_dim,
                      out_channels=num_filters,
                      kernel_size=fs)
            for fs in filter_sizes
        ])

        # relu activation
        self.relu = nn.ReLU()
        # dense layer to map to 100-dim
        self.fc = nn.Linear(num_filters * len(filter_sizes), hidden_dim)

        self.out = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        emb = self.embedding(x)
        emb = emb.permute(0, 2, 1)
        convs = [self.relu(conv(emb)) for conv in self.convs]
        # global max pooling
        pooled = [torch.max(c, dim=2)[0] for c in convs]
        out = torch.cat(pooled, dim=1)
        feat = self.fc(out)
        logits = self.out(feat)
        return logits, feat


def load_glove(path, vocab, embed_dim=300):
    embeddings = {}
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.rstrip().split(" ")
            word, vec = parts[0], parts[1:]
            if len(vec) != embed_dim:  # skip malformed
                continue
            if word in vocab:
                embeddings[word] = np.fromstring(" ".join(vec), sep=" ", dtype=np.float32)
    return embeddings


def train_model(model, train_loader, test_loader, device, epochs, lr):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    model.to(device)
    for epoch in range(1, epochs+1):
        model.train()
        total_loss = 0
        for utt_ids, x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits, _ = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(train_loader)

        # validation on test set
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for utt_ids, x, y in test_loader:
                x, y = x.to(device), y.to(device)
                logits, _ = model(x)
                preds = torch.argmax(logits, dim=1)
                correct += (preds == y).sum().item()
                total += y.size(0)
        acc = correct / total
        print(f"Epoch {epoch}: Loss {avg_loss:.4f}, Test Acc {acc:.4f}")

# extract and save features
def extract_features(model, loader, device, out_path):
    model.eval()
    utt_ids_all, feats_all, labels_all = [], [], []
    with torch.no_grad():
        for utt_ids, x, y in loader:
            x = x.to(device)
            _, feats = model(x)
            utt_ids_all.extend(utt_ids)
            feats_all.append(feats.cpu().numpy())
            labels_all.extend(y.numpy())
    feats_all = np.vstack(feats_all)
    np.savez(out_path, utt_ids=utt_ids_all, feats=feats_all, labels=np.array(labels_all))


def main(args):
    # build vocab from train and test splits
    vocab = {'<pad>':0, '<unk>':1}
    idx = 2
    for split in [args.train_file, args.test_file]:
        with open(split, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) < 3:
                    continue
                text = parts[2]
                for tok in text.lower().split():
                    if tok not in vocab:
                        vocab[tok] = idx
                        idx += 1


    glove_vectors = load_glove(args.glove, vocab)
    embed_dim = 300
    embed_matrix = np.random.uniform(-0.05, 0.05, (len(vocab), embed_dim)).astype(np.float32)
    for word, i in vocab.items():
        if word in glove_vectors:
            embed_matrix[i] = glove_vectors[word]

    train_set = IEMOCAPDataset(args.train_file, vocab, args.max_len)
    test_set = IEMOCAPDataset(args.test_file, vocab, args.max_len)

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    model = TextCNN(vocab_size=len(vocab),
                    embed_dim=embed_dim,
                    embed_matrix=embed_matrix,
                    num_classes=len(train_set.label2idx))

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    train_model(model, train_loader, test_loader, device, args.epochs, args.lr)

    os.makedirs(args.out_dir, exist_ok=True)
    extract_features(model, train_loader, device, os.path.join(args.out_dir, "train_encodings.npz"))
    extract_features(model, test_loader, device, os.path.join(args.out_dir, "test_encodings.npz"))

    torch.save({
        'model_state': model.state_dict(),
        'vocab': vocab,
        'label2idx': train_set.label2idx
    }, os.path.join(args.out_dir, "textcnn_encoder.pth"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_file', type=str, required=True)
    parser.add_argument('--test_file', type=str, required=True)
    parser.add_argument('--glove', type=str, required=True)
    parser.add_argument('--out_dir', type=str, default='/ssd_scratch/farzana/iemocap_textcnn_features')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--epochs', type=int, default=12)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--max_len', type=int, default=50)
    args = parser.parse_args()
    main(args)