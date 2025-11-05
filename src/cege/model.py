"""
CEGE: Conversational Emotion Graph Evolution
Complete implementation of the CEGE model as described in the methodology.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import RGCNConv, GraphConv
import math
import numpy as np


class MatchingAttention(nn.Module):
    """
    Matching Attention mechanism from DialogueGCN.
    Used in sequential context encoding.
    """
    def __init__(self, mem_dim, cand_dim, alpha_dim=None, att_type='general2'):
        super(MatchingAttention, self).__init__()
        assert att_type != 'concat' or alpha_dim != None
        assert att_type != 'dot' or mem_dim == cand_dim
        
        self.mem_dim = mem_dim
        self.cand_dim = cand_dim
        self.att_type = att_type
        
        if att_type == 'general':
            self.transform = nn.Linear(cand_dim, mem_dim, bias=False)
        if att_type == 'general2':
            self.transform = nn.Linear(cand_dim, mem_dim, bias=True)
        elif att_type == 'concat':
            self.transform = nn.Linear(cand_dim + mem_dim, alpha_dim, bias=False)
            self.vector_prod = nn.Linear(alpha_dim, 1, bias=False)

    def forward(self, M, x, mask=None):
        """
        M -> (seq_len, batch, mem_dim)
        x -> (batch, cand_dim)
        mask -> (batch, seq_len)
        """
        if mask is None:
            mask = torch.ones(M.size(1), M.size(0)).type(M.type())

        if self.att_type == 'dot':
            M_ = M.permute(1, 2, 0)  # batch, vector, seqlen
            x_ = x.unsqueeze(1)  # batch, 1, vector
            alpha = F.softmax(torch.bmm(x_, M_), dim=2)  # batch, 1, seqlen
        elif self.att_type == 'general':
            M_ = M.permute(1, 2, 0)  # batch, mem_dim, seqlen
            x_ = self.transform(x).unsqueeze(1)  # batch, 1, mem_dim
            alpha = F.softmax(torch.bmm(x_, M_), dim=2)  # batch, 1, seqlen
        elif self.att_type == 'general2':
            M_ = M.permute(1, 2, 0)  # batch, mem_dim, seqlen
            x_ = self.transform(x).unsqueeze(1)  # batch, 1, mem_dim
            mask_ = mask.unsqueeze(2).repeat(1, 1, self.mem_dim).transpose(1, 2)  # batch, mem_dim, seq_len
            M_ = M_ * mask_
            alpha_ = torch.bmm(x_, M_) * mask.unsqueeze(1)
            alpha_ = torch.tanh(alpha_)
            alpha_ = F.softmax(alpha_, dim=2)
            alpha_masked = alpha_ * mask.unsqueeze(1)  # batch, 1, seqlen
            alpha_sum = torch.sum(alpha_masked, dim=2, keepdim=True)  # batch, 1, 1
            alpha = alpha_masked / alpha_sum  # batch, 1, 1 ; normalized
        else:
            M_ = M.transpose(0, 1)  # batch, seqlen, mem_dim
            x_ = x.unsqueeze(1).expand(-1, M.size()[0], -1)  # batch, seqlen, cand_dim
            M_x_ = torch.cat([M_, x_], 2)  # batch, seqlen, mem_dim+cand_dim
            mx_a = F.tanh(self.transform(M_x_))  # batch, seqlen, alpha_dim
            alpha = F.softmax(self.vector_prod(mx_a), 1).transpose(1, 2)  # batch, 1, seqlen

        attn_pool = torch.bmm(alpha, M.transpose(0, 1))[:, 0, :]  # batch, mem_dim
        return attn_pool, alpha


class SequentialContextEncoder(nn.Module):
    """
    Bidirectional 2-layer GRU with MatchingAttention.
    Matches DialogueGCN implementation.
    
    Input: u_i (context-independent utterance features) - 100-dim
    Output: g_i (context-aware utterance features) - 200-dim (2 * hidden_dim)
    """
    def __init__(self, D_m=100, D_e=100, D_h=100, dropout=0.5, n_classes=6):
        super(SequentialContextEncoder, self).__init__()
        
        self.D_m = D_m  # Input dimension (TextCNN output)
        self.D_e = D_e  # Hidden dimension per direction
        self.D_h = D_h  # Hidden dimension for classification
        self.n_classes = n_classes
        
        # 2-layer Bidirectional GRU (matching DialogueGCN)
        self.gru = nn.GRU(
            input_size=D_m,
            hidden_size=D_e,
            num_layers=2,
            bidirectional=True,
            dropout=dropout,
            batch_first=False  # seq_len first
        )
        
        # Matching Attention (matching DialogueGCN)
        self.matchatt = MatchingAttention(2 * D_e, 2 * D_e, att_type='general2')
        
        # Classification layers (for supervised training)
        self.linear = nn.Linear(2 * D_e, D_h)
        self.dropout = nn.Dropout(dropout)
        self.smax_fc = nn.Linear(D_h, n_classes)
        
        print(f"SequentialContextEncoder initialized:")
        print(f"  Input dim (D_m): {D_m}")
        print(f"  Hidden dim (D_e): {D_e}")
        print(f"  Output dim: {2 * D_e} (bidirectional)")
        print(f"  Num layers: 2")
        print(f"  Attention: MatchingAttention (general2)")
    
    def forward(self, U, umask=None, return_logits=False):
        """
        U: (seq_len, batch, D_m) - utterance features
        umask: (batch, seq_len) - mask for padding
        return_logits: whether to return classification logits
        
        Returns:
            emotions: (seq_len, batch, 2*D_e) - context-aware features
            log_prob: (seq_len, batch, n_classes) - classification logits (if return_logits=True)
        """
        # Pass through BiGRU
        emotions, hidden = self.gru(U)  # emotions: (seq_len, batch, 2*D_e)
        
        if return_logits:
            # Apply attention for classification
            att_emotions = []
            for t in emotions:
                att_em, alpha_ = self.matchatt(emotions, t, mask=umask)
                att_emotions.append(att_em.unsqueeze(0))
            att_emotions = torch.cat(att_emotions, dim=0)
            
            # Classification
            hidden = F.relu(self.linear(att_emotions))
            hidden = self.dropout(hidden)
            log_prob = F.log_softmax(self.smax_fc(hidden), 2)
            
            return emotions, log_prob
        else:
            return emotions


class TemporalMemoryModule(nn.Module):
    """
    LSTM-based temporal memory for each node.
    
    m_i^t = LSTM(m_i^{t-1}, g_i, context^t)
    
    where context includes:
    - Q_s^t: speaker state
    - C^t: global conversation state
    """
    def __init__(self, g_dim=200, speaker_state_dim=150, conv_state_dim=150, memory_dim=200):
        super(TemporalMemoryModule, self).__init__()
        
        self.g_dim = g_dim
        self.speaker_state_dim = speaker_state_dim
        self.conv_state_dim = conv_state_dim
        self.memory_dim = memory_dim
        
        # Input: [g_i, Q_s, C] concatenated
        input_dim = g_dim + speaker_state_dim + conv_state_dim
        
        # LSTM for temporal memory
        self.lstm = nn.LSTMCell(input_dim, memory_dim)
        
        print(f"TemporalMemoryModule initialized:")
        print(f"  Input: g_dim={g_dim}, speaker_state={speaker_state_dim}, conv_state={conv_state_dim}")
        print(f"  Memory dim: {memory_dim}")
    
    def forward(self, g_i, speaker_state, conv_state, prev_memory=None, prev_cell=None):
        """
        g_i: (batch, g_dim) - context-aware utterance embedding
        speaker_state: (batch, speaker_state_dim) - speaker state Q_s
        conv_state: (batch, conv_state_dim) - conversation state C
        prev_memory: (batch, memory_dim) - previous memory state
        prev_cell: (batch, memory_dim) - previous cell state
        
        Returns:
            memory: (batch, memory_dim) - updated memory state
            cell: (batch, memory_dim) - updated cell state
        """
        batch_size = g_i.size(0)
        
        # Initialize if needed
        if prev_memory is None:
            prev_memory = torch.zeros(batch_size, self.memory_dim).to(g_i.device)
        if prev_cell is None:
            prev_cell = torch.zeros(batch_size, self.memory_dim).to(g_i.device)
        
        # Concatenate inputs
        context = torch.cat([g_i, speaker_state, conv_state], dim=-1)
        
        # Update memory via LSTM
        memory, cell = self.lstm(context, (prev_memory, prev_cell))
        
        return memory, cell


class SpeakerStateTracker(nn.Module):
    """
    Tracks speaker states Q_s^t for each speaker.
    Updates based on their utterances.
    """
    def __init__(self, input_dim=200, state_dim=150):
        super(SpeakerStateTracker, self).__init__()
        
        self.state_dim = state_dim
        self.gru = nn.GRUCell(input_dim, state_dim)
        
    def forward(self, utterance_features, prev_state=None):
        """
        utterance_features: (batch, input_dim)
        prev_state: (batch, state_dim)
        
        Returns:
            state: (batch, state_dim) - updated speaker state
        """
        if prev_state is None:
            prev_state = torch.zeros(utterance_features.size(0), self.state_dim).to(utterance_features.device)
        
        state = self.gru(utterance_features, prev_state)
        return state


class ConversationStateTracker(nn.Module):
    """
    Tracks global conversation state C^t.
    Updates with each utterance.
    """
    def __init__(self, input_dim=200, state_dim=150):
        super(ConversationStateTracker, self).__init__()
        
        self.state_dim = state_dim
        self.gru = nn.GRUCell(input_dim, state_dim)
        
    def forward(self, utterance_features, prev_state=None):
        """
        utterance_features: (batch, input_dim)
        prev_state: (batch, state_dim)
        
        Returns:
            state: (batch, state_dim) - updated conversation state
        """
        if prev_state is None:
            prev_state = torch.zeros(utterance_features.size(0), self.state_dim).to(utterance_features.device)
        
        state = self.gru(utterance_features, prev_state)
        return state


class DynamicGraphBuilder(nn.Module):
    """
    Builds and updates the dynamic graph structure.
    
    Handles:
    - Edge weight computation: alpha_ij^t
    - Edge pruning (weights < tau_remove)
    - Edge addition (weights > tau_create)
    - Relation type assignment
    """
    def __init__(self, feature_dim=200, n_speakers=2, tau_remove=0.1, tau_create=0.7):
        super(DynamicGraphBuilder, self).__init__()
        
        self.feature_dim = feature_dim
        self.n_speakers = n_speakers
        self.tau_remove = tau_remove
        self.tau_create = tau_create
        
        # Edge weight predictor
        self.edge_weight_net = nn.Sequential(
            nn.Linear(feature_dim * 2 + 4, 128),  # [g_i, g_j, speaker_info, temporal_info]
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()  # Output in [0, 1]
        )
        
        # Temporal decay parameter
        self.temporal_decay = nn.Parameter(torch.tensor(0.1))
        
    def compute_edge_weights(self, node_features, speaker_ids, timesteps):
        """
        Compute edge weights for all pairs of nodes.
        
        node_features: (N, feature_dim)
        speaker_ids: (N,) - speaker ID for each node
        timesteps: (N,) - timestep for each node
        
        Returns:
            edge_weights: (N, N) - weight matrix
        """
        N = node_features.size(0)
        device = node_features.device
        
        # Create all pairs
        edge_weights = torch.zeros(N, N).to(device)
        
        for i in range(N):
            for j in range(N):
                if i == j:
                    continue
                
                # Concatenate features
                f_i = node_features[i]
                f_j = node_features[j]
                
                # Speaker info: same speaker or different
                same_speaker = float(speaker_ids[i] == speaker_ids[j])
                speaker_info = torch.tensor([same_speaker, 1 - same_speaker]).to(device)
                
                # Temporal info: time difference
                time_diff = abs(timesteps[i] - timesteps[j])
                temporal_decay = torch.exp(-self.temporal_decay * time_diff)
                temporal_info = torch.tensor([time_diff / N, temporal_decay]).to(device)
                
                # Predict edge weight
                edge_input = torch.cat([f_i, f_j, speaker_info, temporal_info])
                weight = self.edge_weight_net(edge_input)
                
                edge_weights[i, j] = weight
        
        return edge_weights
    
    def prune_and_build_edges(self, edge_weights):
        """
        Prune edges below threshold and create edge index.
        
        edge_weights: (N, N)
        
        Returns:
            edge_index: (2, E) - COO format edge list
            edge_weights_filtered: (E,) - weights for kept edges
        """
        # Apply thresholds
        mask = edge_weights >= self.tau_remove
        
        # Get edge indices
        edge_index = mask.nonzero(as_tuple=False).t()  # (2, E)
        edge_weights_filtered = edge_weights[mask]
        
        return edge_index, edge_weights_filtered


class TemporalGCNLayer(nn.Module):
    """
    Temporal Graph Convolutional Layer with relation-specific transformations.
    
    h_i = sigma( sum_r sum_{j in N_i^r} (alpha_ij / c_{i,r}) * W_r [g_j | m_j] + W_0 [g_i | m_i] )
    """
    def __init__(self, input_dim=400, hidden_dim=200, n_relations=8, dropout=0.5):
        super(TemporalGCNLayer, self).__init__()
        
        self.input_dim = input_dim  # g_dim + memory_dim
        self.hidden_dim = hidden_dim
        self.n_relations = n_relations
        
        # Relation-specific GCN
        self.rgcn = RGCNConv(input_dim, hidden_dim, n_relations, num_bases=30)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, node_features, edge_index, edge_type, edge_weights=None):
        """
        node_features: (N, input_dim) - concatenated [g_i | m_i]
        edge_index: (2, E) - edge list
        edge_type: (E,) - relation type for each edge
        edge_weights: (E,) - edge weights (optional, not used by RGCN directly)
        
        Returns:
            h: (N, hidden_dim) - updated node features
        """
        # Apply RGCN (RGCNConv doesn't accept edge_weights parameter)
        # Edge weights could be incorporated by scaling node features
        h = self.rgcn(node_features, edge_index, edge_type)
        
        h = F.relu(h)
        h = self.dropout(h)
        
        return h


class TemporalAttention(nn.Module):
    """
    Temporal attention over historical memory states.
    
    beta_i^t = softmax((h_i^{(2),t})^T W_beta [m_1^t, ..., m_{i-1}^t])
    with temporal decay: beta_ik^t *= exp(-lambda_decay * (t-k))
    """
    def __init__(self, hidden_dim=200, memory_dim=200, lambda_decay=0.1):
        super(TemporalAttention, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.memory_dim = memory_dim
        self.lambda_decay = lambda_decay
        
        # Attention weight matrix
        self.W_beta = nn.Linear(hidden_dim, memory_dim)
        
    def forward(self, h_i, memory_states, current_time):
        """
        h_i: (batch, hidden_dim) - current node's GCN output
        memory_states: list of (batch, memory_dim) - memory states of previous utterances
        current_time: int - current timestep
        
        Returns:
            context: (batch, memory_dim) - attention-weighted context vector
            attention_weights: (batch, num_prev) - attention weights
        """
        if len(memory_states) == 0:
            # No history yet
            return torch.zeros(h_i.size(0), self.memory_dim).to(h_i.device), None
        
        batch_size = h_i.size(0)
        
        # Stack memory states
        M = torch.stack(memory_states, dim=1)  # (batch, num_prev, memory_dim)
        num_prev = M.size(1)
        
        # Compute attention scores
        h_transformed = self.W_beta(h_i)  # (batch, memory_dim)
        scores = torch.bmm(h_transformed.unsqueeze(1), M.transpose(1, 2))  # (batch, 1, num_prev)
        scores = scores.squeeze(1)  # (batch, num_prev)
        
        # Apply temporal decay
        decay_weights = torch.exp(-self.lambda_decay * torch.arange(num_prev, 0, -1, dtype=torch.float32).to(h_i.device))
        scores = scores * decay_weights.unsqueeze(0)
        
        # Softmax
        attention_weights = F.softmax(scores, dim=1)  # (batch, num_prev)
        
        # Weighted sum
        context = torch.bmm(attention_weights.unsqueeze(1), M).squeeze(1)  # (batch, memory_dim)
        
        return context, attention_weights


class CEGEModel(nn.Module):
    """
    Complete CEGE: Conversational Emotion Graph Evolution model.
    
    Integrates:
    1. Sequential Context Encoder (BiGRU)
    2. Temporal Memory Modules (LSTM)
    3. Dynamic Graph Builder
    4. Temporal GCN Layers
    5. Temporal Attention
    6. Classification Head
    """
    def __init__(self, 
                 D_m=100,           # Input feature dimension (TextCNN)
                 D_e=100,           # Sequential encoder hidden dim
                 D_h=100,           # Classification hidden dim
                 speaker_state_dim=150,
                 conv_state_dim=150,
                 memory_dim=200,
                 gcn_hidden_dim=200,
                 n_speakers=2,
                 n_classes=6,
                 n_relations=8,
                 dropout=0.5,
                 tau_remove=0.1,
                 tau_create=0.7,
                 lambda_decay=0.1):
        super(CEGEModel, self).__init__()
        
        self.D_m = D_m
        self.D_e = D_e
        self.n_speakers = n_speakers
        self.n_classes = n_classes
        
        # 1. Sequential Context Encoder (BiGRU with MatchingAttention)
        print("\n=== Initializing Sequential Context Encoder ===")
        self.sequential_encoder = SequentialContextEncoder(
            D_m=D_m, D_e=D_e, D_h=D_h, dropout=dropout, n_classes=n_classes
        )
        
        # 2. Temporal Memory Module
        print("\n=== Initializing Temporal Memory Module ===")
        self.temporal_memory = TemporalMemoryModule(
            g_dim=2*D_e,  # BiGRU output is 2*D_e
            speaker_state_dim=speaker_state_dim,
            conv_state_dim=conv_state_dim,
            memory_dim=memory_dim
        )
        
        # 3. Speaker State Tracker
        self.speaker_tracker = SpeakerStateTracker(
            input_dim=2*D_e, state_dim=speaker_state_dim
        )
        
        # 4. Conversation State Tracker
        self.conv_tracker = ConversationStateTracker(
            input_dim=2*D_e, state_dim=conv_state_dim
        )
        
        # 5. Dynamic Graph Builder
        print("\n=== Initializing Dynamic Graph Builder ===")
        self.graph_builder = DynamicGraphBuilder(
            feature_dim=2*D_e,
            n_speakers=n_speakers,
            tau_remove=tau_remove,
            tau_create=tau_create
        )
        
        # 6. Temporal GCN Layers
        print("\n=== Initializing Temporal GCN Layers ===")
        gcn_input_dim = 2*D_e + memory_dim  # [g | m]
        self.gcn_layer1 = TemporalGCNLayer(
            input_dim=gcn_input_dim,
            hidden_dim=gcn_hidden_dim,
            n_relations=n_relations,
            dropout=dropout
        )
        self.gcn_layer2 = GraphConv(gcn_hidden_dim, gcn_hidden_dim)
        
        # 7. Temporal Attention
        print("\n=== Initializing Temporal Attention ===")
        self.temporal_attention = TemporalAttention(
            hidden_dim=gcn_hidden_dim,
            memory_dim=memory_dim,
            lambda_decay=lambda_decay
        )
        
        # 8. Classification Head
        print("\n=== Initializing Classification Head ===")
        self.matchatt_final = MatchingAttention(
            gcn_hidden_dim + memory_dim,
            gcn_hidden_dim + memory_dim,
            att_type='general2'
        )
        self.linear_final = nn.Linear(gcn_hidden_dim + memory_dim, D_h)
        self.dropout_final = nn.Dropout(dropout)
        self.classifier = nn.Linear(D_h, n_classes)
        
        print("\n=== CEGE Model Initialized ===")
        
    def forward(self, U, qmask, umask, seq_lengths):
        """
        Forward pass for a batch of conversations.
        
        U: (seq_len, batch, D_m) - utterance features (from TextCNN)
        qmask: (seq_len, batch, n_speakers) - speaker one-hot encoding
        umask: (batch, seq_len) - utterance mask (1 for real, 0 for padding)
        seq_lengths: list of int - actual length of each conversation in batch
        
        Returns:
            log_prob: (total_utterances, n_classes) - emotion predictions
            edge_index: edge list for visualization
            edge_weights: edge weights
        """
        seq_len, batch_size, _ = U.size()
        device = U.device
        
        # 1. Sequential Context Encoding
        g = self.sequential_encoder(U, umask)  # (seq_len, batch, 2*D_e)
        
        # Process each conversation in the batch
        all_log_probs = []
        all_edge_indices = []
        all_edge_weights = []
        
        for b in range(batch_size):
            conv_len = seq_lengths[b]
            g_conv = g[:conv_len, b, :]  # (conv_len, 2*D_e)
            qmask_conv = qmask[:conv_len, b, :]  # (conv_len, n_speakers)
            
            # Initialize states
            speaker_states = {s: None for s in range(self.n_speakers)}
            conv_state = None
            memory_states = []
            memory_cells = []
            node_features_list = []
            
            # Process utterances sequentially
            for t in range(conv_len):
                g_t = g_conv[t:t+1, :]  # (1, 2*D_e)
                speaker_id = torch.argmax(qmask_conv[t]).item()
                
                # Update speaker state
                speaker_states[speaker_id] = self.speaker_tracker(
                    g_t, speaker_states[speaker_id]
                )
                
                # Update conversation state
                conv_state = self.conv_tracker(g_t, conv_state)
                
                # Update temporal memory
                memory, cell = self.temporal_memory(
                    g_t,
                    speaker_states[speaker_id],
                    conv_state,
                    memory_states[-1] if len(memory_states) > 0 else None,
                    memory_cells[-1] if len(memory_cells) > 0 else None
                )
                
                memory_states.append(memory)
                memory_cells.append(cell)
                node_features_list.append(g_t)
            
            # Stack node features
            g_all = torch.cat(node_features_list, dim=0)  # (conv_len, 2*D_e)
            m_all = torch.stack(memory_states, dim=0)  # (conv_len, 1, memory_dim)
            m_all = m_all.squeeze(1)  # (conv_len, memory_dim)
            
            # Build dynamic graph
            speaker_ids = torch.argmax(qmask_conv, dim=1)  # (conv_len,)
            timesteps = torch.arange(conv_len, dtype=torch.float32).to(device)
            
            edge_weights = self.graph_builder.compute_edge_weights(
                g_all, speaker_ids, timesteps
            )
            edge_index, edge_weights_filtered = self.graph_builder.prune_and_build_edges(edge_weights)
            
            # Assign relation types (speaker-based + temporal)
            edge_types = torch.zeros(edge_index.size(1), dtype=torch.long).to(device)
            for e in range(edge_index.size(1)):
                src, tgt = edge_index[0, e].item(), edge_index[1, e].item()
                src_speaker = speaker_ids[src].item()
                tgt_speaker = speaker_ids[tgt].item()
                temporal_dir = 0 if src < tgt else 1  # past=0, future=1
                
                # Relation type: speaker_src * n_speakers + speaker_tgt * 2 + temporal_dir
                rel_type = src_speaker * self.n_speakers * 2 + tgt_speaker * 2 + temporal_dir
                edge_types[e] = rel_type
            
            # Apply Temporal GCN
            node_input = torch.cat([g_all, m_all], dim=-1)  # (conv_len, 2*D_e + memory_dim)
            h1 = self.gcn_layer1(node_input, edge_index, edge_types, edge_weights_filtered)
            h2 = self.gcn_layer2(h1, edge_index)
            h2 = F.relu(h2)
            
            # Apply Temporal Attention
            context_vectors = []
            for t in range(conv_len):
                context, _ = self.temporal_attention(
                    h2[t:t+1], memory_states[:t], t
                )
                context_vectors.append(context)
            
            context_all = torch.cat(context_vectors, dim=0)  # (conv_len, memory_dim)
            
            # Final features
            final_features = torch.cat([h2, context_all], dim=-1)  # (conv_len, gcn_hidden + memory_dim)
            
            # Classification (simpler approach without attention for now)
            hidden = F.relu(self.linear_final(final_features))
            hidden = self.dropout_final(hidden)
            log_prob = F.log_softmax(self.classifier(hidden), dim=1)  # (conv_len, n_classes)
            
            all_log_probs.append(log_prob)
            all_edge_indices.append(edge_index)
            all_edge_weights.append(edge_weights_filtered)
        
        # Concatenate all conversations
        log_probs_cat = torch.cat(all_log_probs, dim=0)
        
        return log_probs_cat, all_edge_indices, all_edge_weights


class MaskedNLLLoss(nn.Module):
    """
    Masked Negative Log-Likelihood Loss.
    Handles variable-length sequences with padding.
    """
    def __init__(self, weight=None):
        super(MaskedNLLLoss, self).__init__()
        self.weight = weight
        self.loss = nn.NLLLoss(weight=weight, reduction='sum')

    def forward(self, pred, target, mask):
        """
        pred: (batch*seq_len, n_classes)
        target: (batch*seq_len,)
        mask: (batch, seq_len)
        """
        mask_ = mask.view(-1, 1)  # (batch*seq_len, 1)
        if self.weight is None:
            loss = self.loss(pred * mask_, target) / torch.sum(mask)
        else:
            loss = self.loss(pred * mask_, target) / torch.sum(self.weight[target] * mask_.squeeze())
        return loss
