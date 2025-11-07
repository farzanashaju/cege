import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.utils.rnn import pad_sequence
from torch_geometric.nn import RGCNConv, GraphConv
import numpy as np, itertools, random, copy, math

# for methods and models related to DialogueGCN jump to line 516

# negative log likelihood loss with masking
# used for classification
# applies a mask to ignore padded positions
class MaskedNLLLoss(nn.Module):

    def __init__(self, weight=None):
        super(MaskedNLLLoss, self).__init__()
        self.weight = weight
        self.loss = nn.NLLLoss(weight=weight,
                               reduction='sum')

    def forward(self, pred, target, mask):
        mask_ = mask.view(-1,1) # batch*seq_len, 1
        if type(self.weight)==type(None):
            loss = self.loss(pred*mask_, target)/torch.sum(mask)
        else:
            loss = self.loss(pred*mask_, target)\
                            /torch.sum(self.weight[target]*mask_.squeeze())
        return loss


class SimpleAttention(nn.Module):

    def __init__(self, input_dim):
        super(SimpleAttention, self).__init__()
        self.input_dim = input_dim
        self.scalar = nn.Linear(self.input_dim,1,bias=False)

    def forward(self, M, x=None):
        scale = self.scalar(M) # seq_len, batch, 1
        alpha = F.softmax(scale, dim=0).permute(1,2,0) # batch, 1, seq_len
        attn_pool = torch.bmm(alpha, M.transpose(0,1))[:,0,:] # batch, vector
        return attn_pool, alpha


class MatchingAttention(nn.Module):

    def __init__(self, mem_dim, cand_dim, alpha_dim=None, att_type='general'):
        super(MatchingAttention, self).__init__()
        assert att_type!='dot' or mem_dim==cand_dim
        self.mem_dim = mem_dim
        self.cand_dim = cand_dim
        self.att_type = att_type
        if att_type=='general':
            self.transform = nn.Linear(cand_dim, mem_dim, bias=False)
        if att_type=='general2':
            self.transform = nn.Linear(cand_dim, mem_dim, bias=True)

    def forward(self, M, x, mask=None):
        if type(mask)==type(None):
            mask = torch.ones(M.size(1), M.size(0)).type(M.type())

        if self.att_type=='dot':
            # vector = cand_dim = mem_dim
            M_ = M.permute(1,2,0) # batch, vector, seqlen
            x_ = x.unsqueeze(1) # batch, 1, vector
            alpha = F.softmax(torch.bmm(x_, M_), dim=2) # batch, 1, seqlen
        elif self.att_type=='general':
            M_ = M.permute(1,2,0) # batch, mem_dim, seqlen
            x_ = self.transform(x).unsqueeze(1) # batch, 1, mem_dim
            alpha = F.softmax(torch.bmm(x_, M_), dim=2) # batch, 1, seqlen
        elif self.att_type=='general2':
            M_ = M.permute(1,2,0) # batch, mem_dim, seqlen
            x_ = self.transform(x).unsqueeze(1) # batch, 1, mem_dim
            mask_ = mask.unsqueeze(2).repeat(1, 1, self.mem_dim).transpose(1, 2) # batch, seq_len, mem_dim
            M_ = M_ * mask_
            alpha_ = torch.bmm(x_, M_)*mask.unsqueeze(1)
            alpha_ = torch.tanh(alpha_)
            alpha_ = F.softmax(alpha_, dim=2)
            alpha_masked = alpha_*mask.unsqueeze(1) # batch, 1, seqlen
            alpha_sum = torch.sum(alpha_masked, dim=2, keepdim=True) # batch, 1, 1
            alpha = alpha_masked/alpha_sum # batch, 1, 1 ; normalized
        else:
            M_ = M.transpose(0,1) # batch, seqlen, mem_dim
            x_ = x.unsqueeze(1).expand(-1,M.size()[0],-1) # batch, seqlen, cand_dim
            M_x_ = torch.cat([M_,x_],2) # batch, seqlen, mem_dim+cand_dim
            mx_a = F.tanh(self.transform(M_x_)) # batch, seqlen, alpha_dim
            alpha = F.softmax(self.vector_prod(mx_a),1).transpose(1,2) # batch, 1, seqlen

        attn_pool = torch.bmm(alpha, M.transpose(0,1))[:,0,:] # batch, mem_dim
        return attn_pool, alpha


class Attention(nn.Module):
    def __init__(self, embed_dim, hidden_dim=None, out_dim=None, n_head=1, score_function='dot_product', dropout=0):
        ''' Attention Mechanism
        :param embed_dim:
        :param hidden_dim:
        :param out_dim:
        :param n_head: num of head (Multi-Head Attention)
        :return (?, q_len, out_dim,)
        '''
        super(Attention, self).__init__()
        if hidden_dim is None:
            hidden_dim = embed_dim // n_head
        if out_dim is None:
            out_dim = embed_dim
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.n_head = n_head
        self.score_function = score_function
        self.w_k = nn.Linear(embed_dim, n_head * hidden_dim)
        self.w_q = nn.Linear(embed_dim, n_head * hidden_dim)
        self.proj = nn.Linear(n_head * hidden_dim, out_dim)
        self.dropout = nn.Dropout(dropout)
        if score_function == 'mlp':
            self.weight = nn.Parameter(torch.Tensor(hidden_dim*2))
        elif self.score_function == 'bi_linear':
            self.weight = nn.Parameter(torch.Tensor(hidden_dim, hidden_dim))
        else:  # dot_product / scaled_dot_product
            self.register_parameter('weight', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.hidden_dim)
        if self.weight is not None:
            self.weight.data.uniform_(-stdv, stdv)

    def forward(self, k, q):
        if len(q.shape) == 2:  # q_len missing
            q = torch.unsqueeze(q, dim=1)
        if len(k.shape) == 2:  # k_len missing
            k = torch.unsqueeze(k, dim=1)
        mb_size = k.shape[0]  # ?
        k_len = k.shape[1]
        q_len = q.shape[1]
        kx = self.w_k(k).view(mb_size, k_len, self.n_head, self.hidden_dim)
        kx = kx.permute(2, 0, 1, 3).contiguous().view(-1, k_len, self.hidden_dim)
        qx = self.w_q(q).view(mb_size, q_len, self.n_head, self.hidden_dim)
        qx = qx.permute(2, 0, 1, 3).contiguous().view(-1, q_len, self.hidden_dim)
        if self.score_function == 'dot_product':
            kt = kx.permute(0, 2, 1)
            score = torch.bmm(qx, kt)
        elif self.score_function == 'scaled_dot_product':
            kt = kx.permute(0, 2, 1)
            qkt = torch.bmm(qx, kt)
            score = torch.div(qkt, math.sqrt(self.hidden_dim))
        elif self.score_function == 'mlp':
            kxx = torch.unsqueeze(kx, dim=1).expand(-1, q_len, -1, -1)
            qxx = torch.unsqueeze(qx, dim=2).expand(-1, -1, k_len, -1)
            kq = torch.cat((kxx, qxx), dim=-1)
            score = torch.tanh(torch.matmul(kq, self.weight))
        elif self.score_function == 'bi_linear':
            qw = torch.matmul(qx, self.weight)
            kt = kx.permute(0, 2, 1)
            score = torch.bmm(qw, kt)
        else:
            raise RuntimeError('invalid score_function')
        score = F.softmax(score, dim=0)
        output = torch.bmm(score, kx)
        output = torch.cat(torch.split(output, mb_size, dim=0), dim=-1)
        output = self.proj(output)
        output = self.dropout(output)
        return output, score


class GRUModel(nn.Module):

    def __init__(self, D_m, D_e, D_h, n_classes=7, dropout=0.5):
        
        super(GRUModel, self).__init__()
        
        self.n_classes = n_classes
        self.dropout   = nn.Dropout(dropout)
        self.gru = nn.GRU(input_size=D_m, hidden_size=D_e, num_layers=2, bidirectional=True, dropout=dropout)
        self.matchatt = MatchingAttention(2*D_e, 2*D_e, att_type='general2')
        self.linear = nn.Linear(2*D_e, D_h)
        self.smax_fc = nn.Linear(D_h, n_classes)
        
    def forward(self, U, qmask, umask, att2=True):
        """
        U -> seq_len, batch, D_m
        qmask -> seq_len, batch, party
        """
        emotions, hidden = self.gru(U)
        alpha, alpha_f, alpha_b = [], [], []
        
        if att2:
            att_emotions = []
            alpha = []
            for t in emotions:
                att_em, alpha_ = self.matchatt(emotions,t,mask=umask)
                att_emotions.append(att_em.unsqueeze(0))
                alpha.append(alpha_[:,0,:])
            att_emotions = torch.cat(att_emotions,dim=0)
            hidden = F.relu(self.linear(att_emotions))
        else:
            hidden = F.relu(self.linear(emotions))
        
        # hidden = F.relu(self.linear(emotions))
        hidden = self.dropout(hidden)
        log_prob = F.log_softmax(self.smax_fc(hidden), 2)
        return log_prob, alpha, alpha_f, alpha_b, emotions


class LSTMModel(nn.Module):

    def __init__(self, D_m, D_e, D_h, n_classes=7, dropout=0.5):
        
        super(LSTMModel, self).__init__()
        
        self.n_classes = n_classes
        self.dropout   = nn.Dropout(dropout)
        self.lstm = nn.LSTM(input_size=D_m, hidden_size=D_e, num_layers=2, bidirectional=True, dropout=dropout)
        self.matchatt = MatchingAttention(2*D_e, 2*D_e, att_type='general2')
        self.linear = nn.Linear(2*D_e, D_h)
        self.smax_fc = nn.Linear(D_h, n_classes)

    def forward(self, U, qmask, umask, att2=True):
        """
        U -> seq_len, batch, D_m
        qmask -> seq_len, batch, party
        """
        emotions, hidden = self.lstm(U)
        alpha, alpha_f, alpha_b = [], [], []
        
        if att2:
            att_emotions = []
            alpha = []
            for t in emotions:
                att_em, alpha_ = self.matchatt(emotions,t,mask=umask)
                att_emotions.append(att_em.unsqueeze(0))
                alpha.append(alpha_[:,0,:])
            att_emotions = torch.cat(att_emotions,dim=0)
            hidden = F.relu(self.linear(att_emotions))
        else:
            hidden = F.relu(self.linear(emotions))
        
        # hidden = F.relu(self.linear(emotions))
        hidden = self.dropout(hidden)
        log_prob = F.log_softmax(self.smax_fc(hidden), 2)
        return log_prob, alpha, alpha_f, alpha_b, emotions


class MaskedEdgeAttention(nn.Module):

    def __init__(self, input_dim, max_seq_len, no_cuda):
        """
        Method to compute the edge weights, as in Equation 1. in the paper. 
        attn_type = 'attn1' refers to the equation in the paper.
        For slightly different attention mechanisms refer to attn_type = 'attn2' or attn_type = 'attn3'
        """

        super(MaskedEdgeAttention, self).__init__()
        
        self.input_dim = input_dim
        self.max_seq_len = max_seq_len
        self.scalar = nn.Linear(self.input_dim, self.max_seq_len, bias=False)
        self.matchatt = MatchingAttention(self.input_dim, self.input_dim, att_type='general2')
        self.simpleatt = SimpleAttention(self.input_dim)
        self.att = Attention(self.input_dim, score_function='mlp')
        self.no_cuda = no_cuda

    def forward(self, M, lengths, edge_ind):
        """
        M -> (seq_len, batch, vector)
        lengths -> length of the sequences in the batch
        """
        attn_type = 'attn1'

        if attn_type == 'attn1':

            scale = self.scalar(M)
            # scale = torch.tanh(scale)
            alpha = F.softmax(scale, dim=0).permute(1, 2, 0)
            
            # Check if CUDA is actually available and not disabled
            use_cuda = torch.cuda.is_available() and not self.no_cuda
            if use_cuda:
                mask = Variable(torch.ones(alpha.size()) * 1e-10).detach().cuda()
                mask_copy = Variable(torch.zeros(alpha.size())).detach().cuda()
                
            else:
                mask = Variable(torch.ones(alpha.size()) * 1e-10).detach()
                mask_copy = Variable(torch.zeros(alpha.size())).detach()
            
            # Build mask by iterating over edge indices (safer than numpy-tuple indexing)
            for i_batch, edges in enumerate(edge_ind):
                for x in edges:
                    src = int(x[0])
                    tgt = int(x[1])
                    # ensure indices are within alpha/mask dimensions
                    if i_batch < mask.size(0) and src < mask.size(1) and tgt < mask.size(2):
                        mask[i_batch, src, tgt] = 1
                        mask_copy[i_batch, src, tgt] = 1
            masked_alpha = alpha * mask
            _sums = masked_alpha.sum(-1, keepdim=True)
            scores = masked_alpha.div(_sums) * mask_copy
            return scores

        elif attn_type == 'attn2':
            scores = torch.zeros(M.size(1), self.max_seq_len, self.max_seq_len, requires_grad=True)

            # Check if CUDA is actually available and not disabled
            use_cuda = torch.cuda.is_available() and not self.no_cuda
            if use_cuda:
                scores = scores.cuda()


            for j in range(M.size(1)):
            
                ei = np.array(edge_ind[j])

                for node in range(lengths[j]):
                
                    neighbour = ei[ei[:, 0] == node, 1]

                    M_ = M[neighbour, j, :].unsqueeze(1)
                    t = M[node, j, :].unsqueeze(0)
                    _, alpha_ = self.simpleatt(M_, t)
                    scores[j, node, neighbour] = alpha_

        elif attn_type == 'attn3':
            scores = torch.zeros(M.size(1), self.max_seq_len, self.max_seq_len, requires_grad=True)

            # Check if CUDA is actually available and not disabled
            use_cuda = torch.cuda.is_available() and not self.no_cuda
            if use_cuda:
                scores = scores.cuda()

            for j in range(M.size(1)):

                ei = np.array(edge_ind[j])

                for node in range(lengths[j]):

                    neighbour = ei[ei[:, 0] == node, 1]

                    M_ = M[neighbour, j, :].unsqueeze(1).transpose(0, 1)
                    t = M[node, j, :].unsqueeze(0).unsqueeze(0).repeat(len(neighbour), 1, 1).transpose(0, 1)
                    _, alpha_ = self.att(M_, t)
                    scores[j, node, neighbour] = alpha_[0, :, 0]

        return scores


def pad(tensor, length, no_cuda):
    if isinstance(tensor, Variable):
        var = tensor
        if length > var.size(0):
            #if torch.cuda.is_available():
            if not no_cuda:
                return torch.cat([var, torch.zeros(length - var.size(0), *var.size()[1:]).cuda()])
            else:
                return torch.cat([var, torch.zeros(length - var.size(0), *var.size()[1:])])
        else:
            return var
    else:
        if length > tensor.size(0):
            #if torch.cuda.is_available():
            if not no_cuda:
                return torch.cat([tensor, torch.zeros(length - tensor.size(0), *tensor.size()[1:]).cuda()])
            else:
                return torch.cat([tensor, torch.zeros(length - tensor.size(0), *tensor.size()[1:])])
        else:
            return tensor


def edge_perms(l, window_past, window_future):
    """
    Method to construct the edges considering the past and future window.
    """

    all_perms = set()
    array = np.arange(l)
    for j in range(l):
        perms = set()
        
        if window_past == -1 and window_future == -1:
            eff_array = array
        elif window_past == -1:
            eff_array = array[:min(l, j+window_future+1)]
        elif window_future == -1:
            eff_array = array[max(0, j-window_past):]
        else:
            eff_array = array[max(0, j-window_past):min(l, j+window_future+1)]
        
        for item in eff_array:
            perms.add((j, item))
        all_perms = all_perms.union(perms)
    return list(all_perms)
    
        
def batch_graphify(features, qmask, lengths, window_past, window_future, edge_type_mapping, att_model, no_cuda):
    """
    Method to prepare the data format required for the GCN network. Pytorch geometric puts all nodes for classification 
    in one single graph. Following this, we create a single graph for a mini-batch of dialogue instances. This method 
    ensures that the various graph indexing is properly carried out so as to make sure that, utterances (nodes) from 
    each dialogue instance will have edges with utterances in that same dialogue instance, but not with utternaces 
    from any other dialogue instances in that mini-batch.
    """
    
    edge_index, edge_norm, edge_type, node_features = [], [], [], []
    batch_size = features.size(1)
    length_sum = 0
    edge_ind = []
    edge_index_lengths = []
    
    for j in range(batch_size):
        edge_ind.append(edge_perms(lengths[j], window_past, window_future))
    
    # scores are the edge weights
    scores = att_model(features, lengths, edge_ind)

    for j in range(batch_size):
        node_features.append(features[:lengths[j], j, :])
    
        perms1 = edge_perms(lengths[j], window_past, window_future)
        perms2 = [(item[0]+length_sum, item[1]+length_sum) for item in perms1]
        length_sum += lengths[j]

        edge_index_lengths.append(len(perms1))
    
        for item1, item2 in zip(perms1, perms2):
            edge_index.append(torch.tensor([item2[0], item2[1]]))
            edge_norm.append(scores[j, item1[0], item1[1]])
        
            speaker0 = (qmask[item1[0], j, :] == 1).nonzero()[0][0].tolist()
            speaker1 = (qmask[item1[1], j, :] == 1).nonzero()[0][0].tolist()
        
            if item1[0] < item1[1]:
                # edge_type.append(0) # ablation by removing speaker dependency: only 2 relation types
                # edge_type.append(edge_type_mapping[str(speaker0) + str(speaker1) + '0']) # ablation by removing temporal dependency: M^2 relation types
                edge_type.append(edge_type_mapping[str(speaker0) + str(speaker1) + '0'])
            else:
                # edge_type.append(1) # ablation by removing speaker dependency: only 2 relation types
                # edge_type.append(edge_type_mapping[str(speaker0) + str(speaker1) + '0']) # ablation by removing temporal dependency: M^2 relation types
                edge_type.append(edge_type_mapping[str(speaker0) + str(speaker1) + '1'])
    
    node_features = torch.cat(node_features, dim=0)
    edge_index = torch.stack(edge_index).transpose(0, 1)
    edge_norm = torch.stack(edge_norm)
    edge_type = torch.tensor(edge_type)

    # Check if CUDA is actually available and not disabled
    use_cuda = torch.cuda.is_available() and not no_cuda
    if use_cuda:
        node_features = node_features.cuda()
        edge_index = edge_index.cuda()
        edge_norm = edge_norm.cuda()
        edge_type = edge_type.cuda()
    
    return node_features, edge_index, edge_norm, edge_type, edge_index_lengths 


def attentive_node_features(emotions, seq_lengths, umask, matchatt_layer, no_cuda):
    """
    Method to obtain attentive node features over the graph convoluted features, as in Equation 4, 5, 6. in the paper.
    """
    
    input_conversation_length = torch.tensor(seq_lengths)
    start_zero = input_conversation_length.data.new(1).zero_()
    
    # Check if CUDA is actually available and not disabled
    use_cuda = torch.cuda.is_available() and not no_cuda
    if use_cuda:
        input_conversation_length = input_conversation_length.cuda()
        start_zero = start_zero.cuda()

    max_len = max(seq_lengths)

    start = torch.cumsum(torch.cat((start_zero, input_conversation_length[:-1])), 0)

    emotions = torch.stack([pad(emotions.narrow(0, s, l), max_len, no_cuda) 
                                for s, l in zip(start.data.tolist(),
                                input_conversation_length.data.tolist())], 0).transpose(0, 1)


    alpha, alpha_f, alpha_b = [], [], []
    att_emotions = []

    for t in emotions:
        att_em, alpha_ = matchatt_layer(emotions, t, mask=umask)
        att_emotions.append(att_em.unsqueeze(0))
        alpha.append(alpha_[:,0,:])

    att_emotions = torch.cat(att_emotions, dim=0)

    return att_emotions


def classify_node_features(emotions, seq_lengths, umask, matchatt_layer, linear_layer, dropout_layer, smax_fc_layer, nodal_attn, avec, no_cuda):

    if nodal_attn:

        emotions = attentive_node_features(emotions, seq_lengths, umask, matchatt_layer, no_cuda)
        hidden = F.relu(linear_layer(emotions))
        hidden = dropout_layer(hidden)
        hidden = smax_fc_layer(hidden)

        if avec:
            return torch.cat([hidden[:, j, :][:seq_lengths[j]] for j in range(len(seq_lengths))])

        log_prob = F.log_softmax(hidden, 2)
        log_prob = torch.cat([log_prob[:, j, :][:seq_lengths[j]] for j in range(len(seq_lengths))])
        return log_prob

    else:

        hidden = F.relu(linear_layer(emotions))
        hidden = dropout_layer(hidden)
        hidden = smax_fc_layer(hidden)

        if avec:
            return hidden

        log_prob = F.log_softmax(hidden, 1)
        return log_prob


class TemporalMemoryModule(nn.Module):
    """
    Temporal Memory Module for maintaining and updating memory states of utterances.
    Implements the LSTM-based memory update mechanism.
    """
    def __init__(self, memory_dim, utterance_dim, speaker_dim, global_context_dim):
        super(TemporalMemoryModule, self).__init__()
        self.memory_dim = memory_dim
        self.utterance_dim = utterance_dim
        self.speaker_dim = speaker_dim
        self.global_context_dim = global_context_dim
        # LSTM cell for memory update: input is utterance_emb || speaker_state || global_context
        lstm_input_size = self.utterance_dim + self.speaker_dim + self.global_context_dim
        self.memory_lstm = nn.LSTMCell(lstm_input_size, self.memory_dim)
        
    def forward(self, prev_memory, prev_cell, utterance_emb, speaker_state, global_context):
        """
        Update memory state for a node.
        
        Args:
            prev_memory: Previous memory state (batch, memory_dim)
            prev_cell: Previous LSTM cell state (batch, memory_dim)
            utterance_emb: Context-aware embedding of utterance (batch, emb_dim)
            speaker_state: Speaker state (batch, speaker_dim)
            global_context: Global conversation state (batch, context_dim)
        
        Returns:
            new_memory: Updated memory state (batch, memory_dim)
            new_cell: Updated LSTM cell state (batch, memory_dim)
        """
        # Concatenate context information (speaker_state and global_context)
        # Ensure tensors are of shape (batch, dim)
        if speaker_state is None:
            context_input = global_context
        else:
            context_input = torch.cat([speaker_state, global_context], dim=-1)

        # LSTM update: input is (utterance_emb || context_input)
        lstm_input = torch.cat([utterance_emb, context_input], dim=-1)
        new_memory, new_cell = self.memory_lstm(lstm_input, (prev_memory, prev_cell))
        return new_memory, new_cell


class TemporalStateAnalyzer(nn.Module):
    """
    Analyzes temporal state change based on node memories and speaker states.
    Computes Δ_t as described in the paper.
    """
    def __init__(self, memory_dim, speaker_dim, hidden_dim):
        super(TemporalStateAnalyzer, self).__init__()
        self.fc1 = nn.Linear(memory_dim + speaker_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, memory_dim)
        
    def forward(self, node_memories, utterance_emb, speaker_state):
        """
        Compute temporal state change.
        
        Args:
            node_memories: All node memories at t-1 (seq_len, batch, memory_dim)
            utterance_emb: Current utterance embedding (batch, emb_dim)
            speaker_state: Speaker state (batch, speaker_dim)
        
        Returns:
            delta_t: Temporal state change (batch, memory_dim)
        """
        # Aggregate node memories across nodes to get a summary per batch
        # node_memories: (seq_len, batch, memory_dim) -> memory_summary: (batch, memory_dim)
        if node_memories.dim() == 3:
            memory_summary = torch.mean(node_memories, dim=0)
        else:
            # fallback: assume already (batch, memory_dim)
            memory_summary = node_memories

        # speaker_state expected shape: (batch, speaker_dim)
        if speaker_state is None:
            combined = memory_summary
        else:
            # If speaker_state has an extra time dim (1, batch, dim), squeeze it
            if speaker_state.dim() == 3 and speaker_state.size(0) == 1:
                speaker_state = speaker_state.squeeze(0)
            combined = torch.cat([memory_summary, speaker_state], dim=-1)

        delta_t = torch.tanh(self.fc1(combined))
        delta_t = self.fc2(delta_t)
        return delta_t


class DynamicEdgeUpdateModule(nn.Module):
    """
    Updates edge weights dynamically based on temporal changes and speaker states.
    Implements the edge weight update function f_temporal.
    """
    def __init__(self, memory_dim, speaker_dim, hidden_dim):
        super(DynamicEdgeUpdateModule, self).__init__()
        self.fc1 = nn.Linear(memory_dim + speaker_dim * 2, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)
        
    def forward(self, prev_alpha, delta_t, speaker_state_i, speaker_state_j):
        """
        Update edge weights.
        
        Args:
            prev_alpha: Previous edge weight (batch * seq_len * seq_len)
            delta_t: Temporal state change (batch, memory_dim)
            speaker_state_i: Speaker state of target node (batch, speaker_dim)
            speaker_state_j: Speaker state of source node (batch, speaker_dim)
        
        Returns:
            new_alpha: Updated edge weight (batch * seq_len * seq_len)
        """
        # Ensure inputs have compatible shapes. We expect per-edge tensors
        # prev_alpha: (num_edges, 1) or (num_edges,)
        if prev_alpha.dim() == 1:
            prev_alpha = prev_alpha.unsqueeze(-1)

        # delta_t: (num_edges, memory_dim) or (num_edges, memory_dim)
        # speaker_state_i/j: (num_edges, speaker_dim)
        # Concatenate along feature dim
        combined = torch.cat([delta_t, speaker_state_i, speaker_state_j], dim=-1)
        hidden = torch.tanh(self.fc1(combined))
        update_factor = torch.sigmoid(self.fc2(hidden))  # (num_edges, 1)

        # Update edge weight: small step from prev_alpha
        step = 0.1 * update_factor
        new_alpha = prev_alpha + step
        new_alpha = new_alpha.squeeze(-1)
        new_alpha = torch.clamp(new_alpha, 0.0, 1.0)
        return new_alpha


class TemporalAttentionModule(nn.Module):
    """
    Computes temporal attention over past memory states with decay.
    Implements Equation in the paper for temporal attention.
    """
    def __init__(self, hidden_dim, memory_dim):
        super(TemporalAttentionModule, self).__init__()
        self.attention_weight = nn.Linear(hidden_dim, hidden_dim)
        self.score_projection = nn.Linear(hidden_dim, 1)
        # project memory to attention space
        self.memory_proj = nn.Linear(memory_dim, hidden_dim)
        self.decay_rate = nn.Parameter(torch.tensor(0.1))  # λ_decay
        
    def forward(self, current_features, memory_history):
        """
        Compute attention weights over memory history with temporal decay.
        
        Args:
            current_features: Current node features (batch, hidden_dim)
            memory_history: Historical memory states (seq_len, batch, memory_dim)
        
        Returns:
            context_vector: Weighted sum of historical memories (batch, memory_dim)
            attention_weights: Attention weights (batch, seq_len)
        """
        seq_len = memory_history.size(0)
        batch_size = current_features.size(0)

        # Compute attention scores for each time step using both current features and memory_history
        scores = []
        for t in range(seq_len):
            mem_t = memory_history[t]  # (batch, memory_dim)
            mem_proj = self.memory_proj(mem_t)  # (batch, hidden_dim)
            cur_proj = self.attention_weight(current_features)  # (batch, hidden_dim)
            score_t = self.score_projection(torch.tanh(cur_proj + mem_proj))  # (batch, 1)
            scores.append(score_t)

        scores = torch.cat(scores, dim=-1)  # (batch, seq_len)
        
        # Apply softmax
        attention_weights = F.softmax(scores, dim=-1)  # (batch, seq_len)
        
        # Apply temporal decay
        time_distances = torch.arange(seq_len, dtype=torch.float32, device=current_features.device)
        time_distances = seq_len - time_distances - 1  # Distance from current time
        decay_factors = torch.exp(-self.decay_rate * time_distances)  # Exponential decay
        
        # Apply decay to attention weights
        attention_weights = attention_weights * decay_factors.unsqueeze(0)
        attention_weights = attention_weights / (attention_weights.sum(dim=-1, keepdim=True) + 1e-8)

        # Compute context vector as weighted sum of memories
        # memory_history: (seq_len, batch, memory_dim)
        context_vector = torch.einsum('bt,tbm->bm', attention_weights, memory_history)
        return context_vector, attention_weights


class GraphNetwork(torch.nn.Module):
    def __init__(self, num_features, num_classes, num_relations, max_seq_len, hidden_size=64, dropout=0.5, no_cuda=False):
        """
        The Speaker-level context encoder in the form of a 2 layer GCN with temporal memory.
        """
        super(GraphNetwork, self).__init__()
        
        self.conv1 = RGCNConv(num_features, hidden_size, num_relations, num_bases=30)
        self.conv2 = GraphConv(hidden_size, hidden_size)
        self.matchatt = MatchingAttention(num_features+hidden_size, num_features+hidden_size, att_type='general2')
        self.linear   = nn.Linear(num_features+hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.smax_fc  = nn.Linear(hidden_size, num_classes)
        self.no_cuda = no_cuda
        
        # Temporal memory modules
        self.memory_dim = hidden_size
        # utterance_dim = num_features, speaker_dim ~= num_features, global_context_dim = memory_dim
        self.temporal_memory = TemporalMemoryModule(self.memory_dim, utterance_dim=num_features, speaker_dim=num_features, global_context_dim=self.memory_dim)
        self.temporal_analyzer = TemporalStateAnalyzer(self.memory_dim, num_features, hidden_size)
        self.edge_updater = DynamicEdgeUpdateModule(self.memory_dim, speaker_dim=num_features, hidden_dim=hidden_size)
        self.temporal_attention = TemporalAttentionModule(hidden_size, self.memory_dim)

    def forward(self, x, edge_index, edge_norm, edge_type, seq_lengths, umask, nodal_attn, avec):
        # RGCNConv in this environment expects (x, edge_index, edge_type)
        # edge_norm is not a supported positional argument; omit it here.
        out = self.conv1(x, edge_index, edge_type)
        # Use edge_norm as edge weights in the second GraphConv (if provided)
        if edge_norm is not None:
            try:
                out = self.conv2(out, edge_index, edge_weight=edge_norm)
            except TypeError:
                # Fallback if GraphConv in this version doesn't accept edge_weight
                out = self.conv2(out, edge_index)
        else:
            out = self.conv2(out, edge_index)
        emotions = torch.cat([x, out], dim=-1)
        log_prob = classify_node_features(emotions, seq_lengths, umask, self.matchatt, self.linear, self.dropout, self.smax_fc, nodal_attn, avec, self.no_cuda)
        return log_prob



class DialogueGCNModel(nn.Module):

    def __init__(self, base_model, D_m, D_g, D_p, D_e, D_h, D_a, graph_hidden_size, n_speakers, max_seq_len, window_past, window_future,
                 n_classes=7, listener_state=False, context_attention='simple', dropout_rec=0.5, dropout=0.5, nodal_attention=True, avec=False, no_cuda=False):
        
        super(DialogueGCNModel, self).__init__()

        self.base_model = base_model
        self.avec = avec
        self.no_cuda = no_cuda

        # The base model is the sequential context encoder.
        if self.base_model == 'LSTM':
            self.lstm = nn.LSTM(input_size=D_m, hidden_size=D_e, num_layers=2, bidirectional=True, dropout=dropout)

        elif self.base_model == 'GRU':
            self.gru = nn.GRU(input_size=D_m, hidden_size=D_e, num_layers=2, bidirectional=True, dropout=dropout)


        elif self.base_model == 'None':
            self.base_linear = nn.Linear(D_m, 2*D_e)

        else:
            print ('Base model must be LSTM / GRU')
            raise NotImplementedError 

        n_relations = 2 * n_speakers ** 2
        self.window_past = window_past
        self.window_future = window_future

        self.att_model = MaskedEdgeAttention(2*D_e, max_seq_len, self.no_cuda)
        self.nodal_attention = nodal_attention

        self.graph_net = GraphNetwork(2*D_e, n_classes, n_relations, max_seq_len, graph_hidden_size, dropout, self.no_cuda)

        edge_type_mapping = {}
        for j in range(n_speakers):
            for k in range(n_speakers):
                edge_type_mapping[str(j) + str(k) + '0'] = len(edge_type_mapping)
                edge_type_mapping[str(j) + str(k) + '1'] = len(edge_type_mapping)

        self.edge_type_mapping = edge_type_mapping


    def _reverse_seq(self, X, mask):
        """
        X -> seq_len, batch, dim
        mask -> batch, seq_len
        """
        X_ = X.transpose(0,1)
        mask_sum = torch.sum(mask, 1).int()

        xfs = []
        for x, c in zip(X_, mask_sum):
            xf = torch.flip(x[:c], [0])
            xfs.append(xf)

        return pad_sequence(xfs)


    def forward(self, U, qmask, umask, seq_lengths):
        if self.base_model == 'LSTM':
            emotions, hidden = self.lstm(U)

        elif self.base_model == 'GRU':
            emotions, hidden = self.gru(U)

        elif self.base_model == 'None':
            emotions = self.base_linear(U)

        features, edge_index, edge_norm, edge_type, edge_index_lengths = batch_graphify(emotions, qmask, seq_lengths, self.window_past, self.window_future, self.edge_type_mapping, self.att_model, self.no_cuda)
        log_prob = self.graph_net(features, edge_index, edge_norm, edge_type, seq_lengths, umask, self.nodal_attention, self.avec)

        return log_prob, edge_index, edge_norm, edge_type, edge_index_lengths


class DialogueGCNTemporalModel(nn.Module):
    """
    Enhanced DialogueGCN Model with Temporal Memory and Dynamic Graph Evolution.
    
    This model extends the basic DialogueGCN with:
    - Temporal memory modules for each node
    - Dynamic edge weight updates
    - Temporal attention over history
    - Temporal consistency loss during training
    """

    def __init__(self, base_model, D_m, D_g, D_p, D_e, D_h, D_a, graph_hidden_size, n_speakers, max_seq_len, 
                 window_past, window_future, n_classes=7, listener_state=False, context_attention='simple', 
                 dropout_rec=0.5, dropout=0.5, nodal_attention=True, avec=False, no_cuda=False,
                 temporal_decay=0.1, edge_prune_threshold=0.1):
        
        super(DialogueGCNTemporalModel, self).__init__()

        self.base_model = base_model
        self.avec = avec
        self.no_cuda = no_cuda
        self.graph_hidden_size = graph_hidden_size
        self.n_speakers = n_speakers
        self.max_seq_len = max_seq_len
        self.temporal_decay = temporal_decay
        self.edge_prune_threshold = edge_prune_threshold

        # The base model is the sequential context encoder
        if self.base_model == 'LSTM':
            self.lstm = nn.LSTM(input_size=D_m, hidden_size=D_e, num_layers=2, bidirectional=True, dropout=dropout)

        elif self.base_model == 'GRU':
            self.gru = nn.GRU(input_size=D_m, hidden_size=D_e, num_layers=2, bidirectional=True, dropout=dropout)

        elif self.base_model == 'None':
            self.base_linear = nn.Linear(D_m, 2*D_e)

        else:
            print('Base model must be LSTM / GRU')
            raise NotImplementedError

        n_relations = 2 * n_speakers ** 2
        self.window_past = window_past
        self.window_future = window_future

        self.att_model = MaskedEdgeAttention(2*D_e, max_seq_len, self.no_cuda)
        self.nodal_attention = nodal_attention

        self.graph_net = GraphNetwork(2*D_e, n_classes, n_relations, max_seq_len, graph_hidden_size, dropout, self.no_cuda)

        self.memory_dim = graph_hidden_size
        self.temporal_memory = TemporalMemoryModule(self.memory_dim, utterance_dim=2*D_e, speaker_dim=2*D_e, global_context_dim=self.memory_dim)
        self.temporal_analyzer = TemporalStateAnalyzer(self.memory_dim, 2*D_e, self.memory_dim)
        self.edge_updater = DynamicEdgeUpdateModule(self.memory_dim, speaker_dim=2*D_e, hidden_dim=self.memory_dim)
        self.temporal_attention = TemporalAttentionModule(self.memory_dim, self.memory_dim)
        self.global_context_lstm = nn.LSTMCell(2*D_e, self.memory_dim)

        edge_type_mapping = {}
        for j in range(n_speakers):
            for k in range(n_speakers):
                edge_type_mapping[str(j) + str(k) + '0'] = len(edge_type_mapping)
                edge_type_mapping[str(j) + str(k) + '1'] = len(edge_type_mapping)

        self.edge_type_mapping = edge_type_mapping


    def _reverse_seq(self, X, mask):
        """
        X -> seq_len, batch, dim
        mask -> batch, seq_len
        """
        X_ = X.transpose(0,1)
        mask_sum = torch.sum(mask, 1).int()

        xfs = []
        for x, c in zip(X_, mask_sum):
            xf = torch.flip(x[:c], [0])
            xfs.append(xf)

        return pad_sequence(xfs)

    def _get_speaker_states(self, emotions, qmask):
        """
        Extract speaker states from emotions based on speaker mask.
        
        Args:
            emotions: (seq_len, batch, 2*D_e)
            qmask: (seq_len, batch, n_speakers)
        
        Returns:
            speaker_states: (seq_len, batch, n_speakers*2*D_e)
        """
        # Return per-node active speaker embedding (seq_len, batch, emb_dim)
        seq_len, batch_size, emb_dim = emotions.size()
        speaker_states = torch.zeros(seq_len, batch_size, emb_dim, device=emotions.device)
        for t in range(seq_len):
            for b in range(batch_size):
                # find active speaker index
                inds = (qmask[t, b, :] == 1).nonzero()
                if inds.numel() > 0:
                    sidx = inds[0][0].item()
                    speaker_states[t, b, :] = emotions[t, b, :]
                else:
                    # no active speaker; leave zeros
                    pass
        return speaker_states

    def _update_temporal_memories(self, emotions, qmask, seq_lengths):
        seq_len, batch_size = emotions.size(0), emotions.size(1)

        # Speaker states per node (seq_len, batch, emb_dim)
        speaker_states = self._get_speaker_states(emotions, qmask)

        # Initialize per-node memory and cell states: (seq_len, batch, memory_dim)
        prev_memory = torch.zeros(seq_len, batch_size, self.memory_dim, device=emotions.device)
        prev_cell = torch.zeros(seq_len, batch_size, self.memory_dim, device=emotions.device)

        node_memories = []
        global_contexts = []

        # Initialize global context LSTM states
        global_h = torch.zeros(batch_size, self.memory_dim, device=emotions.device)
        global_c = torch.zeros(batch_size, self.memory_dim, device=emotions.device)

        # Iterate over global time steps; at each time step update global context
        # and update all node memories using the TemporalMemoryModule
        for t in range(seq_len):
            # Update global context with current utterance (batch, emb_dim)
            global_h, global_c = self.global_context_lstm(emotions[t], (global_h, global_c))

            # For each node index i, update its memory using its own utterance embedding
            next_memory = torch.zeros_like(prev_memory)
            next_cell = torch.zeros_like(prev_cell)

            for i in range(seq_len):
                # prev memory and cell for node i: (batch, memory_dim)
                pm = prev_memory[i]
                pc = prev_cell[i]
                # utterance embedding for node i: (batch, emb_dim)
                u_emb = emotions[i]
                # speaker state for node i: (batch, emb_dim)
                sp_state = speaker_states[i]

                # Update via TemporalMemoryModule
                new_mem, new_cell = self.temporal_memory(pm, pc, u_emb, sp_state, global_h)
                next_memory[i] = new_mem
                next_cell[i] = new_cell

            # Save the updated memories for this global time t
            # next_memory: (seq_len, batch, memory_dim) -> per-node memories after time t
            node_memories.append(next_memory.clone())
            global_contexts.append(global_h.clone())

            # Set prev -> next for next global step
            prev_memory = next_memory
            prev_cell = next_cell

        # Build a global memory summary per time step by averaging node memories
        # global_memory_states[t] will be (batch, memory_dim)
        global_memory_states = [nm.mean(dim=0) for nm in node_memories]

        return torch.stack(global_memory_states), torch.stack(global_contexts)

    def _compute_temporal_consistency_loss(self, memories_t, memories_prev):
        return torch.mean((memories_t - memories_prev.detach()) ** 2)

    def _update_dynamic_edges(self, edge_index, edge_norm, node_memories, emotions, 
                             seq_lengths, qmask, prev_delta_t=None):
        # Compute temporal state change: Δ_t = f_analyze(M^{t-1}, g_t, Q_s^{t-1})
        # This captures changes in emotion, topic, and speaker patterns
        # Obtain speaker states and compute Δ_t per batch using temporal analyzer
        speaker_states = self._get_speaker_states(emotions, qmask)
        # Use the latest utterance embedding as utterance_emb (batch, emb_dim)
        utterance_emb = emotions[-1]
        # Use the speaker state at current time
        speaker_state_now = speaker_states[-1]
        delta_t_batch = self.temporal_analyzer(node_memories, utterance_emb, speaker_state_now)  # (batch, memory_dim)
        
        # Update edge weights: α_{ij}^t = f_temporal(α_{ij}^{t-1}, Δ_t, Q_i, Q_j)
        num_edges = edge_norm.size(0)
        
        if num_edges > 0:
            # Build node embeddings (concatenation per dialogue as in batch_graphify)
            batch_size = qmask.size(1)
            node_embeddings = torch.cat([emotions[:seq_lengths[j], j, :] for j in range(batch_size)], dim=0)

            # Map edges to source/target node embeddings
            src_idx = edge_index[0].long()
            tgt_idx = edge_index[1].long()
            speaker_state_src = node_embeddings[src_idx]  # (num_edges, emb_dim)
            speaker_state_tgt = node_embeddings[tgt_idx]  # (num_edges, emb_dim)

            # Average delta_t across batch to obtain a global temporal change vector
            delta_vec = delta_t_batch.mean(dim=0)  # (memory_dim,)
            delta_t_expanded = delta_vec.unsqueeze(0).repeat(num_edges, 1).to(edge_norm.device)

            # Apply edge weight update based on temporal dynamics
            updated_weights = self.edge_updater(
                edge_norm,               # (num_edges,) - previous edge weight
                delta_t_expanded,       # (num_edges, memory_dim) - temporal change
                speaker_state_src,      # (num_edges, emb_dim) - speaker state for source
                speaker_state_tgt       # (num_edges, emb_dim) - speaker state for target
            )
            
            # Clamp to valid range and apply softmax scaling to preserve sparsity
            updated_edge_norm = torch.clamp(updated_weights.squeeze(-1), 0.0, 1.0)
            
            # Optional: Apply edge pruning (set low-weight edges to near-zero)
            # This helps maintain graph sparsity as per spec
            updated_edge_norm = torch.where(
                updated_edge_norm < self.edge_prune_threshold,
                torch.tensor(1e-6, device=updated_edge_norm.device),  # Near-zero, not removed
                updated_edge_norm
            )
            
        else:
            updated_edge_norm = edge_norm
        
        return updated_edge_norm

    def forward(self, U, qmask, umask, seq_lengths, compute_temporal_loss=True):
        if self.base_model == 'LSTM':
            emotions, hidden = self.lstm(U)

        elif self.base_model == 'GRU':
            emotions, hidden = self.gru(U)

        elif self.base_model == 'None':
            emotions = self.base_linear(U)

        # Update temporal memories
        node_memories, global_contexts = self._update_temporal_memories(emotions, qmask, seq_lengths)
        
        # Compute temporal consistency loss if needed
        temporal_loss = torch.tensor(0.0, device=emotions.device)
        if compute_temporal_loss:
            for t in range(1, node_memories.size(0)):
                temporal_loss += self._compute_temporal_consistency_loss(
                    node_memories[t], 
                    node_memories[t-1]
                )
            temporal_loss = temporal_loss / max(1, node_memories.size(0) - 1)

        # Graph construction (keeping initial graph structure as required)
        features, edge_index, edge_norm, edge_type, edge_index_lengths = batch_graphify(
            emotions, qmask, seq_lengths, self.window_past, self.window_future, 
            self.edge_type_mapping, self.att_model, self.no_cuda
        )
        
        # DYNAMIC GRAPH EVOLUTION: Apply temporal state changes to edge weights
        # Compute Δ_t (temporal state change) and update edge weights accordingly
        edge_norm_updated = self._update_dynamic_edges(
            edge_index, edge_norm, node_memories, emotions, seq_lengths, qmask
        )
        
        # Graph neural network forward pass with evolved graph structure (updated edge weights)
        log_prob = self.graph_net(features, edge_index, edge_norm_updated, edge_type, seq_lengths, umask, 
                                  self.nodal_attention, self.avec)

        if compute_temporal_loss:
            return log_prob, edge_index, edge_norm_updated, edge_type, edge_index_lengths, temporal_loss
        else:
            return log_prob, edge_index, edge_norm_updated, edge_type, edge_index_lengths


class CNNFeatureExtractor(nn.Module):
    def __init__(self, vocab_size, embedding_dim, output_size, filters, kernel_sizes, dropout):
        super(CNNFeatureExtractor, self).__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.convs = nn.ModuleList(
            [nn.Conv1d(in_channels=embedding_dim, out_channels=filters, kernel_size=K) for K in kernel_sizes])
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(len(kernel_sizes) * filters, output_size)
        self.feature_dim = output_size

    def init_pretrained_embeddings_from_numpy(self, pretrained_word_vectors):
        self.embedding.weight = nn.Parameter(torch.from_numpy(pretrained_word_vectors).float())
        # if is_static:
        self.embedding.weight.requires_grad = False

    def forward(self, x, umask):
        num_utt, batch, num_words = x.size()

        x = x.long()  # (num_utt, batch, num_words)
        x = x.view(-1, num_words)  # (num_utt, batch, num_words) -> (num_utt * batch, num_words)
        emb = self.embedding(x)  # (num_utt * batch, num_words) -> (num_utt * batch, num_words, 300)
        emb = emb.transpose(-2,
                            -1).contiguous()  # (num_utt * batch, num_words, 300)  -> (num_utt * batch, 300, num_words)

        convoluted = [F.relu(conv(emb)) for conv in self.convs]
        pooled = [F.max_pool1d(c, c.size(2)).squeeze() for c in convoluted]
        concated = torch.cat(pooled, 1)
        features = F.relu(self.fc(self.dropout(concated)))  # (num_utt * batch, 150) -> (num_utt * batch, 100)
        features = features.view(num_utt, batch, -1)  # (num_utt * batch, 100) -> (num_utt, batch, 100)
        mask = umask.unsqueeze(-1).float()  # (batch, num_utt) -> (batch, num_utt, 1)
        mask = mask.transpose(0, 1)  # (batch, num_utt, 1) -> (num_utt, batch, 1)
        mask = mask.repeat(1, 1, self.feature_dim)  # (num_utt, batch, 1) -> (num_utt, batch, 100)
        features = (features * mask)  # (num_utt, batch, 100) -> (num_utt, batch, 100)

        return features