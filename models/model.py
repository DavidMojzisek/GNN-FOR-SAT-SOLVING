import torch
import torch.nn as nn
from torch_geometric.nn import global_mean_pool
from torch_sparse import matmul
import math
import torch.nn.functional as F


class PrimalDualRNN(nn.Module):
    """
    Simplified RNN for Primal-Dual algorithm.
    No weight matrices - just residual connection with bias.
    Based on equations (5-6) from the paper.
    """

    def __init__(self, input_size, hidden_size, nonlinearity='tanh', bias=True):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        self.nonlinearity = nonlinearity

        if bias:
            self.bias_ih = nn.Parameter(torch.Tensor(hidden_size))
        else:
            self.register_parameter('bias_ih', None)

        self.reset_parameters()

        if nonlinearity == 'tanh':
            self.activation = torch.tanh
        elif nonlinearity == 'relu':
            self.activation = torch.relu
        else:
            raise ValueError(f"Unknown nonlinearity: {nonlinearity}")

    def reset_parameters(self):
        if self.bias:
            nn.init.zeros_(self.bias_ih)

    def forward(self, input, hx=None):
        """Primal-dual update: h_t = activation(input + bias + h_{t-1})"""
        if self.bias:
            h_t = self.activation(input + self.bias_ih + hx)
        else:
            h_t = self.activation(input + hx)
        return None, h_t.unsqueeze(0)


class BaseMessageLayer(nn.Module):
    """Base class for message passing layers supporting RNN, LSTM, and Primal-Dual updates."""

    def __init__(self, d_model, update_type='rnn', nonlinearity='tanh', bias=True):
        super().__init__()
        self.d_model = d_model
        self.update_type = update_type

        if update_type == 'lstm':
            self.updater = nn.LSTM(d_model, d_model)
        elif update_type == 'rnn':
            self.updater = nn.RNN(d_model, d_model)
        elif update_type == 'primal_dual':
            self.updater = PrimalDualRNN(d_model, d_model, nonlinearity=nonlinearity, bias=bias)
        else:
            raise ValueError(f"Unknown update_type '{update_type}'. Choose: 'rnn', 'lstm', or 'primal_dual'")


class EdgeFeatureMLP(nn.Module):
    """Separate MLPs for processing positive and negative literal edges."""

    def __init__(self, d_model):
        super().__init__()
        self.pos_mlp = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model)
        )
        self.neg_mlp = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model)
        )

    def forward(self, x, polarities):
        """
        Process edge features based on polarity.
        Args:
            x: [num_edges, d_model] - features of source nodes for each edge
            polarities: [num_edges] - whether each edge is positive/negative
        """
        result = torch.zeros_like(x)
        pos_mask = polarities > 0
        neg_mask = polarities < 0

        if pos_mask.any():
            result[pos_mask] = self.pos_mlp(x[pos_mask])
        if neg_mask.any():
            result[neg_mask] = self.neg_mlp(x[neg_mask])
        return result


class VarToClauseLayer(BaseMessageLayer):
    """Message passing from variables/literals to clauses (updates dual variables)."""

    def __init__(self, d_model, update_type='rnn', use_edge_features=False):
        super().__init__(d_model, update_type, nonlinearity='relu', bias=True)
        self.use_edge_features = use_edge_features
        if use_edge_features:
            self.edge_net = EdgeFeatureMLP(d_model)

    def forward(self, adj_t, x_v, hidden, edge_index=None, edge_attr=None):
        if self.use_edge_features and edge_index is not None and edge_attr is not None:
            source_features = x_v[edge_index[0]]
            processed_features = self.edge_net(source_features, edge_attr)
            msg = torch.zeros((adj_t.size(0), self.d_model), device=x_v.device)
            msg.index_add_(0, edge_index[1], processed_features)
        else:
            msg = matmul(adj_t, x_v)

        if self.updater.__class__.__name__ == 'RNN':
            hidden = hidden[0].unsqueeze(0)
            msg, new_hidden = self.updater(msg.unsqueeze(0), hidden)
            return [new_hidden[0].squeeze(0), None]

        elif self.updater.__class__.__name__ == 'LSTM':
            hidden = (hidden[0].unsqueeze(0), hidden[1].unsqueeze(0))
            msg, new_hidden = self.updater(msg.unsqueeze(0), hidden)
            return [new_hidden[0].squeeze(0), new_hidden[1].squeeze(0)]

        elif self.updater.__class__.__name__ == 'PrimalDualRNN':
            hidden = hidden[0].unsqueeze(0)
            msg, new_hidden = self.updater(msg.unsqueeze(0), hidden)
            return [new_hidden[0].squeeze(0), None]


class ClauseToVarLayer(BaseMessageLayer):
    """Message passing from clauses to variables/literals (updates primal variables)."""

    def __init__(self, d_model, update_type='rnn', use_edge_features=False):
        super().__init__(d_model, update_type, nonlinearity='tanh', bias=False)
        self.use_edge_features = use_edge_features
        if use_edge_features:
            self.edge_net = EdgeFeatureMLP(d_model)

    def forward(self, adj_t, x_c, hidden, v_batch, edge_index=None, edge_attr=None):
        if self.use_edge_features and edge_index is not None and edge_attr is not None:
            source_features = x_c[edge_index[1]]
            processed_features = self.edge_net(source_features, edge_attr)
            msg = torch.zeros((adj_t.size(1), self.d_model), device=x_c.device)
            msg.index_add_(0, edge_index[0], processed_features)
        else:
            msg = matmul(adj_t.t(), x_c)

        x_v = hidden[0]
        if self.updater.__class__.__name__ == 'RNN':
            hidden = x_v.unsqueeze(0)
            msg, new_hidden = self.updater(msg.unsqueeze(0), hidden)
            return [new_hidden[0].squeeze(0), None]

        elif self.updater.__class__.__name__ == 'LSTM':
            hidden = (x_v.unsqueeze(0), hidden[1].unsqueeze(0))
            msg, new_hidden = self.updater(msg.unsqueeze(0), hidden)
            return [new_hidden[0].squeeze(0), new_hidden[1].squeeze(0)]

        elif self.updater.__class__.__name__ == 'PrimalDualRNN':
            hidden = x_v.unsqueeze(0)
            msg, new_hidden = self.updater(msg.unsqueeze(0), hidden)
            return [new_hidden[0].squeeze(0), None]


class GNN_SAT(nn.Module):
    """
    Graph Neural Network for SAT solving.

    Supports three update types:
    - 'rnn': Standard PyTorch RNN with weight matrices
    - 'lstm': Standard PyTorch LSTM (uses hidden state h_t for embeddings, not cell output)
    - 'primal_dual': Simplified gradient-based update (no weights, VCG only)

    Args:
        d_model: Embedding dimension
        update_type: 'rnn' | 'lstm' | 'primal_dual'
        graph_type: 'var' (VCG) | 'lit' (LCG) - primal_dual only supports 'var'
        collect_embeddings: Collect all iteration embeddings during validation
        use_edge_features: Use separate MLPs for positive/negative edges
        use_clause_voting: Use clause embeddings for SAT prediction (vs variable embeddings)
        assignment_CE_loss: Use cross-entropy for assignment (vs binary cross-entropy)
    """

    def __init__(self, d_model, update_type='rnn', graph_type='var', collect_embeddings=False,
                 use_edge_features=False, use_clause_voting=False, assignment_CE_loss=False):
        super().__init__()
        self.d_model = d_model
        self.update_type = update_type
        self.graph_type = graph_type
        self.collect_embeddings = collect_embeddings
        self.use_edge_features = use_edge_features
        self.use_clause_voting = use_clause_voting
        self.assignment_CE_loss = assignment_CE_loss

        if update_type == 'primal_dual' and graph_type != 'var':
            raise ValueError("Primal-dual GNN only supports graph_type='var' (VCG)")

        if graph_type == 'var':
            self.unk_to_clause = VarToClauseLayer(d_model, update_type, use_edge_features)
            self.clause_to_unk = ClauseToVarLayer(d_model, update_type, use_edge_features)
            self.get_x = lambda data: data.x_v
            self.get_batch = lambda data: data.x_v_batch
            self.get_adj = lambda data: data.adj_t_var
            self.get_edge_info = lambda data: (
                (data.edge_index_var, data.polarities)
                if (use_edge_features and hasattr(data, 'edge_index_var'))
                else (None, None)
            )
        elif graph_type == 'lit':
            self.unk_to_clause = VarToClauseLayer(d_model, update_type, use_edge_features)
            self.clause_to_unk = ClauseToVarLayer(d_model, update_type, use_edge_features)
            self.get_x = lambda data: data.x_l
            self.get_batch = lambda data: data.x_l_batch
            self.get_adj = lambda data: data.adj_t_lit
            self.get_edge_info = lambda data: (
                (data.edge_index_lit, data.polarities)
                if (use_edge_features and hasattr(data, 'edge_index_lit'))
                else (None, None)
            )
        else:
            raise ValueError(f"Unknown graph_type '{graph_type}'. Choose 'var' or 'lit'")

        if use_edge_features:
            self.unk_to_clause.edge_net.pos_mlp = self.clause_to_unk.edge_net.pos_mlp
            self.unk_to_clause.edge_net.neg_mlp = self.clause_to_unk.edge_net.neg_mlp

        if self.assignment_CE_loss:
            self.assignment_output = nn.Linear(d_model, 2, bias=False)
        else:
            self.assignment_output = nn.Sequential(nn.Linear(d_model, 1), nn.Sigmoid())

        self.sat_output = nn.Linear(d_model, 1, bias=False)

        self.init_embeddings = nn.Linear(1, d_model)
        self.init_ts = torch.ones(1)
        self.init_ts.requires_grad = False
        self.L_init = nn.Linear(1, d_model)
        self.C_init = nn.Linear(1, d_model)
        self.iter_emb = nn.Linear(1, d_model)

    def random_init_embeddings(self, data, device):
        """Initialize random embeddings for variables/literals and clauses."""
        n_unks, n_clauses = self.get_x(data).size(0), data.x_c.size(0)
        init_ts = self.init_ts.to(device)

        x_unk = torch.rand((n_unks, self.d_model), requires_grad=False).to(device)
        x_c = torch.rand((n_clauses, self.d_model), requires_grad=False).to(device)

        x_unk_h = torch.zeros(x_unk.shape).to(device)
        x_c_h = torch.zeros(x_c.shape).to(device)

        unk_hidden = (x_unk, x_unk_h)
        c_hidden = (x_c, x_c_h)
        return unk_hidden, c_hidden

    def forward(self, data, num_iters):
        device = self.get_x(data).device
        unk_hidden, c_hidden = self.random_init_embeddings(data, device)

        collect_now = self.collect_embeddings and (not self.training)

        all_unk_votes = []
        all_unk_embeds = []
        all_c_embeds = []
        batch = self.get_batch(data)
        adj = self.get_adj(data)
        edge_index, edge_attr = self.get_edge_info(data)

        for current_iter in range(num_iters):
            c_hidden = self.unk_to_clause(adj, unk_hidden[0], c_hidden,
                                          edge_index=edge_index, edge_attr=edge_attr)

            unk_hidden = self.clause_to_unk(adj, c_hidden[0], unk_hidden, batch,
                                            edge_index=edge_index, edge_attr=edge_attr)

            unk_hidden = (unk_hidden[0] / torch.norm(unk_hidden[0], dim=1, keepdim=True), unk_hidden[1])

            if self.update_type != 'primal_dual':
                c_hidden = (c_hidden[0] / torch.norm(c_hidden[0], dim=1, keepdim=True), c_hidden[1])

            if collect_now:
                all_unk_embeds.append(unk_hidden[0])
                all_c_embeds.append(c_hidden[0])

            if self.assignment_CE_loss:
                votes = self.assignment_output(unk_hidden[0])
            else:
                votes = self.assignment_output(unk_hidden[0])
            all_unk_votes.append(votes)

        if self.use_clause_voting:
            sat_votes = self.sat_output(c_hidden[0])
            sat_vote_reduced = global_mean_pool(sat_votes, data.x_c_batch)
        else:
            sat_votes = self.sat_output(unk_hidden[0])
            sat_vote_reduced = global_mean_pool(sat_votes, batch)

        if self.assignment_CE_loss:
            assignment_votes = self.assignment_output(unk_hidden[0])
        else:
            assignment_votes = self.assignment_output(unk_hidden[0])

        result = {
            'vote_reduced': sat_vote_reduced,
            'final_votes': assignment_votes,
            'all_votes': all_unk_votes,
            'final_embeds': unk_hidden[0],
        }

        if collect_now:
            result['all_unk_embeds'] = all_unk_embeds
            result['all_c_embeds'] = all_c_embeds

        return result
