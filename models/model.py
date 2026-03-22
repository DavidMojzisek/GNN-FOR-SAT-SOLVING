import torch
import torch.nn as nn
from torch_geometric.nn import global_mean_pool
from torch_sparse import matmul


class PrimalDualRNN(nn.Module):
    """
    Simplified RNN for Primal-Dual algorithm.
    No weight matrices — residual connection with learned bias.
    Variable update: x^(k+1) = Norm(x^(k) + MLP(M_{C->V}(lambda^(k))))
    Clause update:   lambda^(k+1) = ReLU(lambda^(k) + MLP(M_{V->C}(x^(k+1))) + b)
    """

    def __init__(self, input_size, hidden_size, nonlinearity='tanh', bias=True):
        super().__init__()
        self.hidden_size = hidden_size
        if bias:
            self.bias_ih = nn.Parameter(torch.zeros(hidden_size))
        else:
            self.register_parameter('bias_ih', None)
        self.activation = torch.tanh if nonlinearity == 'tanh' else torch.relu

    def forward(self, input, hx=None):
        """h_t = activation(input + bias + h_{t-1})"""
        bias = self.bias_ih if self.bias_ih is not None else 0
        h_t = self.activation(input + bias + hx)
        return None, h_t.unsqueeze(0)


class BaseMessageLayer(nn.Module):
    """Base class supporting RNN, LSTM, and Primal-Dual update rules."""

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
    """Separate MLPs for positive and negative literal edges (polarity-specific processing)."""

    def __init__(self, d_model):
        super().__init__()
        self.pos_mlp = nn.Sequential(nn.Linear(d_model, d_model), nn.ReLU(), nn.Linear(d_model, d_model))
        self.neg_mlp = nn.Sequential(nn.Linear(d_model, d_model), nn.ReLU(), nn.Linear(d_model, d_model))

    def forward(self, x, polarities):
        # Run both MLPs on all edges and select by polarity mask.
        # torch.where is fully differentiable and avoids masked scatter awkwardness.
        pos_out = self.pos_mlp(x)
        neg_out = self.neg_mlp(x)
        mask = (polarities > 0).unsqueeze(1).expand_as(pos_out)
        return torch.where(mask, pos_out, neg_out)


class VarToClauseLayer(BaseMessageLayer):
    """Message passing from variables/literals to clauses (dual variable update)."""

    def __init__(self, d_model, update_type='rnn', use_edge_features=False,
                 use_polarity_scalar=False):
        super().__init__(d_model, update_type, nonlinearity='relu', bias=True)
        self.use_edge_features = use_edge_features
        self.use_polarity_scalar = use_polarity_scalar
        if use_edge_features:
            self.edge_net = EdgeFeatureMLP(d_model)
        elif use_polarity_scalar:
            # Lightweight polarity encoding: concat scalar +1/-1 to source embedding,
            # project back to d_model with a single shared linear layer.
            self.polarity_proj = nn.Linear(d_model + 1, d_model, bias=True)

    def forward(self, adj_t, x_v, hidden, edge_index=None, edge_attr=None):
        if self.use_edge_features and edge_index is not None and edge_attr is not None:
            source_features = x_v[edge_index[0]]
            processed_features = self.edge_net(source_features, edge_attr)
            msg = torch.zeros((adj_t.size(0), self.d_model), device=x_v.device)
            msg.index_add_(0, edge_index[1], processed_features)
        elif self.use_polarity_scalar and edge_index is not None and edge_attr is not None:
            source_features = x_v[edge_index[0]]
            pol = edge_attr.float().unsqueeze(1)           # [E, 1]  values: +1 or -1
            processed_features = self.polarity_proj(torch.cat([source_features, pol], dim=1))
            msg = torch.zeros((adj_t.size(0), self.d_model), device=x_v.device)
            msg.index_add_(0, edge_index[1], processed_features)
        else:
            msg = matmul(adj_t, x_v)

        if self.update_type == 'rnn':
            h = hidden[0].unsqueeze(0)
            _, new_h = self.updater(msg.unsqueeze(0), h)
            return [new_h[0].squeeze(0), None]
        elif self.update_type == 'lstm':
            h = (hidden[0].unsqueeze(0), hidden[1].unsqueeze(0))
            _, (new_h, new_c) = self.updater(msg.unsqueeze(0), h)
            return [new_h.squeeze(0), new_c.squeeze(0)]
        else:  # primal_dual
            h = hidden[0].unsqueeze(0)
            _, new_h = self.updater(msg.unsqueeze(0), h)
            return [new_h[0].squeeze(0), None]


class ClauseToVarLayer(BaseMessageLayer):
    """Message passing from clauses to variables (VCG / var graph only)."""

    def __init__(self, d_model, update_type='rnn', use_edge_features=False,
                 use_polarity_scalar=False):
        super().__init__(d_model, update_type, nonlinearity='tanh', bias=False)
        self.use_edge_features = use_edge_features
        self.use_polarity_scalar = use_polarity_scalar
        if use_edge_features:
            self.edge_net = EdgeFeatureMLP(d_model)
        elif use_polarity_scalar:
            self.polarity_proj = nn.Linear(d_model + 1, d_model, bias=True)

    def forward(self, adj_t, x_c, hidden, v_batch, edge_index=None, edge_attr=None):
        if self.use_edge_features and edge_index is not None and edge_attr is not None:
            source_features = x_c[edge_index[1]]
            processed_features = self.edge_net(source_features, edge_attr)
            msg = torch.zeros((adj_t.size(1), self.d_model), device=x_c.device)
            msg.index_add_(0, edge_index[0], processed_features)
        elif self.use_polarity_scalar and edge_index is not None and edge_attr is not None:
            source_features = x_c[edge_index[1]]
            pol = edge_attr.float().unsqueeze(1)           # [E, 1]  values: +1 or -1
            processed_features = self.polarity_proj(torch.cat([source_features, pol], dim=1))
            msg = torch.zeros((adj_t.size(1), self.d_model), device=x_c.device)
            msg.index_add_(0, edge_index[0], processed_features)
        else:
            msg = matmul(adj_t.t(), x_c)

        x_v = hidden[0]
        if self.update_type == 'rnn':
            h = x_v.unsqueeze(0)
            _, new_h = self.updater(msg.unsqueeze(0), h)
            return [new_h[0].squeeze(0), None]
        elif self.update_type == 'lstm':
            h = (x_v.unsqueeze(0), hidden[1].unsqueeze(0))
            _, (new_h, new_c) = self.updater(msg.unsqueeze(0), h)
            return [new_h.squeeze(0), new_c.squeeze(0)]
        else:  # primal_dual
            h = x_v.unsqueeze(0)
            _, new_h = self.updater(msg.unsqueeze(0), h)
            return [new_h[0].squeeze(0), None]


class ClauseToLitLayer(nn.Module):
    """Message passing from clauses to literals (LCG / lit graph only).

    Each literal receives the aggregated clause message concatenated with its
    complement literal's current embedding (the 'flip' operation). This allows
    positive and negative literals of the same variable to communicate, which is
    the core mechanism that enables consistent truth-assignment discovery
    (as in NeuroSAT / the original LCG formulation).

    Input to the updater is 2*d_model because of the [clause_msg | flip] cat.
    Only supports update_type in {'rnn', 'lstm'} — primal_dual requires var graph.
    """

    def __init__(self, d_model, update_type='rnn'):
        super().__init__()
        self.d_model = d_model
        self.update_type = update_type
        # input_size = 2*d_model because we cat [clause_msg | flipped_lit_embed]
        if update_type == 'lstm':
            self.updater = nn.LSTM(input_size=2 * d_model, hidden_size=d_model)
        elif update_type == 'rnn':
            self.updater = nn.RNN(input_size=2 * d_model, hidden_size=d_model)
        else:
            raise ValueError(
                f"ClauseToLitLayer does not support update_type='{update_type}'. "
                "Use 'rnn' or 'lstm' for lit graph."
            )

    def _flip_literals(self, x_l, l_batch):
        """Swap positive and negative literal embeddings within each formula.

        Literals are stored as [pos_1, …, pos_n, neg_1, …, neg_n] per formula.
        After flipping each literal gets its complement's embedding as context.
        """
        counts = torch.bincount(l_batch)
        starts = torch.cat([
            torch.tensor([0], device=x_l.device),
            torch.cumsum(counts[:-1], dim=0),
        ])
        flipped = []
        for count, start in zip(counts, starts):
            n_vars = count // 2
            pos = x_l[start: start + n_vars]
            neg = x_l[start + n_vars: start + 2 * n_vars]
            flipped.append(torch.cat([neg, pos]))
        return torch.cat(flipped)

    def forward(self, adj_t, x_c, hidden, l_batch, **kwargs):
        msg = matmul(adj_t.t(), x_c)
        x_l = hidden[0]
        flipped = self._flip_literals(x_l, l_batch)
        inp = torch.cat([msg, flipped], dim=-1).unsqueeze(0)
        if self.update_type == 'rnn':
            h = x_l.unsqueeze(0)
            _, new_h = self.updater(inp, h)
            return [new_h[0].squeeze(0), None]
        else:  # lstm
            h = (x_l.unsqueeze(0), hidden[1].unsqueeze(0))
            _, (new_h, new_c) = self.updater(inp, h)
            return [new_h.squeeze(0), new_c.squeeze(0)]


class GNN_SAT(nn.Module):
    """
    Graph Neural Network for SAT solving.

    Supports three update rules:
      - 'rnn': Standard RNN with weight matrices (VCG or LCG)
      - 'lstm': LSTM, uses hidden state h_t as embedding (VCG or LCG)
      - 'primal_dual': Gradient-inspired update without weight matrices (VCG only)

    Graph representations:
      - 'var' (VCG): bipartite variable-clause graph; edge polarity ∈ {+1, -1}
      - 'lit' (LCG): bipartite literal-clause graph; positive/negative literals are separate nodes
                     clause→lit uses ClauseToLitLayer which concatenates the complement literal
                     embedding (flip) to the clause message — the core NeuroSAT mechanism

    normalize_embeddings:
      - None (default): auto — True for primal_dual (required by the unit-sphere
                        constraint in the primal-dual derivation), False for rnn/lstm
      - True:  apply L2 normalisation to variable embeddings after every iteration
               (and to clause embeddings when update_type != 'primal_dual')
      - False: no normalisation (original NeuroSAT behaviour for rnn/lstm)

    output_bias:
      - True (default): output linear heads have a bias term
      - False: no bias in output heads

    VCG edge MLP options (only when use_edge_features=True, var graph only):
      - separate_direction_mlps=False (default): pos/neg MLPs shared between V→C and C→V
      - separate_direction_mlps=True: each direction has independent pos/neg MLPs

    use_polarity_scalar (VCG only, mutually exclusive with use_edge_features):
      - False (default): no polarity encoding when use_edge_features=False
      - True: concat a +1/-1 scalar to each source embedding before aggregation,
              project back with a single Linear(d+1 → d). Encodes polarity at near-zero
              extra cost compared to the dual-MLP approach of use_edge_features=True.

    collect_all_votes:
      - False (default): only final iteration votes/embeddings returned (saves memory)
      - True: votes and embeddings for every iteration in 'all_votes' / 'all_embeds'
    """

    def __init__(self, d_model, update_type='rnn', graph_type='var',
                 use_edge_features=False, use_polarity_scalar=False,
                 use_clause_voting=False,
                 separate_direction_mlps=False, collect_all_votes=False,
                 normalize_embeddings=None, output_bias=True):
        super().__init__()
        self.d_model = d_model
        self.update_type = update_type
        self.graph_type = graph_type
        self.use_clause_voting = use_clause_voting
        self.collect_all_votes = collect_all_votes

        if update_type == 'primal_dual' and graph_type != 'var':
            raise ValueError("update_type='primal_dual' requires graph_type='var'")
        if use_edge_features and graph_type == 'lit':
            import logging as _logging
            _logging.warning(
                "use_edge_features=True is incompatible with graph_type='lit' — "
                "polarity is already encoded in the LCG flip mechanism. "
                "Automatically setting use_edge_features=False."
            )
            use_edge_features = False
        if use_polarity_scalar and graph_type == 'lit':
            import logging as _logging
            _logging.warning(
                "use_polarity_scalar=True is incompatible with graph_type='lit' — "
                "polarity is already encoded in the LCG flip mechanism. "
                "Automatically setting use_polarity_scalar=False."
            )
            use_polarity_scalar = False
        if use_edge_features and use_polarity_scalar:
            import logging as _logging
            _logging.warning(
                "use_edge_features=True takes precedence over use_polarity_scalar=True. "
                "Setting use_polarity_scalar=False."
            )
            use_polarity_scalar = False
        if graph_type == 'var' and not use_edge_features and not use_polarity_scalar:
            import logging as _logging
            _logging.warning(
                "VCG (graph_type='var') with use_edge_features=False and "
                "use_polarity_scalar=False: polarity information is NOT encoded. "
                "The model cannot distinguish positive from negative literal edges. "
                "Consider setting use_polarity_scalar=True (cheap) or "
                "use_edge_features=True (full dual-MLP)."
            )

        # Resolve normalize_embeddings: None → auto based on update_type
        if normalize_embeddings is None:
            # primal_dual requires normalization (unit-sphere constraint in the primal-dual derivation)
            self.normalize_embeddings = (update_type == 'primal_dual')
        else:
            self.normalize_embeddings = normalize_embeddings

        if graph_type == 'var':
            self.unk_to_clause = VarToClauseLayer(d_model, update_type, use_edge_features,
                                                   use_polarity_scalar)
            self.clause_to_unk = ClauseToVarLayer(d_model, update_type, use_edge_features,
                                                   use_polarity_scalar)
            # Share or separate edge MLPs between V→C and C→V directions
            if use_edge_features and not separate_direction_mlps:
                self.unk_to_clause.edge_net.pos_mlp = self.clause_to_unk.edge_net.pos_mlp
                self.unk_to_clause.edge_net.neg_mlp = self.clause_to_unk.edge_net.neg_mlp
            self.get_x = lambda data: data.x_v
            self.get_batch = lambda data: data.x_v_batch
            self.get_adj = lambda data: data.adj_t_var
            # Provide edge_index + polarities whenever either polarity encoding is active
            self.get_edge_info = lambda data: (
                (data.edge_index_var, data.polarities)
                if (use_edge_features or use_polarity_scalar) else (None, None)
            )
        else:  # lit
            # lit→clause: same aggregation as var graph (VarToClauseLayer works fine)
            self.unk_to_clause = VarToClauseLayer(d_model, update_type, use_edge_features=False)
            # clause→lit: requires flip operation to exchange complement literal embeddings
            self.clause_to_unk = ClauseToLitLayer(d_model, update_type)
            self.get_x = lambda data: data.x_l
            self.get_batch = lambda data: data.x_l_batch
            self.get_adj = lambda data: data.adj_t_lit
            self.get_edge_info = lambda data: (None, None)

        # Output heads
        self.assignment_output = nn.Linear(d_model, 2, bias=output_bias)
        self.sat_output = nn.Linear(d_model, 1, bias=output_bias)

    def _init_hidden(self, data, device):
        """Random initialisation of variable/literal and clause embeddings.

        For LSTM: both hidden state and cell state are initialised (zeros for cell).
        For RNN / primal_dual: cell state slot is None (never accessed by those paths).
        """
        n_unks = self.get_x(data).size(0)
        n_clauses = data.x_c.size(0)
        x_unk = torch.rand((n_unks, self.d_model), device=device)
        x_c = torch.rand((n_clauses, self.d_model), device=device)
        if self.update_type == 'lstm':
            return (x_unk, torch.zeros_like(x_unk)), (x_c, torch.zeros_like(x_c))
        # RNN / primal_dual: second slot unused, keep as None to avoid allocation
        return (x_unk, None), (x_c, None)

    def forward(self, data, num_iters):
        device = self.get_x(data).device
        unk_hidden, c_hidden = self._init_hidden(data, device)

        batch = self.get_batch(data)
        adj = self.get_adj(data)
        edge_index, edge_attr = self.get_edge_info(data)

        all_votes = [] if self.collect_all_votes else None
        all_embeds = [] if self.collect_all_votes else None

        for _ in range(num_iters):
            c_hidden = self.unk_to_clause(adj, unk_hidden[0], c_hidden,
                                          edge_index=edge_index, edge_attr=edge_attr)
            unk_hidden = self.clause_to_unk(adj, c_hidden[0], unk_hidden, batch,
                                            edge_index=edge_index, edge_attr=edge_attr)

            if self.normalize_embeddings:
                # L2 normalisation: enforces the unit-sphere constraint required by
                # primal_dual; empirically beneficial for rnn/lstm as well.
                unk_hidden = (
                    unk_hidden[0] / torch.norm(unk_hidden[0], dim=1, keepdim=True).clamp(min=1e-8),
                    unk_hidden[1],
                )
                # Clause embeddings: normalise for rnn/lstm but NOT for primal_dual —
                # dual variables are constrained to R^+ via ReLU, not the unit sphere.
                if self.update_type != 'primal_dual':
                    c_hidden = (
                        c_hidden[0] / torch.norm(c_hidden[0], dim=1, keepdim=True).clamp(min=1e-8),
                        c_hidden[1],
                    )

            if self.collect_all_votes:
                all_votes.append(self.assignment_output(unk_hidden[0]))
                all_embeds.append(unk_hidden[0])

        final_votes = self.assignment_output(unk_hidden[0])

        if self.use_clause_voting:
            sat_logits = global_mean_pool(self.sat_output(c_hidden[0]), data.x_c_batch)
        else:
            sat_logits = global_mean_pool(self.sat_output(unk_hidden[0]), batch)

        return {
            'vote_reduced': sat_logits,    # [B, 1] graph-level SAT logit
            'final_votes': final_votes,    # [N_vars, 2] assignment logits (last iter)
            # all_votes / all_embeds are lists of length num_iters when collect_all_votes=True,
            # otherwise None (not collected to save memory).
            'all_votes': all_votes,        # list of [N_vars, 2] per iter, or None
            'all_embeds': all_embeds,      # list of [N_vars, d] per iter, or None
            'final_embeds': unk_hidden[0], # [N_vars, d] final embeddings
            'var_batch': batch,            # batch index for each var/lit node
        }
