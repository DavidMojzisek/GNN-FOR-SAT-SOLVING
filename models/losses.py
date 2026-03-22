import torch
import torch.nn.functional as F
from torch_sparse import matmul
from pysat.formula import WCNF
from pysat.examples.rc2 import RC2
import random


# ---------------------------------------------------------------------------
# SAT/UNSAT classification loss
# ---------------------------------------------------------------------------

def compute_sat_loss(outputs, batch):
    """Binary cross-entropy for SAT/UNSAT classification."""
    return F.binary_cross_entropy_with_logits(
        outputs['vote_reduced'].squeeze(), batch.y
    )


# ---------------------------------------------------------------------------
# Assignment supervision losses
# ---------------------------------------------------------------------------

def _get_var_votes(outputs, batch, graph_type):
    """Extract per-variable 2-class logits from model outputs."""
    if graph_type == 'lit':
        n_graphs = int(batch.x_l_batch.max().item()) + 1
        grouped = [outputs['final_votes'][batch.x_l_batch == i] for i in range(n_graphs)]
        return torch.cat([v[:v.size(0) // 2] for v in grouped])
    return outputs['final_votes']


def compute_assignment_CE_loss(outputs, batch, graph_type):
    """CE loss against fixed ground-truth assignments from SAT solver."""
    votes = _get_var_votes(outputs, batch, graph_type)
    targets = ((batch.assignment + 1) / 2).long().to(votes.device)
    return F.cross_entropy(votes, targets)


# ---------------------------------------------------------------------------
# Closest assignment — RC2 (MaxSAT, exact but slow)
# ---------------------------------------------------------------------------

def _find_closest_assignment_rc2(model_probs, clauses, is_sat, device):
    """RC2 MaxSAT solver to find the assignment closest to model predictions."""
    target_probs = model_probs.detach().cpu().numpy()
    num_vars = len(target_probs)
    wcnf = WCNF()

    if is_sat:
        for clause in clauses:
            wcnf.append(clause)                        # hard constraint
    else:
        high_weight = num_vars + 1
        for clause in clauses:
            wcnf.append(clause, weight=high_weight)   # soft, high weight

    for i, prob in enumerate(target_probs):
        var_idx = i + 1
        wcnf.append([var_idx if prob >= 0.5 else -var_idx], weight=1)

    with RC2(wcnf) as solver:
        solution = solver.compute()

    if solution is None:
        return torch.zeros(num_vars, dtype=torch.long, device=device)

    assignment = torch.zeros(num_vars, dtype=torch.long, device=device)
    for lit in solution:
        idx = abs(lit) - 1
        if idx < num_vars:
            assignment[idx] = 1 if lit > 0 else 0
    return assignment


def compute_closest_assignment_CE_loss(outputs, batch, graph_type):
    """CE loss against RC2-found closest satisfying assignment."""
    device = batch.y.device
    votes = _get_var_votes(outputs, batch, graph_type)
    probs = votes.softmax(dim=1)[:, 1]

    var_cumsum = torch.cumsum(batch.num_variables, dim=0)
    targets = []
    for i, clauses in enumerate(batch.clauses):
        start = 0 if i == 0 else var_cumsum[i - 1].item()
        end = var_cumsum[i].item()
        formula_probs = probs[start:end]
        is_sat = batch.y[i].item() == 1
        targets.append(_find_closest_assignment_rc2(formula_probs, clauses, is_sat, device))

    return F.cross_entropy(votes, torch.cat(targets))


# ---------------------------------------------------------------------------
# Closest assignment — WalkSAT (online, fast local search)
# ---------------------------------------------------------------------------

def _run_walksat(binary_assignment, clauses, num_steps, noise_p=0.4, seed=None):
    """
    WalkSAT local search starting from binary_assignment (list of 0/1).
    Each step: pick a random unsatisfied clause, then either
      - with prob noise_p: flip a random literal in it (noise move)
      - with prob 1-noise_p: flip the literal that maximises net satisfied clauses
    Returns: improved binary assignment after num_steps steps.

    Uses a local Random instance (seed) so it does not affect the global random state.

    Optimised with precomputed per-variable occurrence lists and incremental
    satisfaction tracking — only checks clauses adjacent to the flipped variable.
    """
    rng = random.Random(seed)
    a = list(binary_assignment)
    n_vars = len(a)

    # Precompute: pos_occ[v] / neg_occ[v] = clause indices where var v appears positive/negative
    pos_occ = [[] for _ in range(n_vars)]
    neg_occ = [[] for _ in range(n_vars)]
    for ci, clause in enumerate(clauses):
        for lit in clause:
            vi = abs(lit) - 1
            (pos_occ[vi] if lit > 0 else neg_occ[vi]).append(ci)

    def is_clause_sat(ci):
        return any(
            (lit > 0 and a[abs(lit) - 1] == 1) or (lit < 0 and a[abs(lit) - 1] == 0)
            for lit in clauses[ci]
        )

    # Initialise satisfaction state
    sat = [is_clause_sat(ci) for ci in range(len(clauses))]
    unsat = set(i for i, s in enumerate(sat) if not s)

    for _ in range(num_steps):
        if not unsat:
            break

        ci = rng.choice(list(unsat))
        vars_in_clause = [abs(lit) - 1 for lit in clauses[ci]]

        if rng.random() < noise_p:
            flip_var = rng.choice(vars_in_clause)
        else:
            best_gain, best_vars = -len(clauses) - 1, []
            for var in vars_in_clause:
                new_val = 1 - a[var]
                # Clauses var would newly satisfy after flipping
                newly_sat_occ = pos_occ[var] if new_val == 1 else neg_occ[var]
                # Clauses that currently rely on var for satisfaction
                losing_sat_occ = pos_occ[var] if a[var] == 1 else neg_occ[var]

                gain = sum(1 for c2 in newly_sat_occ if not sat[c2])

                # Temporarily flip to check if losing clauses still have other supporters
                a[var] = new_val
                gain -= sum(1 for c2 in losing_sat_occ if sat[c2] and not is_clause_sat(c2))
                a[var] = 1 - new_val  # restore

                if gain > best_gain:
                    best_gain, best_vars = gain, [var]
                elif gain == best_gain:
                    best_vars.append(var)
            flip_var = rng.choice(best_vars)

        # Apply flip and update satisfaction tracking incrementally
        var = flip_var
        a[var] ^= 1

        # Clauses var now satisfies (may transition unsat → sat)
        now_satisfies = pos_occ[var] if a[var] == 1 else neg_occ[var]
        # Clauses var no longer satisfies (may transition sat → unsat)
        no_longer = pos_occ[var] if a[var] == 0 else neg_occ[var]

        for c2 in now_satisfies:
            if not sat[c2]:
                sat[c2] = True
                unsat.discard(c2)
        for c2 in no_longer:
            if sat[c2] and not is_clause_sat(c2):
                sat[c2] = False
                unsat.add(c2)

    return a


def _run_greedy(initial_probs, binary_assignment, clauses, num_steps, seed=None):
    """
    Greedy heuristic: each step flip the variable that maximises
    (satisfied clauses gained) - 0.1 * (Hamming change from initial GNN prediction).
    Stops early if no improving flip exists.

    Uses a local Random instance (seed) so it does not affect the global random state.
    """
    rng = random.Random(seed)
    assignment = list(binary_assignment)
    initial_binary = [1 if p >= 0.5 else 0 for p in initial_probs]

    def num_satisfied():
        return sum(any((lit > 0 and assignment[abs(lit) - 1] == 1) or
                       (lit < 0 and assignment[abs(lit) - 1] == 0) for lit in clause)
                   for clause in clauses)

    for _ in range(num_steps):
        base_score = num_satisfied()
        best_gain, best_var = 0.0, -1
        # Randomize evaluation order to avoid always preferring low-index vars
        for var in rng.sample(range(len(assignment)), len(assignment)):
            assignment[var] = 1 - assignment[var]
            new_score = num_satisfied()
            assignment[var] = 1 - assignment[var]
            gain = (new_score - base_score) - 0.1 * abs(assignment[var] - initial_binary[var])
            if gain > best_gain:
                best_gain, best_var = gain, var
        if best_var == -1:
            break
        assignment[best_var] = 1 - assignment[best_var]

    return assignment


def compute_walksat_assignment_CE_loss(outputs, batch, graph_type, num_steps, method='walksat'):
    """
    Online local-search CE loss. For each formula in the batch:
      1. Get soft GNN prediction → initial binary assignment (argmax)
      2. Run WalkSAT or Greedy local search for num_steps
      3. Use the result as supervision target for CE loss
    """
    device = batch.y.device
    votes = _get_var_votes(outputs, batch, graph_type)
    probs = votes.softmax(dim=1)[:, 1]         # prob of True

    var_cumsum = torch.cumsum(batch.num_variables, dim=0)
    targets = []

    for i, clauses in enumerate(batch.clauses):
        start = 0 if i == 0 else var_cumsum[i - 1].item()
        end = var_cumsum[i].item()

        formula_probs = probs[start:end].detach().cpu().tolist()
        binary_init = [1 if p >= 0.5 else 0 for p in formula_probs]

        if method == 'walksat':
            result = _run_walksat(binary_init, clauses, num_steps, seed=i)
        elif method == 'greedy':
            result = _run_greedy(formula_probs, binary_init, clauses, num_steps, seed=i)
        else:
            raise ValueError(f"Unknown local search method: {method}. Choose 'walksat' or 'greedy'")

        targets.append(torch.tensor(result, dtype=torch.long, device=device))

    return F.cross_entropy(votes, torch.cat(targets))


# ---------------------------------------------------------------------------
# Unsupervised losses (clause satisfaction probability)
# ---------------------------------------------------------------------------

def _compute_clause_satisfaction_probs(outputs, batch, graph_type):
    """
    Compute V_c = 1 - ∏_{i∈c+}(1-p_i) ∏_{i∈c-} p_i for all clauses.
    Uses log-space for numerical stability.

    Uses edge_index_var (variable indices 0..n_vars-1) for clause-variable connectivity
    in all cases. For LCG, variable predictions are extracted from the positive-literal
    embeddings (first n_vars per formula) before the connectivity lookup.

    Returns (clause_probs, c_batch).
    """
    device = outputs['final_votes'].device
    epsilon = 1e-10

    # Variable probabilities (prob of True) — works for both VCG and LCG
    votes = _get_var_votes(outputs, batch, graph_type)
    var_pred = votes.softmax(dim=1)[:, 1]

    # Always use variable-clause edges (not literal-clause) for clause satisfaction
    edge_index = batch.edge_index_var
    polarities = batch.polarities
    c_size = batch.x_c.size(0)
    c_batch = batch.x_c_batch

    log_complement = torch.where(
        polarities > 0,
        torch.log(1 - var_pred[edge_index[0]] + epsilon),
        torch.log(var_pred[edge_index[0]] + epsilon)
    )

    clause_log_prod = torch.zeros(c_size, device=device)
    clause_log_prod.index_add_(0, edge_index[1], log_complement)

    clause_probs = 1 - torch.exp(clause_log_prod).clamp(max=1.0 - epsilon)
    return clause_probs, c_batch


def compute_unsupervised_loss_linear(outputs, batch, graph_type):
    """L_lin = -∑_c V_c  (weak gradients near V_c=1)."""
    clause_probs, c_batch = _compute_clause_satisfaction_probs(outputs, batch, graph_type)
    formula_scores = torch.zeros(batch.num_graphs, device=clause_probs.device)
    formula_scores.index_add_(0, c_batch, clause_probs)
    return -formula_scores.mean()


def compute_unsupervised_loss_log(outputs, batch, graph_type):
    """L_log = -∑_c log(V_c)  (amplifies gradient for nearly-unsatisfied clauses)."""
    epsilon = 1e-10
    clause_probs, c_batch = _compute_clause_satisfaction_probs(outputs, batch, graph_type)
    log_probs = torch.log(clause_probs + epsilon)
    formula_log_scores = torch.zeros(batch.num_graphs, device=clause_probs.device)
    formula_log_scores.index_add_(0, c_batch, log_probs)
    return -formula_log_scores.mean()


def compute_unsupervised_loss_quad(outputs, batch, graph_type):
    """L_quad = ∑_c (1 - V_c)²  (stronger gradients than linear, bounded penalty)."""
    clause_probs, c_batch = _compute_clause_satisfaction_probs(outputs, batch, graph_type)
    penalties = (1 - clause_probs) ** 2
    formula_penalties = torch.zeros(batch.num_graphs, device=clause_probs.device)
    formula_penalties.index_add_(0, c_batch, penalties)
    return formula_penalties.mean()
