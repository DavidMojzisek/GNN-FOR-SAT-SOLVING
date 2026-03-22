import torch
import numpy as np


def _assignment_from_votes(final_votes):
    """Convert 2-class logits to binary assignment ∈ {0, 1}."""
    return torch.argmax(final_votes, dim=1)


def _count_gap(binary_assignment, clauses):
    """Count unsatisfied clauses. binary_assignment is a list of 0/1 (0-indexed)."""
    unsatisfied = 0
    for clause in clauses:
        satisfied = False
        for lit in clause:
            var = abs(lit) - 1
            val = binary_assignment[var]
            if (lit > 0 and val == 1) or (lit < 0 and val == 0):
                satisfied = True
                break
        if not satisfied:
            unsatisfied += 1
    return unsatisfied


def _assignment_from_embeddings_kmeans(final_embeds, var_batch, num_graphs, clauses_list):
    """
    Infer binary variable assignment from embeddings via k-means (k=2).
    For each formula, tries both cluster→{0,1} mappings, picks the one with fewer
    unsatisfied clauses.
    Returns: list of lists (binary assignments per formula).
    """
    from sklearn.cluster import KMeans

    assignments = []
    for i in range(num_graphs):
        mask = (var_batch == i)
        embeds = final_embeds[mask].detach().cpu().numpy()
        clauses = clauses_list[i]

        if embeds.shape[0] == 0:
            assignments.append([])
            continue

        if embeds.shape[0] == 1:
            a0, a1 = [0], [1]
            assignments.append(a0 if _count_gap(a0, clauses) <= _count_gap(a1, clauses) else a1)
            continue

        try:
            km = KMeans(n_clusters=2, n_init=3, random_state=0)
            labels = km.fit_predict(embeds).tolist()
        except Exception:
            assignments.append([0] * embeds.shape[0])
            continue

        labels_flip = [1 - l for l in labels]
        assignments.append(
            labels if _count_gap(labels, clauses) <= _count_gap(labels_flip, clauses) else labels_flip
        )

    return assignments


def _compute_gaps(binary_assignments, batch):
    """
    Compute per-formula gap (number of unsatisfied clauses).
    binary_assignments: list of lists (per formula, 0/1 per variable)
    Returns: gaps tensor [num_formulas] (long).
    """
    num_formulas = len(batch.clauses)
    gaps = torch.zeros(num_formulas, dtype=torch.long)

    for i, clauses in enumerate(batch.clauses):
        if not clauses or i >= len(binary_assignments):
            continue
        asgn = binary_assignments[i]
        if not asgn:
            continue

        actual_vars = sorted(set(abs(lit) for clause in clauses for lit in clause))
        var_to_idx = {v: idx for idx, v in enumerate(actual_vars)}

        unsatisfied = 0
        for clause in clauses:
            satisfied = False
            for lit in clause:
                idx = var_to_idx[abs(lit)]
                val = asgn[idx] if idx < len(asgn) else 0
                if (lit > 0 and val == 1) or (lit < 0 and val == 0):
                    satisfied = True
                    break
            if not satisfied:
                unsatisfied += 1
        gaps[i] = unsatisfied

    return gaps


def _votes_to_binary_list(final_votes, batch, graph_type):
    """
    Convert model assignment logits to per-formula lists of binary values (0/1).
    Works for both VCG (graph_type='var') and LCG (graph_type='lit').
    """
    if graph_type == 'lit':
        grouped = [final_votes[batch.x_l_batch == i]
                   for i in range(int(batch.x_l_batch.max().item()) + 1)]
        votes = torch.cat([v[:v.size(0) // 2] for v in grouped])
        batch_idx = batch.x_v_batch   # same as var batch, just used for splitting
    else:
        votes = final_votes
        batch_idx = batch.x_v_batch

    binary = torch.argmax(votes, dim=1).cpu().tolist()

    # Split into per-formula lists
    num_formulas = int(batch.num_graphs)
    result = []
    if hasattr(batch, 'num_variables') and batch.num_variables is not None:
        num_vars = batch.num_variables
        if not isinstance(num_vars, torch.Tensor):
            num_vars = torch.tensor(num_vars)
        cumsum = torch.cumsum(num_vars, dim=0).tolist()
        for i in range(num_formulas):
            start = 0 if i == 0 else int(cumsum[i - 1])
            end = int(cumsum[i])
            result.append(binary[start:end])
    else:
        # Fallback: use batch index
        batch_idx_list = batch_idx.cpu().tolist()
        per_formula = [[] for _ in range(num_formulas)]
        for node_i, (b, val) in enumerate(zip(batch_idx_list, binary)):
            per_formula[b].append(val)
        result = per_formula

    return result


def compute_metrics_resampled(model, batch, num_iters, supervision_mode, graph_type, n_samples):
    """
    Test-time resampling: run the model n_samples times with different random initialisations
    (embeddings are randomly seeded each forward pass) and pick, per formula, the assignment
    with the fewest unsatisfied clauses. Then compute metrics on those best assignments.

    This exploits the stochastic embedding initialisation to explore multiple assignments
    cheaply — matches the test-time scaling described in the paper.
    """
    num_formulas = batch.num_graphs

    # Collect binary assignments from each sample
    all_assignments = [[] for _ in range(num_formulas)]  # [formula_i][sample_j] = list of 0/1

    with torch.no_grad():
        for _ in range(n_samples):
            outputs = model(batch, num_iters)
            if supervision_mode == 'sat':
                sample_asgns = _assignment_from_embeddings_kmeans(
                    outputs['final_embeds'], outputs['var_batch'],
                    num_formulas, batch.clauses
                )
            else:
                sample_asgns = _votes_to_binary_list(outputs['final_votes'], batch, graph_type)

            for i, asgn in enumerate(sample_asgns):
                all_assignments[i].append(asgn)

    # For each formula pick the sample with fewest unsatisfied clauses
    best_assignments = []
    for i, clauses in enumerate(batch.clauses):
        candidates = all_assignments[i]
        best_asgn = min(candidates, key=lambda a: _count_gap(a, clauses)) if candidates else []
        best_assignments.append(best_asgn)

    gaps = _compute_gaps(best_assignments, batch)

    is_sat = (batch.y.cpu() == 1)
    is_unsat = ~is_sat
    sat_count = is_sat.sum().item()
    unsat_count = is_unsat.sum().item()

    correct = ((gaps == 0) & is_sat) | ((gaps > 0) & is_unsat)
    return {
        'dec_acc': correct.float().mean().item(),
        'sat_acc': (gaps[is_sat] == 0).float().mean().item() if sat_count > 0 else 0.0,
        'avg_gap': gaps.float().mean().item() if num_formulas > 0 else 0.0,
        'avg_gap_on_sat': gaps[is_sat].float().mean().item() if sat_count > 0 else 0.0,
        'avg_gap_on_unsat': gaps[is_unsat].float().mean().item() if unsat_count > 0 else 0.0,
    }


def compute_gap_per_iteration(all_votes, batch, graph_type):
    """
    Compute average gap at each message-passing iteration.

    all_votes: list of tensors [n_vars, 2], one per iteration (from collect_all_votes=True).
    Returns a numpy array of shape [num_iters] with mean gap across instances at each step.
    This is used for test-time scaling analysis (e.g. plotting gap vs iteration count).
    """
    num_iters = len(all_votes)
    avg_gaps = np.zeros(num_iters)

    for t, votes_t in enumerate(all_votes):
        binary_assignments = _votes_to_binary_list(votes_t, batch, graph_type)
        gaps = _compute_gaps(binary_assignments, batch)
        avg_gaps[t] = gaps.float().mean().item() if len(gaps) > 0 else 0.0

    return avg_gaps


def compute_metrics(outputs, batch, supervision_mode, graph_type):
    """
    Compute unified evaluation metrics for all supervision modes.

    Always returns:
      dec_acc         — Decision Accuracy: +1 if (gap==0 & SAT) or (gap>0 & UNSAT)
                        Note: UNSAT instances always get +1 since no assignment can satisfy them
      sat_acc         — SAT Accuracy: % of SAT instances where a satisfying assignment was found
      avg_gap         — Mean unsatisfied clauses over all instances
      avg_gap_on_sat  — Mean gap over SAT instances only
      avg_gap_on_unsat— Mean gap over UNSAT instances only
    """
    num_graphs = batch.num_graphs

    if supervision_mode == 'sat':
        # Infer assignment via k-means on final embeddings
        binary_assignments = _assignment_from_embeddings_kmeans(
            outputs['final_embeds'], outputs['var_batch'],
            num_graphs, batch.clauses
        )
    else:
        binary_assignments = _votes_to_binary_list(
            outputs['final_votes'], batch, graph_type
        )

    gaps = _compute_gaps(binary_assignments, batch)

    is_sat = (batch.y.cpu() == 1)
    is_unsat = ~is_sat

    # Decision accuracy: +1 if a satisfying assignment was found for a SAT instance,
    # or if the formula is UNSAT (gap > 0 is expected — no assignment can satisfy it).
    correct = ((gaps == 0) & is_sat) | ((gaps > 0) & is_unsat)
    dec_acc = correct.float().mean().item()

    # SAT accuracy
    sat_count = is_sat.sum().item()
    sat_acc = (gaps[is_sat] == 0).float().mean().item() if sat_count > 0 else 0.0

    # Gap statistics
    avg_gap = gaps.float().mean().item() if num_graphs > 0 else 0.0
    avg_gap_on_sat = gaps[is_sat].float().mean().item() if sat_count > 0 else 0.0
    unsat_count = is_unsat.sum().item()
    avg_gap_on_unsat = gaps[is_unsat].float().mean().item() if unsat_count > 0 else 0.0

    return {
        'dec_acc': dec_acc,
        'sat_acc': sat_acc,
        'avg_gap': avg_gap,
        'avg_gap_on_sat': avg_gap_on_sat,
        'avg_gap_on_unsat': avg_gap_on_unsat,
    }
