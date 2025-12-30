import torch

def compute_metrics(outputs, batch, supervision_mode, graph_type):

    metrics = {}

    if supervision_mode == 'sat':

        pred_sat = outputs['vote_reduced'].squeeze() > 0

        accuracy = (pred_sat == batch.y).float().mean()

        metrics['accuracy'] = accuracy

        return metrics

       

    # Handle assignment-based metrics

    if graph_type == 'lit':

        votes = [outputs['final_votes'][batch.x_l_batch==i] for i in range(batch.x_l_batch.max()+1)]

        assignment = torch.cat([v[:v.size(0)//2] for v in votes])

    else:  # var case

        votes = outputs['final_votes']

        assignment = votes

           

    if assignment.shape[1] == 2:

        assignment = torch.argmax(assignment, dim=1)

           

    assignment = assignment * 2 - 1

    

    # Pre-allocate arrays for vectorized operations

    var_cumsum = torch.cumsum(batch.num_variables, dim=0)

    num_formulas = len(batch.clauses)

    

    # Initialize metric counters

    gaps = torch.zeros(num_formulas, dtype=torch.int64, device=batch.y.device)

    num_clauses = torch.tensor([len(c) for c in batch.clauses], dtype=torch.int64, device=batch.y.device)

    total_clauses = num_clauses.sum().item()

    

    # Process each formula once

    for i, clauses in enumerate(batch.clauses):

        if len(clauses) == 0:

            continue

            

        # Extract assignment for this formula

        start_idx = 0 if i == 0 else var_cumsum[i-1].item()

        end_idx = var_cumsum[i].item()

        example_assignment = assignment[start_idx:end_idx]

        

        # Create a faster lookup for variable indices

        actual_vars = sorted(list(set(abs(lit) for clause in clauses for lit in clause)))

        var_to_idx = {var: idx for idx, var in enumerate(actual_vars)}

        

        # Count satisfied clauses

        satisfied = 0

        for clause in clauses:

            # Process entire clause at once

            is_satisfied = False

            for lit in clause:

                var_idx = var_to_idx[abs(lit)]

                lit_val = example_assignment[var_idx] if lit > 0 else -example_assignment[var_idx]

                

                if lit_val > 0:

                    satisfied += 1

                    break

                    

        gaps[i] = len(clauses) - satisfied

    

    # Split metrics for SAT and UNSAT instances

    is_sat = batch.y == 1

    is_unsat = ~is_sat

    

    sat_instances = is_sat.sum().item()

    unsat_instances = is_unsat.sum().item()

    

    # Calculate gap metrics

    total_gap = gaps.sum().item()

    total_gap_sat = gaps[is_sat].sum().item() if sat_instances > 0 else 0

    total_gap_unsat = gaps[is_unsat].sum().item() if unsat_instances > 0 else 0

    

    # Calculate clause counts

    total_clauses_sat = num_clauses[is_sat].sum().item() if sat_instances > 0 else 0

    total_clauses_unsat = num_clauses[is_unsat].sum().item() if unsat_instances > 0 else 0

    

    # Calculate accuracy

    correct_pred = (gaps == 0) | ((batch.y == 0) & (gaps > 0))

    accuracy = correct_pred.float().mean().item()

    

    # Calculate satisfaction rates

    satisfaction_rate = 1 - (total_gap / total_clauses) if total_clauses > 0 else 0.0

    satisfaction_rate_sat = 1 - (total_gap_sat / total_clauses_sat) if total_clauses_sat > 0 else 0.0

    satisfaction_rate_unsat = 1 - (total_gap_unsat / total_clauses_unsat) if total_clauses_unsat > 0 else 0.0

    

    # Average gaps

    avg_gap = total_gap / num_formulas if num_formulas > 0 else 0.0

    avg_gap_on_sat = total_gap_sat / sat_instances if sat_instances > 0 else 0.0

    avg_gap_on_unsat = total_gap_unsat / unsat_instances if unsat_instances > 0 else 0.0

    

    # Collect all metrics

    metrics.update({

        'avg_gap': avg_gap,

        'avg_gap_on_sat': avg_gap_on_sat,

        'avg_gap_on_unsat': avg_gap_on_unsat,

        'satisfaction_rate': satisfaction_rate,

        'satisfaction_rate_sat': satisfaction_rate_sat,

        'satisfaction_rate_unsat': satisfaction_rate_unsat,

        'accuracy': accuracy,

        'sat_instances': sat_instances,

        'unsat_instances': unsat_instances

    })

    

    return metrics