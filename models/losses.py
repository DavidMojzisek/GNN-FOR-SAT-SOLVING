import torch
import torch.nn.functional as F
from torch_sparse import matmul
from pysat.formula import WCNF
from pysat.examples.rc2 import RC2

def compute_sat_loss(outputs, batch):
    """Compute binary cross entropy loss for SAT classification."""
    return torch.nn.functional.binary_cross_entropy_with_logits(
        outputs['vote_reduced'].squeeze(),
        batch.y
    )

def compute_assignment_CE_loss(outputs, batch, graph_type):
    """Compute cross-entropy loss for assignment prediction."""
    if graph_type == 'lit':    
        grouped_votes = [outputs['final_votes'][batch.x_l_batch==i] for i in range(batch.x_l_batch.max()+1)]
        processed_votes = []
        for votes in grouped_votes:
            n_vars = votes.size(0) // 2
            var_votes = votes[:n_vars]
            processed_votes.append(var_votes)
        votes = torch.cat(processed_votes)
    else:
        votes = outputs['final_votes']

    node_labels = ((batch.assignment + 1)/2)
    loss = torch.nn.functional.cross_entropy(votes, node_labels.long().to(votes.device)) 
    
    return loss

def _find_closest_assignment(model_predictions, clauses, is_sat, device):
    """
    Finds the closest satisfying assignment to the current model predictions.
    
    Args:
        model_predictions: Probabilities from model (values between 0-1)
        clauses: List of clauses for the formula
        is_sat: Boolean indicating if the formula is SAT
        device: Torch device
        
    Returns:
        Closest satisfying assignment (values are 0 or 1 for CE loss)
    """
    # Convert probabilities to numpy
    target_probs = model_predictions.detach().cpu().numpy()
    num_vars = len(target_probs)
    
    # Create weighted CNF formula
    wcnf = WCNF()
    
    # Add clauses
    if is_sat:
        # For SAT, add original clauses as hard constraints
        for clause in clauses:
            wcnf.append(clause, weight=None)  # Hard constraint
    else:
        # For UNSAT, add as soft constraints with high weight
        high_weight = num_vars + 1  # Higher than weights for variable preferences
        for clause in clauses:
            wcnf.append(clause, weight=high_weight)
    
    # Add preferences based on model predictions
    for i, prob in enumerate(target_probs):
        var_idx = i + 1  # PySAT uses 1-indexed variables
        
        if prob >= 0.5:
            # Model predicts variable is True - add preference for True
            wcnf.append([var_idx], weight=1)
        else:
            # Model predicts variable is False - add preference for False
            wcnf.append([-var_idx], weight=1)
    
    # Solve with RC2
    with RC2(wcnf) as solver:
        solution = solver.compute()
    
    if solution is None:
        # This shouldn't happen for SAT instances, but handle it just in case
        return torch.zeros(num_vars, dtype=torch.long, device=device)
    
    # Convert solution to class indices (0 or 1) for CE loss
    assignment = torch.zeros(num_vars, dtype=torch.long, device=device)
    for lit in solution:
        var_idx = abs(lit) - 1  # Convert to 0-indexed
        if var_idx < num_vars:  # Ensure we're within bounds
            assignment[var_idx] = 1 if lit > 0 else 0
    
    return assignment

def compute_closest_assignment_CE_loss(outputs, batch, graph_type):
    """
    Compute CE loss between model predictions and closest valid assignments.
    
    Args:
        outputs: Model outputs containing predictions
        batch: Batch data including clauses
        graph_type: 'lit' or 'var'
        
    Returns:
        Loss tensor
    """
    device = batch.y.device
    
    # Get model predictions and convert to probabilities
    if graph_type == 'lit':    
        # For lit graph, extract just the positive literals
        grouped_votes = [outputs['final_votes'][batch.x_l_batch==i] for i in range(batch.x_l_batch.max()+1)]
        processed_votes = []
        for votes in grouped_votes:
            n_vars = votes.size(0) // 2
            var_votes = votes[:n_vars]
            processed_votes.append(var_votes)
        votes = torch.cat(processed_votes)
    else:
        # For var graph, use all predictions
        votes = outputs['final_votes']
    
    # Extract probability of positive class
    probs = votes.softmax(dim=1)[:, 1]
    
    # Process each formula in the batch
    var_cumsum = torch.cumsum(batch.num_variables, dim=0)
    closest_assignments = []
    
    for i, clauses in enumerate(batch.clauses):
        # Extract predictions for this formula
        start_idx = 0 if i == 0 else var_cumsum[i-1].item()
        end_idx = var_cumsum[i].item()
        formula_probs = probs[start_idx:end_idx]
        
        # Determine if this formula is SAT
        is_sat = batch.y[i].item() == 1
        
        # Find closest satisfying assignment
        assignment = _find_closest_assignment(formula_probs, clauses, is_sat, device)
        closest_assignments.append(assignment)
    
    # Combine all closest assignments
    targets = torch.cat(closest_assignments)
    
    # Compute cross entropy loss
    return torch.nn.functional.cross_entropy(votes, targets)


def _compute_clause_satisfaction_probs(outputs, batch, graph_type):
    """
    Compute clause satisfaction probabilities V_c for all clauses.

    Based on equation from paper:
    V_c(p) = 1 - ∏_{i∈c+}(1-p_i) ∏_{i∈c-} p_i

    Works for both 'lit' and 'var' graph types.

    Returns:
        clause_probs: Tensor of clause satisfaction probabilities
        c_batch: Clause batch indices
    """
    device = outputs['final_votes'].device
    epsilon = 1e-10

    # Get variable predictions (probabilities)
    var_pred = outputs['final_votes']
    if var_pred.dim() > 1 and var_pred.shape[1] == 2:  # Handle CE loss case
        var_pred = F.softmax(var_pred, dim=1)[:, 1]

    if graph_type == 'lit':
        # For literal graph: extract positive literals only
        votes = [outputs['final_votes'][batch.x_l_batch==i] for i in range(batch.num_graphs)]
        batch_preds = []
        for v in votes:
            n_vars = v.size(0) // 2
            var_pred = v[:n_vars]
            batch_preds.append(var_pred)
        var_pred = torch.cat(batch_preds)
        if var_pred.dim() > 1 and var_pred.shape[1] == 2:
            var_pred = F.softmax(var_pred, dim=1)[:, 1]

        edge_index = batch.edge_index_lit
        c_size = batch.x_c.size(0)
        c_batch = batch.x_c_batch

        # For lit graph, we need polarities from the first n_vars literals
        # Build polarities: positive literals have polarity +1, negative have -1
        polarities = batch.polarities

    else:  # 'var' graph type
        edge_index = batch.edge_index_var
        polarities = batch.polarities
        c_size = batch.x_c.size(0)
        c_batch = batch.x_c_batch

    # Compute clause satisfaction probabilities
    # V_c = 1 - ∏_{i∈c+}(1-p_i) ∏_{i∈c-} p_i

    # For each edge (variable -> clause):
    # - If positive literal: contributes (1 - p_i) to the product
    # - If negative literal: contributes p_i to the product

    # Compute log probabilities for numerical stability
    log_complement_contrib = torch.where(
        polarities > 0,
        torch.log(1 - var_pred[edge_index[0]] + epsilon),  # log(1 - p_i) for positive
        torch.log(var_pred[edge_index[0]] + epsilon)        # log(p_i) for negative
    )

    # Sum log contributions per clause
    clause_log_product = torch.zeros(c_size, device=device)
    clause_log_product.index_add_(0, edge_index[1], log_complement_contrib)

    # V_c = 1 - exp(sum(log(...))) = 1 - ∏(...)
    clause_probs = 1 - torch.exp(clause_log_product).clamp(max=1.0-epsilon)

    return clause_probs, c_batch


def compute_unsupervised_loss_linear(outputs, batch, graph_type):
    """
    Linear unsupervised loss: L_lin = -∑_c V_c
    Counts expected satisfied clauses but provides weak gradients when V_c ≈ 1.

    Works for both 'lit' and 'var' graph types.

    Args:
        outputs: Model outputs dictionary
        batch: Input batch data
        graph_type: 'lit' or 'var'

    Returns:
        Scalar loss tensor
    """
    clause_probs, c_batch = _compute_clause_satisfaction_probs(outputs, batch, graph_type)

    # Sum clause probabilities per formula
    formula_scores = torch.zeros(batch.num_graphs, device=clause_probs.device)
    formula_scores.index_add_(0, c_batch, clause_probs)

    # Loss = -sum(V_c), averaged over batch
    # We want to maximize sum of clause satisfaction, so minimize negative
    return -formula_scores.mean()


def compute_unsupervised_loss_log(outputs, batch, graph_type):
    """
    Logarithmic unsupervised loss: L_log = -∑_c log(V_c)
    Amplifies gradients for nearly-unsatisfied clauses. As V_c → 0, penalty diverges,
    forcing optimizer to address every clause.

    Works for both 'lit' and 'var' graph types.

    Args:
        outputs: Model outputs dictionary
        batch: Input batch data
        graph_type: 'lit' or 'var'

    Returns:
        Scalar loss tensor
    """
    epsilon = 1e-10
    clause_probs, c_batch = _compute_clause_satisfaction_probs(outputs, batch, graph_type)

    # Log of clause probabilities
    log_clause_probs = torch.log(clause_probs + epsilon)

    # Sum log probabilities per formula
    formula_log_scores = torch.zeros(batch.num_graphs, device=clause_probs.device)
    formula_log_scores.index_add_(0, c_batch, log_clause_probs)

    # Loss = -sum(log(V_c)), averaged over batch
    return -formula_log_scores.mean()


def compute_unsupervised_loss_quad(outputs, batch, graph_type):
    """
    Quadratic unsupervised loss: L_quad = ∑_c (1 - V_c)²
    Provides stronger gradients than linear aggregation near V_c = 1
    while avoiding unbounded penalties.

    Works for both 'lit' and 'var' graph types.

    Args:
        outputs: Model outputs dictionary
        batch: Input batch data
        graph_type: 'lit' or 'var'

    Returns:
        Scalar loss tensor
    """
    clause_probs, c_batch = _compute_clause_satisfaction_probs(outputs, batch, graph_type)

    # Squared penalty for unsatisfied clauses
    clause_penalties = (1 - clause_probs) ** 2

    # Sum penalties per formula
    formula_penalties = torch.zeros(batch.num_graphs, device=clause_probs.device)
    formula_penalties.index_add_(0, c_batch, clause_penalties)

    # Loss = sum((1-V_c)²), averaged over batch
    return formula_penalties.mean()


# Keep old names for backwards compatibility
def compute_unsupervised_loss_1(outputs, batch, graph_type):
    """Alias for linear loss for backwards compatibility."""
    return compute_unsupervised_loss_linear(outputs, batch, graph_type)


def compute_unsupervised_loss_2(outputs, batch, graph_type):
    """Alias for log loss for backwards compatibility."""
    return compute_unsupervised_loss_log(outputs, batch, graph_type)

