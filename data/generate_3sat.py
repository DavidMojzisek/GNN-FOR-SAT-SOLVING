import math
import numpy as np
import random
import os
import shutil
import argparse
from pysat.solvers import Glucose3
import networkx as nx
from cnfgen import RandomKCNF

def write_dimacs_to(n_vars, iclauses, out_filename):
    with open(out_filename, 'w') as f:
        f.write("p cnf %d %d\n" % (n_vars, len(iclauses)))
        for c in iclauses:
            for x in c:
                f.write("%d " % x)
            f.write("0\n")

def mk_out_filename(opts, n_vars, t, is_sat):
    prefix = "%s/3sat_n=%.4d_r=%.2f_t=%d" % \
        (opts.out_dir, n_vars, opts.ratio, t)
    return "%s_sat=%d.dimacs" % (prefix, 1 if is_sat else 0)

def check_connected(n_vars, clauses):
    """Check if variable interaction graph is connected."""
    G = nx.Graph()
    G.add_nodes_from(range(1, n_vars + 1))
    
    for clause in clauses:
        variables = [abs(lit) for lit in clause]
        for i in range(len(variables)):
            for j in range(i + 1, len(variables)):
                G.add_edge(variables[i], variables[j])
                
    return nx.is_connected(G)

def generate_3sat_formula(n_vars, ratio):
    """Generate a 3-SAT formula with specified clause/variable ratio."""
    n_clauses = int(ratio * n_vars)
    
    cnf = RandomKCNF(3, n_vars, n_clauses)
    clauses = list(cnf.clauses())
    clauses = [list(map(int, clause)) for clause in clauses]
    
    return clauses

def generate_instance(opts):
    """Generate a single 3-SAT instance."""
    n = random.randint(opts.min_n, opts.max_n)
    
    max_attempts = 100
    attempts = 0
    
    while attempts < max_attempts:
        clauses = generate_3sat_formula(n, opts.ratio)
        
        if not opts.no_connectivity_check and not check_connected(n, clauses):
            attempts += 1
            continue
            
        solver = Glucose3()
        for clause in clauses:
            solver.add_clause(clause)
            
        is_sat = solver.solve()
        return n, clauses, is_sat
            
        attempts += 1
    
    raise RuntimeError(f"Failed to generate instance after {max_attempts} attempts")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--out_dir', action='store', type=str)
    parser.add_argument('--n_instances', action='store', type=int,
                       help='Number of instances to generate')
    parser.add_argument('--min_n', action='store', dest='min_n', type=int, default=40)
    parser.add_argument('--max_n', action='store', dest='max_n', type=int, default=40)
    parser.add_argument('--ratio', type=float, default=4.26, 
                       help='Clause to variable ratio (default: 4.26 phase transition)')
    parser.add_argument('--no_connectivity_check', action='store_true',
                       help='Disable checking if variable interaction graph is connected')
    parser.add_argument('--py_seed', action='store', dest='py_seed', type=int, default=None)
    parser.add_argument('--np_seed', action='store', dest='np_seed', type=int, default=None)
    parser.add_argument('--print_interval', action='store', dest='print_interval', type=int, default=10000)
    opts = parser.parse_args()
    
    if opts.py_seed is not None: random.seed(opts.py_seed)
    if opts.np_seed is not None: np.random.seed(opts.np_seed)
    
    out_dir = opts.out_dir
    if os.path.exists(out_dir):
        print("Output directory %s already exists. Replacing." % out_dir)
        shutil.rmtree(out_dir)
    os.makedirs(out_dir)
    
    for i in range(opts.n_instances):
        if i % opts.print_interval == 0: print("[%d]" % i)
        n_vars, clauses, is_sat = generate_instance(opts)
        
        out_filename = mk_out_filename(opts, n_vars, i, is_sat)
        write_dimacs_to(n_vars, clauses, out_filename)