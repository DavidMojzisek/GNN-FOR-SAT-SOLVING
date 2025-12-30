import torch
from torch_geometric.data import Data, Dataset, InMemoryDataset
from torch_geometric.loader import DataLoader
from torch_sparse import SparseTensor
import pytorch_lightning as pl
from os import listdir
import os
import os.path as osp
import numpy as np
from pysat.solvers import Glucose3
from pysat.examples.rc2 import RC2
from pysat.formula import CNF
from tqdm import tqdm
import logging

class SATInstance(Data):
   def __init__(self, edge_index_lit=None, edge_index_var=None, polarities=None, 
                x_l=None, x_v=None, x_c=None, y=None, assignment=None, clauses=None):
       super().__init__()
       self.edge_index_lit = edge_index_lit
       self.edge_index_var = edge_index_var
       self.polarities = polarities
       self.x_l = x_l
       self.x_v = x_v
       self.x_c = x_c
       self.y = y
       self.assignment = assignment
       self.clauses = clauses
       
       self.num_literals = x_l.size(0) if x_l is not None else 0
       self.num_variables = x_v.size(0) if x_v is not None else 0
       self.num_clauses = x_c.size(0) if x_c is not None else 0
       
       if edge_index_lit is not None:
           self.adj_t_lit = SparseTensor(
               row=edge_index_lit[1],
               col=edge_index_lit[0],
               sparse_sizes=[self.num_clauses, self.num_literals]
           )
           
       if edge_index_var is not None:
           self.adj_t_var = SparseTensor(
               row=edge_index_var[1],
               col=edge_index_var[0],
               value=polarities,
               sparse_sizes=[self.num_clauses, self.num_variables]
           )

   def __inc__(self, key, value, store):
       if key == 'edge_index_lit':
           return torch.tensor([[self.x_l.size(0)], [self.x_c.size(0)]])
       elif key == 'edge_index_var':
           return torch.tensor([[self.x_v.size(0)], [self.x_c.size(0)]])
       return super().__inc__(key, value)

class BaseSATDataset:
   def _parse_dimacs(self, filename):
       with open(filename) as f:
           lines = f.readlines()
           
       for line in lines:
           tokens = line.strip().split()
           if tokens and tokens[0] == 'p':
               num_vars = int(tokens[2])
               break
               
       clauses = []
       for line in lines:
           tokens = line.strip().split()
           if tokens and tokens[0] not in ['c', 'p']:
               clause = [int(x) for x in tokens[:-1]]
               clauses.append(clause)
               
       return num_vars, clauses

   def _create_instance(self, num_vars, clauses, is_sat, assignment):
       edge_index_lit = [[], []]
       edge_index_var = [[], []]
       polarities = []
       
       for clause_idx, clause in enumerate(clauses):
           for lit in clause:
               var_idx = abs(lit) - 1
               lit_idx = var_idx if lit > 0 else var_idx + num_vars
               
               edge_index_lit[0].append(lit_idx)
               edge_index_lit[1].append(clause_idx)
               edge_index_var[0].append(var_idx)
               edge_index_var[1].append(clause_idx)
               polarities.append(1 if lit > 0 else -1)

       edge_index_lit = torch.tensor(edge_index_lit, dtype=torch.long)
       edge_index_var = torch.tensor(edge_index_var, dtype=torch.long)
       polarities = torch.tensor(polarities, dtype=torch.float)
       
       x_l = self.l_init.repeat(2 * num_vars, 1)
       x_v = self.v_init.repeat(num_vars, 1)
       x_c = self.c_init.repeat(len(clauses), 1)
       
       return SATInstance(
           edge_index_lit=edge_index_lit,
           edge_index_var=edge_index_var,
           polarities=polarities,
           x_l=x_l,
           x_v=x_v,
           x_c=x_c,
           y=torch.tensor([is_sat], dtype=torch.float),
           assignment=torch.sign(torch.tensor(assignment, dtype=torch.float)),
           clauses=clauses
       )

   def _prepare_instance(self, raw_path):
           num_vars, clauses = self._parse_dimacs(raw_path)
           solver = Glucose3()
           
           for clause in clauses:
               solver.add_clause(clause)
               
           is_sat = solver.solve()
           
           if is_sat:
               assignment = solver.get_model()
           else:
               cnf = CNF(from_clauses=clauses)
               wcnf = cnf.weighted()
               with RC2(wcnf) as rc2:
                   assignment = rc2.compute() 
           instance = self._create_instance(num_vars, clauses, is_sat, assignment)
           return instance

class SATInMemoryDataset(InMemoryDataset, BaseSATDataset):
   def __init__(self, root, d_model=128, graph_type='both'):
       self.d_model = d_model
       self.graph_type = graph_type
       self._data_root = osp.abspath(root)  # Store absolute path before super().__init__
       self.l_init = torch.randn(1, d_model)
       self.v_init = torch.randn(1, d_model)
       self.c_init = torch.randn(1, d_model)
       super().__init__(root)
       self.data, self.slices = torch.load(self.processed_paths[0])

   @property
   def raw_dir(self):
       return self._data_root

   @property
   def raw_file_names(self):
       return sorted([f for f in listdir(self._data_root) if f.endswith('.dimacs')])

   @property
   def processed_file_names(self):
       return ['data.pt']

   def process(self):
       data_list = []

       for raw_path in tqdm(self.raw_paths, desc="Processing CNFs"):
           instance = self._prepare_instance(raw_path)
           if instance is not None:
               data_list.append(instance)

       data, slices = self.collate(data_list)
       torch.save((data, slices), self.processed_paths[0])

class SATOnDiskDataset(Dataset, BaseSATDataset):
   def __init__(self, root, d_model=128, graph_type='both'):
       self.d_model = d_model
       self.graph_type = graph_type
       self._data_root = osp.abspath(root)  # Store absolute path before super().__init__
       self.l_init = torch.randn(1, d_model)
       self.v_init = torch.randn(1, d_model)
       self.c_init = torch.randn(1, d_model)

       # Create processed directory if it doesn't exist
       os.makedirs(osp.join(self._data_root, "processed"), exist_ok=True)

       super().__init__(root)

   @property
   def raw_dir(self):
       return self._data_root

   @property
   def raw_file_names(self):
       return sorted([f for f in listdir(self._data_root) if f.endswith('.dimacs')])

   @property
   def processed_file_names(self):
       if not osp.exists(osp.join(self.processed_dir)):
           os.makedirs(osp.join(self.processed_dir), exist_ok=True)
       
       processed_files = [f'data_{i}.pt' for i in range(len(self.raw_file_names))]
       
       # Check if the processed files already exist
       existing_processed_files = [f for f in processed_files if osp.exists(osp.join(self.processed_dir, f))]
       
       # If no processed files exist yet, create them
       if not existing_processed_files:
           self.process()
           
       return processed_files

   def process(self):
       for idx, raw_path in enumerate(tqdm(self.raw_paths, desc="Processing CNFs")):
           try:
               instance = self._prepare_instance(raw_path)
               if instance is not None:
                   torch.save(instance, osp.join(self.processed_dir, f'data_{idx}.pt'))
           except Exception as e:
               print(f"Error processing file {raw_path}: {e}")
               continue

   def len(self):
       return len(self.raw_file_names)

   def get(self, idx):
       processed_path = osp.join(self.processed_dir, f'data_{idx}.pt')
       
       # Check if processed file exists, if not, process it
       if not osp.exists(processed_path):
           raw_path = self.raw_paths[idx]
           instance = self._prepare_instance(raw_path)
           if instance is not None:
               torch.save(instance, processed_path)
               return instance
           else:
               return None
       
       return torch.load(processed_path)

class SATDataModule(pl.LightningDataModule):
    def __init__(self, data, batch_size=32, d_model=128, graph_type='both', in_memory=True, num_workers=0, supervision_mode='sat'):
        super().__init__()
        self.train_batch_size = batch_size
        self.val_batch_size = batch_size
        self.test_batch_size = batch_size
        self.d_model = d_model
        self.graph_type = graph_type
        self.in_memory = in_memory
        self.num_workers = num_workers
        self.follow_batch = ['x_l', 'x_v', 'x_c']
        self.train_dataset, self.val_dataset, self.test_dataset = data['train'], data['val'], data['test']
        self.supervision_mode = supervision_mode

    def free_memory(self):
        """Explicitly free dataset memory."""
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        
        # Force garbage collection
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    @staticmethod
    def filter_sat_only(dataset):
        """Filter a dataset to include only satisfiable instances."""
        if hasattr(dataset, 'data') and hasattr(dataset.data, 'y'):
            # For InMemoryDataset format
            sat_mask = dataset.data.y == 1
            sat_count = torch.sum(sat_mask).item()
            logging.info(f"Before filtering: {len(dataset)} total instances, {sat_count} SAT instances")
            
            sat_indices = torch.nonzero(sat_mask).squeeze().tolist()
            # Handle edge cases for single instance or empty results
            if isinstance(sat_indices, int):
                sat_indices = [sat_indices]
            elif len(sat_indices) == 0:
                logging.warning("No satisfiable instances found!")
                return dataset
                
            filtered_dataset = dataset[sat_indices]
            logging.info(f"After filtering: {len(filtered_dataset)} SAT instances")
            return filtered_dataset
        else:
            # For list-like datasets
            sat_count = sum(1 for inst in dataset if inst.y.item() == 1)
            logging.info(f"Before filtering: {len(dataset)} total instances, {sat_count} SAT instances")
            
            filtered_dataset = [inst for inst in dataset if inst.y.item() == 1]
            logging.info(f"After filtering: {len(filtered_dataset)} SAT instances")
            return filtered_dataset

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.train_batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            follow_batch=self.follow_batch,
            persistent_workers=False,  # Don't keep workers alive between epochs
            pin_memory=False  # Avoid extra memory usage with pinned memory
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset, 
            batch_size=self.val_batch_size, 
            num_workers=self.num_workers,
            follow_batch=self.follow_batch,
            persistent_workers=False,
            pin_memory=False
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset, 
            batch_size=self.test_batch_size, 
            num_workers=self.num_workers,
            follow_batch=self.follow_batch,
            persistent_workers=False,
            pin_memory=False
        )