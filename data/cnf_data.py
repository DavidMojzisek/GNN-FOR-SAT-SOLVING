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

_PIN_MEMORY = torch.cuda.is_available()


class SATInstance(Data):
    def __init__(self, edge_index_lit=None, edge_index_var=None, polarities=None,
                 x_l=None, x_v=None, x_c=None, y=None, assignment=None, clauses=None):
        super().__init__()
        self.edge_index_lit = edge_index_lit
        self.edge_index_var = edge_index_var
        self.polarities = polarities
        # x_l / x_v / x_c are 1-D dummy tensors used only to carry node counts and
        # to let PyG generate x_l_batch / x_v_batch / x_c_batch via follow_batch.
        # The model reinitialises node embeddings randomly at each forward pass;
        # these tensors are never read as feature vectors.
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

        # 1-D dummy tensors: carry node counts for PyG batching (follow_batch).
        # Shape [n] instead of [n, d_model] saves ~d_model× RAM per instance.
        # The model never reads these as features — it reinits embeddings randomly.
        x_l = torch.zeros(2 * num_vars)
        x_v = torch.zeros(num_vars)
        x_c = torch.zeros(len(clauses))

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
    def __init__(self, root, graph_type='both'):
        self.graph_type = graph_type
        self._data_root = osp.abspath(root)  # Store absolute path before super().__init__
        super().__init__(root)
        self.data, self.slices = torch.load(self.processed_paths[0], weights_only=False)

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
    def __init__(self, root, graph_type='both'):
        self.graph_type = graph_type
        self._data_root = osp.abspath(root)  # Store absolute path before super().__init__

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
        # Return the expected file names; PyG's framework calls process() if any are missing.
        return [f'data_{i}.pt' for i in range(len(self.raw_file_names))]

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

        return torch.load(processed_path, weights_only=False)


class SATDataModule(pl.LightningDataModule):
    def __init__(self, data, batch_size=32, graph_type='both', in_memory=True, num_workers=0, supervision_mode='sat',
                 curr_val_dataset=None):
        super().__init__()
        self.train_batch_size = batch_size
        self.val_batch_size = batch_size
        self.test_batch_size = batch_size
        self.graph_type = graph_type
        self.in_memory = in_memory
        self.num_workers = num_workers
        self.follow_batch = ['x_l', 'x_v', 'x_c']
        self.train_dataset, self.val_dataset, self.test_dataset = data['train'], data['val'], data['test']
        self.supervision_mode = supervision_mode
        # Mixed-size val dataset for curriculum-stage monitoring.
        # When provided and curriculum is active, filtered by current window → val_curr_* metrics.
        # When None, falls back to filtering val_dataset (warns if no match, e.g. val is single-size).
        self.curr_val_dataset = curr_val_dataset
        self._curriculum_max_vars = None       # None = no curriculum filtering
        self._curriculum_min_vars_window = None

    def set_curriculum_max_vars(self, max_vars, min_vars_window=None):
        """
        Called by CurriculumCallback to update curriculum training window.
        Includes instances where min_vars_window <= num_variables <= max_vars.
        min_vars_window=None means include all sizes up to max_vars.
        """
        self._curriculum_max_vars = max_vars
        self._curriculum_min_vars_window = min_vars_window

    def _filter_dataset_by_size(self, dataset, max_vars, min_vars=None):
        """Filter a dataset to instances within [min_vars, max_vars]."""
        window_str = f"[{min_vars}, {max_vars}]" if min_vars is not None else f"[*, {max_vars}]"
        if hasattr(dataset, 'data') and hasattr(dataset.data, 'num_variables'):
            nvars = dataset.data.num_variables
            if not isinstance(nvars, torch.Tensor):
                nvars = torch.tensor(nvars)
            mask = nvars <= max_vars
            if min_vars is not None:
                mask = mask & (nvars >= min_vars)
            indices = torch.nonzero(mask).squeeze()
            if indices.dim() == 0:
                indices = [indices.item()]
            else:
                indices = indices.tolist()
            if not indices:
                logging.warning(
                    f"Curriculum filter {window_str}: 0 / {len(dataset)} instances matched "
                    f"(dataset size range: [{nvars.min().item()}, {nvars.max().item()}]). "
                    "Returning full dataset — curriculum val metrics will NOT reflect current stage."
                )
                return dataset
            return dataset[indices]
        # Fallback for list-like datasets
        filtered = [inst for inst in dataset
                    if inst.num_variables <= max_vars
                    and (min_vars is None or inst.num_variables >= min_vars)]
        if not filtered:
            all_sizes = sorted(set(inst.num_variables for inst in dataset))
            logging.warning(
                f"Curriculum filter {window_str}: 0 / {len(dataset)} instances matched "
                f"(dataset size range: {all_sizes[0]}–{all_sizes[-1]}). "
                "Returning full dataset — curriculum val metrics will NOT reflect current stage."
            )
            return dataset
        return filtered

    def _get_curriculum_dataset(self):
        """Training dataset filtered by current curriculum window."""
        if self._curriculum_max_vars is None:
            return self.train_dataset
        return self._filter_dataset_by_size(
            self.train_dataset, self._curriculum_max_vars,
            self._curriculum_min_vars_window
        )

    def _get_curriculum_val_dataset(self):
        """Validation dataset filtered by current curriculum window (for advancement decision).

        Uses curr_val_dataset (mixed-size) when provided — this is the correct source for
        curriculum-stage monitoring. Falls back to filtering val_dataset with a warning if not set.
        """
        if self._curriculum_max_vars is None:
            return self.curr_val_dataset if self.curr_val_dataset is not None else self.val_dataset
        source = self.curr_val_dataset if self.curr_val_dataset is not None else self.val_dataset
        return self._filter_dataset_by_size(
            source, self._curriculum_max_vars,
            self._curriculum_min_vars_window
        )

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
            self._get_curriculum_dataset(),
            batch_size=self.train_batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            follow_batch=self.follow_batch,
            persistent_workers=(self.num_workers > 0),
            pin_memory=_PIN_MEMORY,
        )

    def val_dataloader(self):
        # Always return a list so validation_step always receives dataloader_idx.
        # [0] = full val set (always); [1] = curriculum-filtered val set (only when active).
        full_loader = DataLoader(
            self.val_dataset,
            batch_size=self.val_batch_size,
            num_workers=self.num_workers,
            follow_batch=self.follow_batch,
            persistent_workers=(self.num_workers > 0),
            pin_memory=_PIN_MEMORY,
        )
        if self._curriculum_max_vars is not None:
            curr_dataset = self._get_curriculum_val_dataset()
            curr_loader = DataLoader(
                curr_dataset,
                batch_size=self.val_batch_size,
                num_workers=self.num_workers,
                follow_batch=self.follow_batch,
                persistent_workers=(self.num_workers > 0),
                pin_memory=_PIN_MEMORY,
            )
            return [full_loader, curr_loader]
        return [full_loader]

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.test_batch_size,
            num_workers=self.num_workers,
            follow_batch=self.follow_batch,
            persistent_workers=(self.num_workers > 0),
            pin_memory=_PIN_MEMORY,
        )
