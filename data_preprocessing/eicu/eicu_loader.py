
import numpy as np
import torch
from typing import Dict, List, Tuple, Any
from torch.utils.data import Dataset
from .eicu_preprocessor import preprocess_eicu
import logging


class eICUDataset(Dataset):
    """    
    IMPORTANT: Must match exactly what FedFed training loop expects!!! Double check!
    """
    
    def __init__(self, X: np.ndarray, y: np.ndarray, transform=None, target_transform=None):
        """        
        Args:
            X: Feature matrix [n_samples, n_features]
            y: Labels [n_samples]
        """
        self.X = torch.FloatTensor(X)
        self.y = torch.LongTensor(y)
        self.transform = transform
        self.target_transform = target_transform
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        X, y = self.X[idx], self.y[idx]
        if not torch.is_tensor(X):
            X = torch.FloatTensor(X)
        
        if self.transform is not None:
            X = self.transform(X)
            
        if self.target_transform is not None:
            y = self.target_transform(y)
        else:
            if not torch.is_tensor(y):
                y = torch.LongTensor([y]).squeeze()
        
        return X, y


class eICUDataset_truncated(Dataset):
    """
    Truncated eICU dataset for specific client (hospital)
    Matches the pattern of other truncated datasets in FedFed
    """
    
    def __init__(self, root, dataidxs=None, train=True, transform=None, 
                 target_transform=None, full_dataset=None):
        self.root = root
        self.dataidxs = dataidxs
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
        self.full_dataset = full_dataset
        
        if full_dataset is not None:
            if dataidxs is not None:
                self.data = full_dataset.data[dataidxs]
                self.targets = full_dataset.targets[dataidxs]
            else:
                self.data = full_dataset.data
                self.targets = full_dataset.targets
        else:
            # This shouldn't happen in normal flow
            self.data = np.array([])
            self.targets = np.array([])
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        X, y = self.data[idx], self.targets[idx]
        
        if not torch.is_tensor(X):
            X = torch.FloatTensor(X)
            
        if self.transform is not None:
            X = self.transform(X)
            
        if self.target_transform is not None:
            y = self.target_transform(y)
        else:
            if not torch.is_tensor(y):
                y = torch.LongTensor([y]).squeeze()
        
        return X, y


def load_preprocessing_configs(args):
    '''
    Helper - Load eICU preprocessing configs from sepcified args
    '''
    config = {
        'prediction_task': getattr(args, 'prediction_task', 'mortality'),
        'target_hospitals': getattr(args, 'target_hospitals', None),
        'test_size': getattr(args, 'test_size', 0.2),
        'val_size': getattr(args, 'val_size', 0.1),
        'random_seed': getattr(args, 'seed', 42),
        'demo_db_path': getattr(args, 'eicu_demo_path', './data/eicu_demo.db')
    }
    # no hospital specified, use top 10 as in fedweight
    if config['target_hospitals'] is None:
        config['target_hospitals'] = [167, 420, 199, 458, 252, 165, 148, 281, 449, 283]
    
    return config

def create_eicu_datasets(datadir, args = None):
    '''
    Returns:
        train_ds: Full training dataset
        test_ds: Full test dataset
        hospital_to_indices: Mapping of hospital IDs to patient indices
    '''
    config = load_preprocessing_configs(args)
    logging.info(f"Loading eICU data for {config['prediction_task']} prediction...")

    # preprocessing
    preprocessed_data = preprocess_eicu(config['demo_db_path'], config)
    hospital_datasets = preprocessed_data['hospital_datasets']

    all_X_list, all_y_list = [], []
    hospital_to_indices = {}
    current_idx = 0
    sorted_hospitals = sorted(hospital_datasets.keys())
    
    # combine all hospital data
    for hospital_id in sorted_hospitals:
        dataset = hospital_datasets[hospital_id]
        X, y = dataset['X'], dataset['y']
        
        # Record indices for this hospital
        num_samples = len(X)
        hospital_to_indices[hospital_id] = list(range(current_idx, current_idx + num_samples))
        current_idx += num_samples
        
        all_X_list.append(X)
        all_y_list.append(y)
        
    all_X = np.concatenate(all_X_list, axis=0)
    all_y = np.concatenate(all_y_list, axis=0)
    
    from sklearn.model_selection import train_test_split
    train_indices, test_indices = [], []
    
    for hospital_id, indices in hospital_to_indices.items():
        hospital_X = all_X[indices]
        hospital_y = all_y[indices]
        
        # Split this hospital's data
        n_samples = len(indices)
        n_train = int(n_samples * (1 - config['test_size']))
        
        # Stratified split if possible
        if len(np.unique(hospital_y)) > 1:
            train_idx, test_idx = train_test_split(
                indices, test_size=config['test_size'], 
                random_state=config['random_seed'],
                stratify=hospital_y
            )
        else:
            train_idx = indices[:n_train]
            test_idx = indices[n_train:]
        
        train_indices.extend(train_idx)
        test_indices.extend(test_idx)
    
    # Create train/test data
    X_train = all_X[train_indices]
    y_train = all_y[train_indices]
    X_test = all_X[test_indices]
    y_test = all_y[test_indices]
    
    train_indices, test_indices = [], []
    
    for hospital_id, indices in hospital_to_indices.items():
        hospital_X = all_X[indices]
        hospital_y = all_y[indices]
        
        # Split this hospital's data
        n_samples = len(indices)
        n_train = int(n_samples * (1 - config['test_size'])) # ?
        
        # Stratified split if possible
        if len(np.unique(hospital_y)) > 1:
            train_idx, test_idx = train_test_split(
                indices, test_size=config['test_size'], 
                random_state=config['random_seed'],
                stratify=hospital_y
            )
        else:
            train_idx = indices[:n_train]
            test_idx = indices[n_train:]
        
        train_indices.extend(train_idx)
        test_indices.extend(test_idx)
    
    X_train = all_X[train_indices]
    y_train = all_y[train_indices]
    X_test = all_X[test_indices]
    y_test = all_y[test_indices]
    
    train_ds = eICUDataset(X_train, y_train)
    test_ds = eICUDataset(X_test, y_test)
    
    # Store metadata
    train_ds.hospital_to_indices = {}
    test_ds.hospital_to_indices = {}
    
    # Update hospital_to_indices for train/test split
    for hospital_id in sorted_hospitals:
        train_ds.hospital_to_indices[hospital_id] = [i for i, idx in enumerate(train_indices) 
                                                     if idx in hospital_to_indices[hospital_id]]
        test_ds.hospital_to_indices[hospital_id] = [i for i, idx in enumerate(test_indices) 
                                                    if idx in hospital_to_indices[hospital_id]]
    
    # Add metadata from preprocessing
    train_ds.metadata = preprocessed_data['metadata']
    test_ds.metadata = preprocessed_data['metadata']
    
    logging.info(f"Created eICU datasets: {len(train_ds)} train, {len(test_ds)} test samples")
    logging.info(f"Number of hospitals: {len(sorted_hospitals)}")
    
    return train_ds, test_ds

def partition_eicu_data(train_ds, test_ds, client_number, args = None):
    '''
    Partition eICU data by hospitals
    Note: for eICU data, each hospital is a client
    
    Returns:
        -   client_dataidx_map: Mapping of client_id to data indices
        -   train_cls_counts: Class distribution per client
    
    '''
    hospital_to_indices = train_ds.hospital_to_indices
    sorted_hospitals = sorted(hospital_to_indices.keys())

    # more hospitals than clients, subset
    if len(sorted_hospitals) > client_number:
        selected_hospitals = sorted_hospitals[:client_number]
        logging.info(f"Selected {client_number} hospitals out of {len(sorted_hospitals)}")
    else:
        selected_hospitals = sorted_hospitals
        logging.info(f"Using all {len(selected_hospitals)} hospitals as clients")
    
    client_dataidx_map = {}
    for client_id, hospital_id in enumerate(selected_hospitals):
        client_dataidx_map[client_id] = np.array(hospital_to_indices[hospital_id])
        
    # Calculate class distribution
    train_cls_counts = {}
    y_train = np.array(train_ds.targets)
    
    for client_id, indices in client_dataidx_map.items():
        client_y = y_train[indices]
        unique, counts = np.unique(client_y, return_counts=True)
        train_cls_counts[client_id] = {int(u): int(c) for u, c in zip(unique, counts)}
        
        # missing classes with count 0
        for c in range(2):  # Binary cls
            if c not in train_cls_counts[client_id]:
                train_cls_counts[client_id][c] = 0
    
    return client_dataidx_map, train_cls_counts    
    

# def get_eicu_data_info(args) -> Dict[str, Any]:
#     """
#     Get metadata about eICU dataset
    
#     Returns:
#         Dictionary with dataset information
#     """
#     # This would be called to get dataset info without loading full data
#     config = {
#         'prediction_task': getattr(args, 'prediction_task', 'mortality'),
#         'target_hospitals': getattr(args, 'target_hospitals',
#             [167, 420, 199, 458, 252, 165, 148, 281, 449, 283])
#     }
    
#     return {
#         'dataset_name': 'eicu',
#         'prediction_task': config['prediction_task'],
#         'num_hospitals': len(config['target_hospitals']),
#         'tasks_supported': ['mortality', 'ventilator', 'sepsis'],
#         'feature_types': ['drug_binary', 'demographics'],
#         'preprocessing_method': 'fedweight_compatible'
#     }