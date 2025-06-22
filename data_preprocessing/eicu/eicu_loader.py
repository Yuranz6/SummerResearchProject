
import numpy as np
import torch
from typing import Dict, List, Tuple, Any
from torch.utils.data import Dataset
from .eicu_preprocessor import preprocess_eicu



class eICUDataset(Dataset):
    """
    PyTorch Dataset for eICU data - FedFed compatible
    
    IMPORTANT: Must match exactly what FedFed training loop expects!!! Double check!
    """
    
    def __init__(self, X: np.ndarray, y: np.ndarray):
        """
        Simple dataset compatible with FedFed trainer
        
        Args:
            X: Feature matrix [n_samples, n_features]
            y: Labels [n_samples]
        """
        self.X = torch.FloatTensor(X)
        self.y = torch.LongTensor(y)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def load_eicu_data(args) -> Dict[int, Dict]:
    """
    Load eICU data for federated learning
    
    Args:
        args: Configuration object with eICU-specific parameters
        
    Returns:
        Dictionary mapping client_id (hospital) to data splits
    """
    # Configuration for eICU preprocessing
    config = {
        'prediction_task': getattr(args, 'prediction_task', 'mortality'),
        'target_hospitals': getattr(args, 'target_hospitals', 
            [167, 420, 199, 458, 252, 165, 148, 281, 449, 283]),
        'test_size': getattr(args, 'test_size', 0.2),
        'val_size': getattr(args, 'val_size', 0.1),
        'random_seed': getattr(args, 'seed', 42)
    }
    
    # Get eICU demo database path
    demo_db_path = getattr(args, 'eicu_demo_path', './data/eicu_demo.db')
    
    print(f"Loading eICU data for {config['prediction_task']} prediction...")
    
    # Preprocess data 
    preprocessed_data = preprocess_eicu(demo_db_path, config)
    hospital_datasets = preprocessed_data['hospital_datasets']
    
    federated_data = {}
    
    print(f"Creating federated datasets for {len(hospital_datasets)} hospitals...")
    
    for hospital_id, dataset in hospital_datasets.items():
        X, y = dataset['X'], dataset['y']
        
        from sklearn.model_selection import train_test_split
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=config['test_size'], 
            random_state=config['random_seed'],
            stratify=y if len(np.unique(y)) > 1 else None
        )
        
        train_dataset = eICUDataset(X_train, y_train)
        test_dataset = eICUDataset(X_test, y_test)
        
        federated_data[hospital_id] = {
            'train_dataset': train_dataset,
            'test_dataset': test_dataset, 
            'num_train_samples': len(X_train),
            'num_test_samples': len(X_test)
        }
        
        print(f"  Hospital {hospital_id}: {len(X_train)} train, {len(X_test)} test samples")
    
    print(f"Successfully created federated data for {len(federated_data)} hospitals")
    
    return federated_data


def get_eicu_data_info(args) -> Dict[str, Any]:
    """
    Get metadata about eICU dataset
    
    Returns:
        Dictionary with dataset information
    """
    # This would be called to get dataset info without loading full data
    config = {
        'prediction_task': getattr(args, 'prediction_task', 'mortality'),
        'target_hospitals': getattr(args, 'target_hospitals',
            [167, 420, 199, 458, 252, 165, 148, 281, 449, 283])
    }
    
    return {
        'dataset_name': 'eicu',
        'prediction_task': config['prediction_task'],
        'num_hospitals': len(config['target_hospitals']),
        'tasks_supported': ['mortality', 'ventilator', 'sepsis'],
        'feature_types': ['drug_binary', 'demographics'],
        'preprocessing_method': 'fedweight_compatible'
    }