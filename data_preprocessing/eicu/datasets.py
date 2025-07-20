import torch
import torch.utils.data as data
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import logging
import os

def data_transforms_eicu(resize=32, augmentation="default", dataset_type="full_dataset", image_resolution=32):
    """
    For tabular data, return None transforms but maintain the return structure
    """
    MEAN = 0.0  
    STD = 1.0   
    
    train_transform = None
    test_transform = None
    
    return MEAN, STD, train_transform, test_transform


class eICU_Medical_Dataset(data.Dataset):
    """
    eICU Medical Dataset for medical tabular data
    """
    
    def __init__(self, root, train=True, transform=None, target_transform=None, 
                 task='death', download=False):
        self.root = root
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
        self.task = task
        
        # Load preprocessed data
        self.data, self.targets, self.hospital_ids = self._load_data()
        
    def _load_data(self):
        """Load the harmonized eICU data"""
        data_path = os.path.join(self.root, 'eicu_harmonized.csv')
        
        if not os.path.exists(data_path):
            raise FileNotFoundError(
                f"eICU harmonized data not found at {data_path}. "
                f"Please ensure drug_harmonization.py has been run."
            )
        
        logging.info(f"Loading eICU data from {data_path}")
        df = pd.read_csv(data_path)
        
        feature_cols = [col for col in df.columns if col not in 
                       ['patientunitstayid', 'hospitalid', 'death', 'ventilation', 'sepsis']]
        
        X = df[feature_cols].values.astype(np.float32)
        
        if self.task == 'ventilation':
            y = df['ventilation'].values.astype(np.float32)
        elif self.task == 'sepsis':
            y = df['sepsis'].values.astype(np.float32)
        else:  
            y = df['death'].values.astype(np.float32)
        
        hospital_ids = df['hospitalid'].values
        
        # Question: stratification needed?
        X_train, X_test, y_train, y_test, hospital_train, hospital_test = train_test_split(
            X, y, hospital_ids, test_size=0.2, random_state=42, stratify=y
        )
        
        if self.train:
            data = torch.FloatTensor(X_train)
            targets = torch.FloatTensor(y_train)
            hospital_ids = hospital_train
        else:
            data = torch.FloatTensor(X_test)
            targets = torch.FloatTensor(y_test)
            hospital_ids = hospital_test
        
        logging.info(f"Medical dataset ({'train' if self.train else 'test'}): "
                    f"{len(data)} samples, {data.shape[1]} features, task={self.task}, "
                    f"positive_rate={targets.mean():.3f}")
        
        return data, targets, hospital_ids
    
    def __getitem__(self, index):
        """Get a single sample"""
        features, target = self.data[index], self.targets[index]
        
        if self.transform is not None:
            features = self.transform(features)
        
        if self.target_transform is not None:
            target = self.target_transform(target)
        
        return features, target
    
    def __len__(self):
        return len(self.data)


class eICU_Medical_Dataset_truncated_WO_reload(data.Dataset):
    """Truncated medical dataset for federated learning clients"""
    
    def __init__(self, datadir, dataidxs=None, train=True, transform=None,
                 target_transform=None, full_dataset=None):
        self.datadir = datadir
        self.dataidxs = dataidxs
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
        
        if full_dataset is None:
            raise ValueError("full_dataset must be provided for efficient loading")
        
        # Extract subset based on indices
        if dataidxs is not None:
            self.data = full_dataset.data[dataidxs]
            self.targets = full_dataset.targets[dataidxs]
            if hasattr(full_dataset, 'hospital_ids'):
                self.hospital_ids = full_dataset.hospital_ids[dataidxs]
        else:
            self.data = full_dataset.data
            self.targets = full_dataset.targets
            if hasattr(full_dataset, 'hospital_ids'):
                self.hospital_ids = full_dataset.hospital_ids
    
    def __getitem__(self, index):
        """Get a single sample"""
        features, target = self.data[index], self.targets[index]
        
        if self.transform is not None:
            features = self.transform(features)
        
        if self.target_transform is not None:
            target = self.target_transform(target)
        
        return features, target
    
    def __len__(self):
        return len(self.data)
