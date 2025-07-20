import logging
import os
from collections import defaultdict
import numpy as np
import pandas as pd
import torch
import torch.utils.data as data
from sklearn.model_selection import train_test_split

from .datasets import eICU_Medical_Dataset, eICU_Medical_Dataset_truncated_WO_reload, data_transforms_eicu

def partition_eicu_data_by_hospital(dataset, client_num, min_samples_per_hospital=10):
    """
    Partition eICU data naturally by hospital IDs
    
    Args:
        dataset: eICU dataset with hospital_ids attribute
        client_num: number of FL clients needed
        min_samples_per_hospital: minimum samples required per hospital
        
    Returns:
        dict_users: {client_idx: array of data indices}
        dict_users_train_cls_counts: {client_idx: {class: count}}
    """
    
    hospital_to_indices = defaultdict(list)
    for idx in range(len(dataset)):
        hospital_id = dataset.hospital_ids[idx]
        hospital_to_indices[hospital_id].append(idx)
    
    # Filter hospitals 
    valid_hospitals = []
    for hospital_id, indices in hospital_to_indices.items():
        if len(indices) >= min_samples_per_hospital:
            valid_hospitals.append(hospital_id)
    
    logging.info(f"Found {len(valid_hospitals)} valid hospitals with >= {min_samples_per_hospital} samples")
    
    hospital_sizes = [(h, len(hospital_to_indices[h])) for h in valid_hospitals]
    hospital_sizes.sort(key=lambda x: x[1], reverse=True)
    
    # Select top hospitals
    if len(valid_hospitals) > client_num:
        selected_hospitals = [h for h, _ in hospital_sizes[:client_num]]
        logging.info(f"Selected top {client_num} hospitals by sample count")
    else:
        selected_hospitals = valid_hospitals
        if len(selected_hospitals) < client_num:
            logging.warning(f"Only {len(selected_hospitals)} hospitals available, "
                          f"but {client_num} clients requested. Will duplicate hospitals.")
    
    dict_users = {}
    dict_users_train_cls_counts = {}
    
    for client_idx in range(client_num):
        if client_idx < len(selected_hospitals):
            hospital_id = selected_hospitals[client_idx]
        else:
            # Duplicate hospitals in round-robin fashion
            hospital_id = selected_hospitals[client_idx % len(selected_hospitals)]
            
        indices = np.array(hospital_to_indices[hospital_id])
        dict_users[client_idx] = indices
        
        # class distribution
        targets = dataset.targets[indices]
        unique_classes, counts = np.unique(targets, return_counts=True)
        
        class_counts = {0: 0, 1: 0}
        for cls, count in zip(unique_classes, counts):
            class_counts[int(cls)] = int(count)
            
        dict_users_train_cls_counts[client_idx] = class_counts
        
        logging.info(f"Client {client_idx} (Hospital {hospital_id}): "
                    f"{len(indices)} samples, class distribution: {class_counts}")
    
    return dict_users, dict_users_train_cls_counts


def load_partition_eicu_medical(dataset, data_dir, partition_method, partition_alpha,
                               client_number, batch_size, logger, args=None):
    """
    Main entry point for loading and partitioning eICU data (same logic already implemented in loader.py)
    """
    
    if args and hasattr(args, 'medical_task'):
        eICU_Medical_Dataset.task = args.medical_task
    
    train_dataset = eICU_Medical_Dataset(data_dir, train=True, download=False)
    test_dataset = eICU_Medical_Dataset(data_dir, train=False, download=False)
    
    train_data_num = len(train_dataset)
    test_data_num = len(test_dataset)
    class_num = 2  # Binary classification
    
    # Partition data by hospital (natural partitioning for medical data)
    if partition_method == "hospital":
        dict_users, dict_users_train_cls_counts = partition_eicu_data_by_hospital(
            train_dataset, client_number
        )
    else:
        logging.warning(f"Partition method '{partition_method}' not optimal for medical data. "
                       f"Using random partitioning.")
        # random partitioning
        indices = np.arange(train_data_num)
        np.random.shuffle(indices)
        dict_users = {}
        dict_users_train_cls_counts = {}
        
        samples_per_client = train_data_num // client_number
        for i in range(client_number):
            start_idx = i * samples_per_client
            if i == client_number - 1:
                dict_users[i] = indices[start_idx:]
            else:
                dict_users[i] = indices[start_idx:start_idx + samples_per_client]
                
            targets = train_dataset.targets[dict_users[i]]
            class_counts = {0: 0, 1: 0}
            for cls in [0, 1]:
                class_counts[cls] = int(np.sum(targets == cls))
            dict_users_train_cls_counts[i] = class_counts
    
    # Create global data loaders
    train_data_global = data.DataLoader(
        train_dataset, batch_size=batch_size, 
        shuffle=True, drop_last=False
    )
    test_data_global = data.DataLoader(
        test_dataset, batch_size=batch_size,
        shuffle=False, drop_last=False
    )
    
    train_data_local_num_dict = {}
    train_data_local_dict = {}
    test_data_local_dict = {}
    train_data_local_cls_counts_dict = {}
    
    for client_idx in range(client_number):
        dataidxs = dict_users[client_idx]
        
        train_data_local_num_dict[client_idx] = len(dataidxs)
        
        train_data_local = eICU_Medical_truncated_WO_reload(
            data_dir, dataidxs=dataidxs, train=True,
            transform=None, full_dataset=train_dataset
        )
        
        # For test data, all clients use the same test set (following FedFed pattern)
        test_data_local = test_dataset
        
        train_data_local_dict[client_idx] = train_data_local
        test_data_local_dict[client_idx] = test_data_local
        train_data_local_cls_counts_dict[client_idx] = dict_users_train_cls_counts[client_idx]
    
    return (train_dataset, test_dataset, train_data_num, test_data_num,
            train_data_global, test_data_global, train_data_local_num_dict,
            train_data_local_dict, test_data_local_dict, class_num,
            train_data_local_cls_counts_dict)