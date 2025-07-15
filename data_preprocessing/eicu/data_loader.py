import logging
import os
import numpy as np
import pandas as pd
import torch
import torch.utils.data as data
from sklearn.model_selection import train_test_split

from .datasets import eICU_Medical_Dataset, eICU_Medical_Dataset_truncated_WO_reload, data_transforms_eicu

def load_eicu_medical_data(datadir, args=None, resize=None, augmentation="default"):
    """
    Load eICU medical data 
    
    This function loads the preprocessed eICU data and creates train/test datasets
    following the same pattern as CIFAR10 data loading.
    
    Args:
        datadir: Directory containing eicu_harmonized.csv
        args: Arguments object containing medical task and other parameters
        resize: Not used for medical data (kept for interface compatibility)  
        augmentation: Augmentation type for medical data
    
    Returns:
        tuple: (train_dataset, test_dataset) - Medical datasets 
    """
    
    task = 'death'  
    if args and hasattr(args, 'medical_task'):
        task = args.medical_task
    elif args and hasattr(args, 'task'):
        if args.task in ['death', 'ventilation', 'sepsis']:
            task = args.task
    
    logging.info(f"Loading eICU medical data for task: {task}")
    
    # mote: The eICU_Medical_Dataset handles train/test splitting internally
    medical_train_ds = eICU_Medical_Dataset(
        root=datadir, train=True, transform=None, target_transform=None, task=task
    )
    medical_test_ds = eICU_Medical_Dataset(
        root=datadir, train=False, transform=None, target_transform=None, task=task  
    )
    
    logging.info(f"Medical data loaded - Train: {len(medical_train_ds)} samples, "
                f"Test: {len(medical_test_ds)} samples")
    
    return medical_train_ds, medical_test_ds


def partition_medical_data_by_hospital(train_dataset, client_number, min_samples_per_hospital=10):
    """
    Partition medical data by hospital for federated learning
    
    This function implements hospital-based partitioning where each hospital
    becomes a federated learning client. This replaces artificial partitioning
    methods like LDA with natural institutional boundaries.
    
    Args:
        train_dataset: Medical training dataset with hospital_ids
        client_number: Number of federated learning clients to create
        min_samples_per_hospital: Minimum samples required per hospital to be included
    
    Returns:
        tuple: (client_dataidx_map, train_cls_local_counts_dict)
            - client_dataidx_map: Dict mapping client_id to data indices
            - train_cls_local_counts_dict: Dict mapping client_id to class distribution
    """
    
    logging.info(f"Partitioning medical data by hospital for {client_number} clients")
    
    if not hasattr(train_dataset, 'hospital_ids'):
        raise ValueError("Medical dataset must have hospital_ids attribute for partitioning")
    
    hospital_ids = train_dataset.hospital_ids
    unique_hospitals = np.unique(hospital_ids)
    
    logging.info(f"Found {len(unique_hospitals)} unique hospitals: {unique_hospitals}")
    
    # Filter hospitals by minimum sample count
    valid_hospitals = []
    for hospital_id in unique_hospitals:
        hospital_mask = (hospital_ids == hospital_id)
        hospital_sample_count = np.sum(hospital_mask)
        
        if hospital_sample_count >= min_samples_per_hospital:
            valid_hospitals.append(hospital_id)
            logging.info(f"Hospital {hospital_id}: {hospital_sample_count} samples - INCLUDED")
        else:
            logging.info(f"Hospital {hospital_id}: {hospital_sample_count} samples - EXCLUDED "
                        f"(< {min_samples_per_hospital} minimum)")
    
    # Adjust client_number if have fewer valid hospitals
    if len(valid_hospitals) < client_number:
        logging.warning(f"Number of valid hospitals ({len(valid_hospitals)}) is less than "
                       f"requested clients ({client_number}). Using {len(valid_hospitals)} clients.")
        client_number = len(valid_hospitals)
    
    # Use the hospitals with the most samples if we need to select a subset
    if len(valid_hospitals) > client_number:
        # Count samples per hospital and select top hospitals
        hospital_sample_counts = {}
        for hospital_id in valid_hospitals:
            hospital_mask = (hospital_ids == hospital_id)
            hospital_sample_counts[hospital_id] = np.sum(hospital_mask)
        
        # Sort hospitals by sample count (descending) and take top client_number
        sorted_hospitals = sorted(hospital_sample_counts.items(), key=lambda x: x[1], reverse=True)
        selected_hospitals = [h[0] for h in sorted_hospitals[:client_number]]
        
        logging.info(f"Selected top {client_number} hospitals by sample count: {selected_hospitals}")
    else:
        selected_hospitals = valid_hospitals
    
    # Create client data index mapping
    client_dataidx_map = {}
    train_cls_local_counts_dict = {}
    
    for client_idx, hospital_id in enumerate(selected_hospitals):
        hospital_mask = (hospital_ids == hospital_id)
        hospital_indices = np.where(hospital_mask)[0]
        
        client_dataidx_map[client_idx] = hospital_indices
        
        # Calculate class distribution for this hospital
        hospital_labels = train_dataset.targets[hospital_mask]
        if isinstance(hospital_labels, torch.Tensor):
            hospital_labels = hospital_labels.numpy()
        
        unique_classes, class_counts = np.unique(hospital_labels, return_counts=True)
        train_cls_local_counts_dict[client_idx] = dict(zip(unique_classes.astype(int), class_counts))
        
        positive_rate = np.mean(hospital_labels)
        logging.info(f"Client {client_idx} (Hospital {hospital_id}): "
                    f"{len(hospital_indices)} samples, "
                    f"positive_rate={positive_rate:.3f}, "
                    f"class_distribution={train_cls_local_counts_dict[client_idx]}")
    
    return client_dataidx_map, train_cls_local_counts_dict


def get_medical_dataloader(datadir, train_bs, test_bs, dataidxs=None, 
                          resize=None, augmentation="default", args=None,
                          full_train_dataset=None, full_test_dataset=None):
    """
    Create DataLoaders for medical data - equivalent to get_dataloader_CIFAR10 （NOT USED?)
    
    Args:
        datadir: Data directory
        train_bs: Training batch size
        test_bs: Test batch size  
        dataidxs: Data indices for client subset (None = use all data)
        resize: Not used for medical data (kept for interface compatibility)
        augmentation: Augmentation type
        args: Arguments object
        full_train_dataset: Full training dataset to subset from
        full_test_dataset: Full test dataset
    
    Returns:
        tuple: (train_dataloader, test_dataloader)
    """
    
    # Get transforms for medical data
    train_transform, test_transform = data_transforms_eicu(
        resize=resize, augmentation=augmentation, dataset_type="sub_dataset"
    )
    
    if args and args.data_efficient_load:
        # Use truncated dataset that loads from full dataset (efficient loading)
        dl_obj = eICU_Medical_Dataset_truncated_WO_reload
        train_ds = dl_obj(datadir, dataidxs=dataidxs, train=True, transform=train_transform,
                         full_dataset=full_train_dataset)
        test_ds = dl_obj(datadir, train=False, transform=test_transform,
                        full_dataset=full_test_dataset)
    else:
        # Direct loading (less efficient but simpler)
        # Note: This would require implementing a different approach for dataidxs
        raise NotImplementedError("Non-efficient loading not implemented for medical data")
    
    # Handle batch size adjustment if dataset is smaller than batch size
    drop_last = True
    if train_bs > len(train_ds):
        drop_last = False
        logging.warning(f"Training batch size ({train_bs}) larger than dataset size ({len(train_ds)}). "
                       f"Setting drop_last=False")
    
    # Create DataLoaders
    train_dl = data.DataLoader(dataset=train_ds, batch_size=train_bs, shuffle=True, drop_last=drop_last)
    test_dl = data.DataLoader(dataset=test_ds, batch_size=test_bs, shuffle=False, drop_last=False)
    
    return train_dl, test_dl


def load_partition_data_eicu_medical(dataset, data_dir, partition_method, partition_alpha, 
                                    client_number, batch_size, args=None):
    """
    Load and partition eICU medical data for federated learning
    
    This is the main function that integrates medical data loading and partitioning,
    following the same pattern as load_partition_data_cifar10.
    
    Args:
        dataset: Dataset name (should be "eicu")
        data_dir: Directory containing eicu_harmonized.csv  
        partition_method: Partitioning method ("hospital" for medical data)
        partition_alpha: Not used for hospital-based partitioning
        client_number: Number of federated learning clients
        batch_size: Batch size for training
        args: Arguments object
    
    Returns:
        tuple: (X_train, y_train, X_test, y_test, net_dataidx_map, traindata_cls_counts, 
                medical_train_ds, medical_test_ds)
    """
    
    logging.info("********* Partitioning medical data ***************")
    
    # Load medical data
    medical_train_ds, medical_test_ds = load_eicu_medical_data(
        data_dir, args=args, augmentation="default"
    )
    
    # Extract data arrays for compatibility with existing interfaces
    X_train = medical_train_ds.data.numpy()
    y_train = medical_train_ds.targets.numpy()
    X_test = medical_test_ds.data.numpy()
    y_test = medical_test_ds.targets.numpy()
    
    n_train = X_train.shape[0]
    
    # Partition data based on method
    if partition_method == "hospital" or partition_method == "hetero":
        # Use hospital-based partitioning (natural for medical data)
        net_dataidx_map, traindata_cls_counts = partition_medical_data_by_hospital(
            medical_train_ds, client_number
        )
    elif partition_method == "homo":
        # Homogeneous partitioning (not recommended for medical data)
        logging.warning("Homogeneous partitioning not recommended for medical data")
        # Simple random partitioning
        total_num = n_train
        idxs = np.random.permutation(total_num)
        batch_idxs = np.array_split(idxs, client_number)
        net_dataidx_map = {i: batch_idxs[i] for i in range(client_number)}
        
        # Calculate class distributions
        traindata_cls_counts = {}
        for client_idx in range(client_number):
            local_labels = y_train[net_dataidx_map[client_idx]]
            unique, counts = np.unique(local_labels, return_counts=True)
            traindata_cls_counts[client_idx] = dict(zip(unique.astype(int), counts))
    else:
        raise ValueError(f"Unsupported partition method for medical data: {partition_method}")
    
    logging.info("Medical data partitioning completed")
    logging.info(f"traindata_cls_counts = {traindata_cls_counts}")
    
    return X_train, y_train, X_test, y_test, net_dataidx_map, traindata_cls_counts, \
           medical_train_ds, medical_test_ds


def record_medical_data_stats(y_train, net_dataidx_map, task='death'):
    """
    Record medical data statistics per client
    
    Args:
        y_train: Training labels
        net_dataidx_map: Mapping of client_id to data indices
        task: Medical prediction task
    
    Returns:
        dict: Statistics per client
    """
    net_cls_counts = {}
    
    for net_i, dataidx in net_dataidx_map.items():
        unq, unq_cnt = np.unique(y_train[dataidx], return_counts=True)
        tmp = {unq[i]: unq_cnt[i] for i in range(len(unq))}
        net_cls_counts[net_i] = tmp
        
        # Log additional medical-specific statistics
        positive_rate = np.mean(y_train[dataidx])
        logging.info(f"Client {net_i}: {len(dataidx)} samples, "
                    f"{task}_positive_rate={positive_rate:.3f}, "
                    f"class_distribution={tmp}")
    
    logging.debug(f'Medical data statistics: {str(net_cls_counts)}')
    return net_cls_counts