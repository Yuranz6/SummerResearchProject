import logging
import random
import math
import functools
import os

import numpy as np
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
from torchvision.datasets import (
    CIFAR10,
    CIFAR100,
    SVHN,
    FashionMNIST,
    MNIST,
)

from .cifar10.datasets import CIFAR10_truncated_WO_reload
from .cifar100.datasets import CIFAR100_truncated_WO_reload
from .SVHN.datasets import SVHN_truncated_WO_reload
from .FashionMNIST.datasets import FashionMNIST_truncated_WO_reload
from .eicu.datasets import eICU_Medical_Dataset, eICU_Medical_Dataset_truncated_WO_reload

from .cifar10.datasets import data_transforms_cifar10
from .cifar100.datasets import data_transforms_cifar100
from .SVHN.datasets import data_transforms_SVHN
from .FashionMNIST.datasets import data_transforms_fmnist
from .eicu.datasets import data_transforms_eicu

from data_preprocessing.utils.stats import record_net_data_stats

NORMAL_DATASET_LIST = ["cifar10", "cifar100", "SVHN", "mnist", "fmnist", "femnist-digit", "Tiny-ImageNet-200", "eicu"]

class Data_Loader(object):

    full_data_obj_dict = {
        "cifar10": CIFAR10,
        "cifar100": CIFAR100,
        "SVHN": SVHN,
        "fmnist": FashionMNIST,
        "eicu": eICU_Medical_Dataset,  
    }
    
    sub_data_obj_dict = {
        "cifar10": CIFAR10_truncated_WO_reload,
        "cifar100": CIFAR100_truncated_WO_reload,
        "SVHN": SVHN_truncated_WO_reload,
        "fmnist": FashionMNIST_truncated_WO_reload,
        "eicu": eICU_Medical_Dataset_truncated_WO_reload,  
    }

    transform_dict = {
        "cifar10": data_transforms_cifar10,
        "cifar100": data_transforms_cifar100,
        "SVHN": data_transforms_SVHN,
        "fmnist": data_transforms_fmnist,
        "eicu": data_transforms_eicu,  
    }

    num_classes_dict = {
        "cifar10": 10,
        "cifar100": 100,
        "SVHN": 10,
        "fmnist": 10,
        "eicu": 2,  
    }

    image_resolution_dict = {
        "cifar10": 32,
        "cifar100": 32,
        "SVHN": 32,
        "fmnist": 32,
        "eicu": 1,  
    }

    def __init__(self, args=None, process_id=0, mode="centralized", task="centralized",
                data_efficient_load=True, dirichlet_balance=False, dirichlet_min_p=None,
                dataset="", datadir="./", partition_method="hetero", partition_alpha=0.5, client_number=1, batch_size=128, num_workers=4,
                data_sampler=None,
                resize=32, augmentation="default", other_params={}):

        self.args = args

        # For partition
        self.process_id = process_id
        self.mode = mode
        self.task = task
        self.data_efficient_load = data_efficient_load 
        self.dirichlet_balance = dirichlet_balance
        self.dirichlet_min_p = dirichlet_min_p

        self.dataset = dataset
        self.datadir = datadir
        self.partition_method = partition_method
        self.partition_alpha = partition_alpha
        self.client_number = client_number
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.data_sampler = data_sampler

        self.augmentation = augmentation
        self.other_params = other_params

        # For image
        self.resize = resize

        self.init_dataset_obj()

    def load_data(self):
        if self.dataset == "eicu":
            self.federated_medical_split()
        else:
            self.federated_standalone_split()
            
        self.other_params["train_cls_local_counts_dict"] = self.train_cls_local_counts_dict
        self.other_params["client_dataidx_map"] = self.client_dataidx_map

        return self.train_data_global_num, self.test_data_global_num, self.train_data_global_dl, self.test_data_global_dl, \
               self.train_data_local_num_dict, self.test_data_local_num_dict, self.test_data_local_dl_dict, self.train_data_local_ori_dict,self.train_targets_local_ori_dict,\
               self.class_num, self.other_params

    def init_dataset_obj(self):
        self.full_data_obj = Data_Loader.full_data_obj_dict[self.dataset]
        self.sub_data_obj = Data_Loader.sub_data_obj_dict[self.dataset]
        logging.info(f"dataset augmentation: {self.augmentation}, resize: {self.resize}")
        self.transform_func = Data_Loader.transform_dict[self.dataset]
        self.class_num = Data_Loader.num_classes_dict[self.dataset]
        self.image_resolution = Data_Loader.image_resolution_dict[self.dataset]

    def get_transform(self, resize, augmentation, dataset_type, image_resolution=32):
        MEAN, STD, train_transform, test_transform = \
            self.transform_func(
                resize=resize, augmentation=augmentation, dataset_type=dataset_type, image_resolution=image_resolution)
        return MEAN, STD, train_transform, test_transform

    def load_full_data(self):
        if self.dataset == "eicu":
            # no image transforms needed
            MEAN, STD, train_transform, test_transform = 0.0, 1.0, None, None
            
            if hasattr(self.args, 'medical_task'):
                self.full_data_obj.task = self.args.medical_task
                
            train_ds = self.full_data_obj(self.datadir, train=True, download=False, transform=train_transform)
            test_ds = self.full_data_obj(self.datadir, train=False, download=False, transform=test_transform)
            
            return train_ds, test_ds
        else:
            return self.load_image_data()
    
    def load_image_data(self):
        """Load image datasets (CIFAR, SVHN, FMNIST) - Original implementation"""
        MEAN, STD, train_transform, test_transform = self.get_transform(
            self.resize, self.augmentation, "full_dataset", self.image_resolution)

        logging.debug(f"Train_transform is {train_transform} Test_transform is {test_transform}")
        if self.dataset == "SVHN":
            train_ds = self.full_data_obj(self.datadir,  "train", download=True, transform=train_transform, target_transform=None)
            test_ds = self.full_data_obj(self.datadir,  "test", download=True, transform=test_transform, target_transform=None)
            train_ds.data = train_ds.data.transpose((0,2,3,1))
        else:
            train_ds = self.full_data_obj(self.datadir,  train=True, download=True, transform=train_transform)
            test_ds = self.full_data_obj(self.datadir,  train=False, download=True, transform=test_transform)
        
        return train_ds, test_ds

    def load_sub_data(self, client_index, train_ds, test_ds):
        dataidxs = self.client_dataidx_map[client_index]
        train_data_local_num = len(dataidxs)

        MEAN, STD, train_transform, test_transform = self.get_transform(
            self.resize, self.augmentation, "sub_dataset", self.image_resolution)

        logging.debug(f"Train_transform is {train_transform} Test_transform is {test_transform}")
        train_ds_local = self.sub_data_obj(self.datadir, dataidxs=dataidxs, train=True, transform=train_transform,
                full_dataset=train_ds)

        if self.dataset == "eicu":
            train_ori_data = train_ds_local.data.numpy() if isinstance(train_ds_local.data, torch.Tensor) else train_ds_local.data
            train_ori_targets = train_ds_local.targets.numpy() if isinstance(train_ds_local.targets, torch.Tensor) else train_ds_local.targets
        else:
            # For image data, get as numpy arrays in [0, 255] range
            train_ori_data = np.array(train_ds_local.data)
            train_ori_targets = np.array(train_ds_local.targets)
            
        test_ds_local = self.sub_data_obj(self.datadir, train=False, transform=test_transform,
                        full_dataset=test_ds)   

        test_data_local_num = len(test_ds_local)
        return train_ds_local, test_ds_local, train_ori_data, train_ori_targets, train_data_local_num, test_data_local_num

    def get_dataloader(self, train_ds, test_ds, shuffle=True, drop_last=False, train_sampler=None, num_workers=1):
        logging.info(f"shuffle: {shuffle}, drop_last:{drop_last}, train_sampler:{train_sampler} ")
        train_dl = data.DataLoader(dataset=train_ds, batch_size=self.batch_size, shuffle=shuffle,
                                drop_last=drop_last, sampler=train_sampler, num_workers=num_workers)
        test_dl = data.DataLoader(dataset=test_ds, batch_size=self.batch_size, shuffle=True,
                                drop_last=False, num_workers=num_workers)
        
        return train_dl, test_dl

    def get_y_train_np(self, train_ds):
        """Extract labels as numpy array from dataset"""
        if self.dataset == "eicu":
            y_train = train_ds.targets
            y_train_np = np.array(y_train)
        elif self.dataset in ["fmnist"]:
            y_train = train_ds.targets.data
            y_train_np = np.array(y_train)
        elif self.dataset in ["SVHN"]:
            y_train = train_ds.labels
            y_train_np = np.array(y_train)
        else:
            y_train = train_ds.targets
            y_train_np = np.array(y_train)
        return y_train_np

    def federated_medical_split(self):
        """hospital-based partitioning"""
        train_ds, test_ds = self.load_full_data()
        y_train_np = self.get_y_train_np(train_ds)
        
        self.train_data_global_num = y_train_np.shape[0]
        self.test_data_global_num = len(test_ds)
        
        from .eicu.data_loader import partition_eicu_data_by_hospital
        self.client_dataidx_map, self.train_cls_local_counts_dict = partition_eicu_data_by_hospital(
            train_ds, self.client_number
        )
        
        logging.info("train_cls_local_counts_dict = " + str(self.train_cls_local_counts_dict))
        
        # global data loaders
        self.train_data_global_dl, self.test_data_global_dl = self.get_dataloader(
            train_ds, test_ds,
            shuffle=True, drop_last=False, train_sampler=None, num_workers=self.num_workers
        )
        
        self.train_data_local_num_dict = dict()
        self.test_data_local_num_dict = dict()
        self.train_data_local_ori_dict = dict()
        self.train_targets_local_ori_dict = dict()
        self.test_data_local_dl_dict = dict()
        
        # Create local data for each client
        for client_index in range(self.client_number):
            train_data_local, test_data_local, train_ori_data, train_ori_targets, train_data_local_num, test_data_local_num = self.load_sub_data(client_index, train_ds, test_ds)
            
            train_data_local_dl, test_data_local_dl = self.get_dataloader(
                train_data_local, test_data_local,
                shuffle=True, drop_last=False, train_sampler=None, num_workers=self.num_workers
            )
            
            self.train_data_local_num_dict[client_index] = train_data_local_num
            self.test_data_local_num_dict[client_index] = test_data_local_num
            self.test_data_local_dl_dict[client_index] = test_data_local_dl
            self.train_data_local_ori_dict[client_index] = train_ori_data
            self.train_targets_local_ori_dict[client_index] = train_ori_targets
        
        return self.train_data_local_num_dict, self.train_cls_local_counts_dict


    def federated_standalone_split(self):
        """Original federated split for image datasets"""
        train_ds, test_ds = self.load_full_data()
        y_train_np = self.get_y_train_np(train_ds)

        self.train_data_global_num = y_train_np.shape[0]
        self.test_data_global_num = len(test_ds)

        self.client_dataidx_map, self.train_cls_local_counts_dict = self.partition_data(y_train_np, self.train_data_global_num)

        logging.info("train_cls_local_counts_dict = " + str(self.train_cls_local_counts_dict))

        self.train_data_global_dl, self.test_data_global_dl = self.get_dataloader(
                train_ds, test_ds,   
                shuffle=True, drop_last=False, train_sampler=None, num_workers=self.num_workers)
        logging.info("train_dl_global number = " + str(len(self.train_data_global_dl)))
        logging.info("test_dl_global number = " + str(len(self.test_data_global_dl)))

        self.train_data_local_num_dict = dict()  
        self.test_data_local_num_dict = dict()
        self.train_data_local_ori_dict = dict()
        self.train_targets_local_ori_dict = dict()
        self.test_data_local_dl_dict = dict()

        for client_index in range(self.client_number):
            train_ds_local, test_ds_local, train_ori_data, train_ori_targets, \
            train_data_local_num, test_data_local_num = self.load_sub_data(client_index, train_ds, test_ds)

            self.train_data_local_num_dict[client_index] = train_data_local_num
            self.test_data_local_num_dict[client_index] = test_data_local_num
            logging.info("client_ID = %d, local_train_sample_number = %d, local_test_sample_number = %d" % \
                         (client_index, train_data_local_num, test_data_local_num))

            train_data_local_dl, test_data_local_dl = self.get_dataloader(train_ds_local, test_ds_local,
                                                                          shuffle=True, drop_last=False, num_workers=self.num_workers)
            logging.info("client_index = %d, batch_num_train_local = %d, batch_num_test_local = %d" % (
                client_index, len(train_data_local_dl), len(test_data_local_dl)))

            self.test_data_local_dl_dict[client_index] = test_data_local_dl
            self.train_data_local_ori_dict[client_index] = train_ori_data
            self.train_targets_local_ori_dict[client_index] = train_ori_targets
            self.test_data_local_dl_dict[client_index] = test_data_local_dl

    # centralized loading
    def load_centralized_data(self):
        self.train_ds, self.test_ds = self.load_full_data()
        self.train_data_num = len(self.train_ds)
        self.test_data_num = len(self.test_ds)
        self.train_dl, self.test_dl = self.get_dataloader(
                self.train_ds, self.test_ds,
                shuffle=True, drop_last=False, train_sampler=None, num_workers=self.num_workers)

    def partition_data(self, y_train_np, train_data_num):
        """Data partitioning for image datasets (LDA, etc.) - Placeholder for original implementation"""
        if self.partition_method == "hetero":
            return self.partition_data_lda(y_train_np, train_data_num)
        elif self.partition_method == "homo":
            return self.partition_data_homo(y_train_np, train_data_num)
        else:
            raise ValueError(f"Unsupported partition method: {self.partition_method}")
    
    def partition_data_lda(self, y_train_np, train_data_num):
        ''' refer to original implementation'''
        pass
    
    def partition_data_homo(self, y_train_np, train_data_num):
        """Homogeneous partitioning """
        pass