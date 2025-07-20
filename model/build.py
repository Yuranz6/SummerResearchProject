

from data_preprocessing.utils.stats import get_dataset_image_size
import logging

from model.cv.resnet_v2 import ResNet18, ResNet34, ResNet50, ResNet10
from model.cv.others import (ModerateCNNMNIST, ModerateCNN)
from model.FL_VAE import *

# Import medical models
from model.FL_VAE import FL_CVAE_Medical
from model.tabular.models import Medical_MLP_Classifier


CV_MODEL_LIST = []
RNN_MODEL_LIST = ["rnn"]
MEDICAL_MODEL_LIST = ["medical_mlp"]  


def create_model(args, model_name, output_dim, pretrained=False, device=None, **kwargs):
    """
    Create model based on model_name and dataset type
    
    This function maintains the exact interface from the original implementation
    while adding support for medical models for tabular federated learning.
    
    Args:
        args: Configuration arguments containing dataset and model parameters
        model_name: String identifier for model type
        output_dim: Number of output dimensions/classes
        pretrained: Whether to use pretrained weights (not applicable for medical models)
        device: Device to place model on
        **kwargs: Additional arguments for model initialization
    
    Returns:
        torch.nn.Module: Instantiated model ready for training
    """
    model = None
    logging.info(f"model name: {model_name}")

    is_medical_dataset = getattr(args, 'dataset', '') == 'eicu'
    
    if model_name in RNN_MODEL_LIST:
        pass
    elif model_name in MEDICAL_MODEL_LIST or is_medical_dataset:
        model = _create_medical_model(args, model_name, output_dim, device, **kwargs)
    else:
        image_size = get_dataset_image_size(args.dataset)
        model = _create_cv_model(args, model_name, output_dim, image_size, device, **kwargs)

    if model is None:
        raise NotImplementedError(f"Model {model_name} not implemented for dataset {args.dataset}")

    return model


def _create_medical_model(args, model_name, output_dim, device, **kwargs):
    """
    Create medical models for tabular federated learning
    
    This function handles the creation of medical-specific models that process
    tabular data rather than images. It supports both single-task and multi-task
    medical prediction scenarios.
    
    Args:
        args: Configuration arguments
        model_name: Medical model identifier
        output_dim: Number of output classes (typically 1 for binary medical tasks)
        device: Device for model placement
        **kwargs: Additional model arguments
    
    Returns:
        torch.nn.Module: Medical model instance
    """
    input_dim = getattr(args, 'VAE_input_dim', 256)  
    
    if model_name == "medical_mlp" or args.dataset == 'eicu':
        logging.info(f"Creating Medical_MLP_Classifier: input_dim={input_dim}, output_dim={output_dim}")
        
        if input_dim >= 256:
            hidden_dims = [128, 64]
        elif input_dim >= 128:
            hidden_dims = [64, 32]
        else:
            hidden_dims = [32, 16]
        
        model = Medical_MLP_Classifier(
            input_dim=input_dim,
            num_classes=output_dim,
            hidden_dims=hidden_dims,
            dropout_rate=getattr(args, 'dropout_rate', 0.2),
            use_batch_norm=getattr(args, 'use_batch_norm', True)
        )
        
    else:
        raise NotImplementedError(f"Medical model {model_name} not implemented")
    
    num_params = sum(param.numel() for param in model.parameters())
    logging.info(f"Medical model created with {num_params} parameters")
    
    return model


def _create_cv_model(args, model_name, output_dim, image_size, device, **kwargs):
    """
    Create computer vision models (original implementation preserved)
    
    This function maintains the exact original implementation for creating
    computer vision models, ensuring backward compatibility with existing
    FedFed experiments.
    
    Args:
        args: Configuration arguments
        model_name: CV model identifier
        output_dim: Number of output classes
        image_size: Input image size
        device: Device for model placement
        **kwargs: Additional model arguments
    
    Returns:
        torch.nn.Module: Computer vision model instance
    """
    model = None
    
    if model_name == "vgg-9":
        if args.dataset in ("mnist", 'femnist', 'fmnist'):
            model = ModerateCNNMNIST(output_dim=output_dim,
                                   input_channels=args.model_input_channels)
        elif args.dataset in ("cifar10", "cifar100", "cinic10", "svhn"):
            model = ModerateCNN(args, output_dim=output_dim)
            logging.info("------------------params number-----------------------")
            num_params = sum(param.numel() for param in model.parameters())
            logging.info(f"VGG-9 parameters: {num_params}")
            
    elif model_name == "resnet18_v2":
        logging.info("ResNet18_v2")
        model = ResNet18(args=args, num_classes=output_dim, image_size=image_size,
                        model_input_channels=args.model_input_channels)
                        
    elif model_name == "resnet34_v2":
        logging.info("ResNet34_v2")
        model = ResNet34(args=args, num_classes=output_dim, image_size=image_size,
                        model_input_channels=args.model_input_channels, device=device)
                        
    elif model_name == "resnet50_v2":
        model = ResNet50(args=args, num_classes=output_dim, image_size=image_size,
                        model_input_channels=args.model_input_channels)
                        
    elif model_name == "resnet10_v2":
        logging.info("ResNet10_v2")
        model = ResNet10(args=args, num_classes=output_dim, image_size=image_size,
                        model_input_channels=args.model_input_channels, device=device)
    
    return model

def create_vae_model(args, device):

    if args.dataset == 'eicu':
        logging.info("Creating FL_CVAE_Medical for tabular feature distillation")
        from model.FL_VAE import FL_CVAE_Medical
        
        vae_model = FL_CVAE_Medical(
            args=args,
            d=args.VAE_d,
            z=args.VAE_z,
            device=device,
            with_classifier=True
        )
    else:
        logging.info("Creating FL_CVAE_cifar for image feature distillation")
        from model.FL_VAE import FL_CVAE_cifar
        
        vae_model = FL_CVAE_cifar(
            args=args,
            d=args.VAE_d,
            z=args.VAE_z,
            device=device,
            with_classifier=True
        )
    
    return vae_model

def get_model_info(model):
    """
    Get information about a model for logging and debugging
    
    Args:
        model: PyTorch model
    
    Returns:
        dict: Model information including parameter count and architecture type
    """
    num_params = sum(param.numel() for param in model.parameters())
    num_trainable = sum(param.numel() for param in model.parameters() if param.requires_grad)
    
    model_type = "unknown"
    if isinstance(model, (ResNet18, ResNet34, ResNet50, ResNet10)):
        model_type = "computer_vision_resnet"
    elif isinstance(model, (ModerateCNN, ModerateCNNMNIST)):
        model_type = "computer_vision_cnn"
    elif isinstance(model, Medical_MLP_Classifier):
        model_type = "medical_single_task"
    elif isinstance(model, FL_CVAE_Medical):
        model_type = "medical_vae"
    elif isinstance(model, FL_CVAE_cifar):
        model_type = "image_vae"
    
    return {
        "model_type": model_type,
        "total_parameters": num_params,
        "trainable_parameters": num_trainable,
        "model_class": model.__class__.__name__
    }