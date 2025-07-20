import argparse
import logging
import os
import socket
import sys
import numpy as np
import torch

# Add FedFed root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../../../")))
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../../")))

from utils.logger import logging_config
from configs import get_cfg
from algorithms_standalone.fedavg.FedAVGManager import FedAVGManager
from utils.set import set_random_seed

def add_args(parser):
    """
    Add medical-specific arguments
    """
    parser.add_argument("--config_file", default="configs/eicu_config.yaml", type=str)
    parser.add_argument("--medical_task", default="death", type=str,
                       choices=['death', 'ventilation', 'sepsis'],
                       help="Medical prediction task")
    parser.add_argument("opts", help="Modify config options using the command-line",
                       default=None, nargs=argparse.REMAINDER)
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser()
    args = add_args(parser)
    
    # Print args to see what it contains
    print("=== ARGS CONTENTS ===")
    print(f"args.config_file: {args.config_file}")
    print(f"args.medical_task: {args.medical_task}")
    print(f"args.opts: {args.opts}")
    print(f"All args attributes: {vars(args)}")
    print("=====================")
    
    # Setup configuration
    cfg = get_cfg()
    cfg.setup(args)
    
    # Override with medical task
    if hasattr(args, 'medical_task'):
        print("success")
        cfg.medical_task = args.medical_task
    
    # Force standalone mode
    cfg.mode = 'standalone'
    cfg.server_index = -1
    cfg.client_index = -1
    
    # Setup logging
    process_id = 0
    logging_config(args=cfg, process_id=process_id)
    
    hostname = socket.gethostname()
    logging.info("#############process ID = " + str(process_id) +
                ", host name = " + hostname + "########" +
                ", process ID = " + str(os.getpid()))
    
    # Show configuration
    logging.info("Configuration:")
    logging.info(dict(cfg))
    
    # Set random seed
    set_random_seed(cfg.seed)
    torch.backends.cudnn.deterministic = True
    
    # Setup device
    device = torch.device("cuda:" + str(cfg.gpu_index) if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")
    
    # Setup wandb if needed
    if cfg.record_tool == 'wandb' and cfg.wandb_record:
        import wandb
        wandb.init(config=dict(cfg), name=f'FedFed-eICU-{cfg.medical_task}',
                  project='FedFed-Medical')
    else:
        os.environ['WANDB_MODE'] = 'dryrun'
    
    # Run FedFed
    if cfg.algorithm == 'FedAvg':
        logging.info("Starting FedAvg with FedFed for medical data")
        fedavg_manager = FedAVGManager(device, cfg)
        fedavg_manager.train()
    else:
        raise NotImplementedError(f"Algorithm {cfg.algorithm} not implemented for medical data")