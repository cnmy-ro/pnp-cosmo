from pathlib import Path
import argparse

import numpy as np
import torch

from cosmo_systems import MUNIT
from data.nyudicom_t1wt2w_cosmo_dataset import NYUDicomT1WT2WCoSMoDataset
from utils import *



def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    return args


def main():
    
    # Load config
    args = get_args()
    conf = load_config(args.config)

    # Reproducibility
    np.random.seed(conf['seed'])
    torch.manual_seed(conf['seed'])

    # Logging
    wandb_init(conf)

    # Data
    train_dataset = NYUDicomT1WT2WCoSMoDataset(**conf['dataset'])
    train_dataloader = InfiniteDataLoader(train_dataset, batch_size=1, num_workers=2, shuffle=True)

    # CoSMo model
    model = MUNIT(conf, mode='train')
    if conf['load_checkpoint'] is not None: model, prev_it = load_checkpoint(model, conf); start_it = prev_it + 1        
    else:                                   start_it = 1

    # Training loop
    for it in range(start_it, conf['num_training_iters'] + 1):
        batch = next(iter(train_dataloader))
        model.set_input(batch)
        model.training_step()
        
        if it % conf['log_freq'] == 0:
            losses = model.get_losses()
            visuals = model.get_visuals()            
            print(it, {k: f"{v:.3f}" for k,v in losses.items()})
            # wandb_log_iter(it, losses, visuals, mode='train')
        
        if it % conf['val_freq'] == 0:
            save_checkpoint(model, it, conf)
        


# ---
# Run

if __name__ == '__main__':
    main()