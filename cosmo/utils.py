import sys
from pathlib import Path

import ruamel.yaml
import numpy as np
import torch
from torch.utils.data import DataLoader
import wandb

sys.path.append("//wsl.localhost/Ubuntu/home/csrao/git-personal/llmr")  # Workstation
from llmr.intensity import rescale_intensity



class InfiniteDataLoader(DataLoader):
    """
    Taken from:  https://gist.github.com/MFreidank/821cc87b012c53fade03b0c7aba13958
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Initialize an iterator over the dataset.
        self.dataset_iterator = super().__iter__()

    def __iter__(self):
        return self

    def __next__(self):
        try:
            batch = next(self.dataset_iterator)
        except StopIteration:
            # Dataset exhausted, use a new fresh iterator.
            self.dataset_iterator = super().__iter__()
            batch = next(self.dataset_iterator)
        return batch


def load_config(path):
    with open(path, 'r') as fs:
        conf = ruamel.yaml.YAML().load(fs)
    return dict(conf)


def dump_config(path, conf):
    with open(path, 'w') as fs:
        ruamel.yaml.YAML().dump(conf, fs)


def save_checkpoint(model, it, conf):

    print('Dropping checkpoint:', it)

    checkpoint_root, run_name = conf['checkpoint_root'], conf['run_name']
    checkpoint_dir = checkpoint_root / run_name
    checkpoint_dir.mkdir(exist_ok=True, parents=True)
    
    # Drop config
    dump_config(checkpoint_dir / Path('config.yaml'))

    # Drop checkpoint
    checkpoint_path = checkpoint_dir / Path(f"iter_{str(it).zfill(6)}.pt")
    checkpoint = {'conf': conf, 'iter_counter': it}
    checkpoint.update({f"net_{k}": v.state_dict() for k,v in model.networks.items()})
    checkpoint.update({f"opt_{k}": v.state_dict() for k,v in model.optimizers.items()})
    torch.save(checkpoint, checkpoint_path)


def load_checkpoint(model, conf):
    checkpoint_path = conf['load_from_checkpoint']
    checkpoint = torch.load(checkpoint_path)
    prev_it = checkpoint['iter_counter']

    for k in model.networks.keys():
        model.networks[k].load_state_dict(checkpoint[f'net_{k}'])
    for k in model.optimizers.keys():
        model.optimizers[k].load_state_dict(checkpoint[f'opt_{k}'])

    print(f"Continuing training from iteration {prev_it}")
    return model, prev_it


def wandb_init(conf):
    wandb.init(project=conf['prj_name'], name=conf['run_name'], config=conf)


def wandb_log_iter(iter_counter, losses=None, visuals=None, mode='train'):

        log_dict = {'Domain u': 1, 'Domain v': 2}

        if losses is not None:
            for k in losses.keys():
                log_dict[f'Train loss: {k}'] = float(losses[k].clone().detach().cpu().numpy())
        
        if visuals is not None:
            visuals_to_log = []
            for set_name in visuals.keys():
                visuals_set = visuals[set_name]
                
                # Skip visuals set if empty (typically during val)
                if list(visuals_set.keys()) == []:
                    continue

                visuals_keys = list(visuals_set.keys())         
                visuals_images = list(visuals_set.values())  
                visuals_images = [viz[0] for viz in visuals_images]  # Keeping only the first example of this batch. Shape (C,H,W)        
                visuals_label = ' - '.join(visuals_keys)

                # Cap image size to save on wandb storage
                # TODO

                # If num_channels=2, assume pseudo-complex and convert to abs
                if visuals_images[0].shape[0] == 2:
                    visuals_images = [torch.norm(viz, p=2, dim=0, keepdim=True) for viz in visuals_images]

                # Make grid
                visuals_images = [rescale_intensity(viz, to_range=(0, 1)) for viz in visuals_images]
                visuals_grid = torch.cat(visuals_images, dim=2).permute(1, 2, 0)  # (CHW) -> (HWC)
                visuals_grid = wandb.Image(visuals_grid.cpu().numpy(), caption=visuals_label)
                visuals_to_log.append(visuals_grid)

            log_dict[f'{mode.capitalize()} images'] = visuals_to_log
        
        wandb.log(log_dict, step=iter_counter)

