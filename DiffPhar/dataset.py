import os
import numpy as np
import torch
from torch.utils.data import Dataset


class ProcessedLigandPharPocketDataset(Dataset):
    def __init__(self, npz_path, center=True):

        with np.load(npz_path, allow_pickle=True) as f:
            data = {key: val for key, val in f.items()}

        # split data based on mask
        self.data = {}
        for (k, v) in data.items():
            if k == 'names':
                self.data[k] = v
                continue

            sections = np.where(np.diff(data['phar_mask']))[0] + 1 \
                if 'phar' in k \
                else np.where(np.diff(data['pocket_mask']))[0] + 1
            self.data[k] = [torch.from_numpy(x) for x in np.split(v, sections)]

            # add number of nodes for convenience
            if k == 'phar_mask':
                self.data['num_phar_atoms'] = \
                    torch.tensor([len(x) for x in self.data['phar_mask']])
            elif k == 'pocket_mask':
                self.data['num_pocket_nodes'] = \
                    torch.tensor([len(x) for x in self.data['pocket_mask']])

        if center:
            for i in range(len(self.data['phar_coords'])):
                mean = (self.data['phar_coords'][i].sum(0) +
                        self.data['pocket_c_alpha'][i].sum(0)) / \
                       (len(self.data['phar_coords'][i]) + len(self.data['pocket_c_alpha'][i]))
                self.data['phar_coords'][i] = self.data['phar_coords'][i] - mean
                self.data['pocket_c_alpha'][i] = self.data['pocket_c_alpha'][i] - mean

    def __len__(self):
        return len(self.data['names'])

    def __getitem__(self, idx):
        return {key: val[idx] for key, val in self.data.items()}

    @staticmethod
    def collate_fn(batch):
        out = {}
        for prop in batch[0].keys():

            if prop == 'names':
                out[prop] = [x[prop] for x in batch]
            elif prop == 'num_phar_atoms' or prop == 'num_pocket_nodes':
                out[prop] = torch.tensor([x[prop] for x in batch])
            elif 'mask' in prop:
                # make sure indices in batch start at zero (needed for
                # torch_scatter)
                out[prop] = torch.cat([i * torch.ones(len(x[prop]))
                                       for i, x in enumerate(batch)], dim=0)
            else:
                out[prop] = torch.cat([x[prop] for x in batch], dim=0)

        return out
