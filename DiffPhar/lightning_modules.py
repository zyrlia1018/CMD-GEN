import math
from argparse import Namespace
from typing import Optional
from time import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import pytorch_lightning as pl
import wandb
from torch_scatter import scatter_add, scatter_mean
from Bio.PDB import PDBParser
from Bio.PDB.Polypeptide import three_to_one

from constants import dataset_params, FLOAT_TYPE, INT_TYPE
from equivariant_diffusion.dynamics import EGNNDynamics
from equivariant_diffusion.en_diffusion import EnVariationalDiffusion
from equivariant_diffusion.conditional_model import ConditionalDDPM, \
    SimpleConditionalDDPM
from dataset import ProcessedLigandPharPocketDataset
import utils
from analysis.visualization import save_xyz_file, visualize, visualize_chain
from analysis.metrics import check_stability, BasicMolecularMetrics, \
    CategoricalDistribution
from analysis.molecule_builder import build_molecule, process_molecule


class PharPocketDDPM(pl.LightningModule):
    def __init__(
            self,
            outdir,
            dataset,
            datadir,
            batch_size,
            lr,
            egnn_params: Namespace,
            diffusion_params,
            num_workers,
            augment_noise,
            augment_rotation,
            clip_grad,
            eval_epochs,
            eval_params,
            mode,
            node_histogram,
            pocket_representation='CA',
    ):
        super(PharPocketDDPM, self).__init__()
        self.save_hyperparameters()

        ddpm_models = {'joint': EnVariationalDiffusion,
                       'pocket_conditioning': ConditionalDDPM,
                       'pocket_conditioning_simple': SimpleConditionalDDPM}
        assert mode in ddpm_models
        self.mode = mode
        assert pocket_representation in {'CA', 'full-atom'}
        self.pocket_representation = pocket_representation

        self.dataset_name = dataset
        self.datadir = datadir
        self.outdir = outdir
        self.batch_size = batch_size
        self.eval_batch_size = eval_params.eval_batch_size \
            if 'eval_batch_size' in eval_params else batch_size
        self.lr = lr
        self.loss_type = diffusion_params.diffusion_loss_type
        self.eval_epochs = eval_epochs
        self.eval_params = eval_params
        self.num_workers = num_workers
        self.augment_noise = augment_noise
        self.augment_rotation = augment_rotation
        self.dataset_info = dataset_params[dataset]
        self.T = diffusion_params.diffusion_steps
        self.clip_grad = clip_grad
        if clip_grad:
            self.gradnorm_queue = utils.Queue()
            # Add large value that will be flushed.
            self.gradnorm_queue.add(3000)

        self.phar_type_distribution = CategoricalDistribution(
            self.dataset_info['phar_hist'], self.dataset_info['phar_encoder'])
        if self.pocket_representation == 'CA':
            self.pocket_type_distribution = CategoricalDistribution(
                self.dataset_info['aa_hist'], self.dataset_info['aa_encoder'])
        else:
            # TODO: full-atom case
            self.pocket_type_distribution = None

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

        self.phar_type_encoder = self.dataset_info['phar_encoder']
        self.phar_type_decoder = self.dataset_info['phar_decoder']
        self.pocket_type_encoder = self.dataset_info['aa_encoder'] \
            if self.pocket_representation == 'CA' \
            else self.dataset_info['atom_encoder']
        self.pocket_type_decoder = self.dataset_info['aa_decoder'] \
            if self.pocket_representation == 'CA' \
            else self.dataset_info['atom_decoder']

        self.phar_nf = len(self.phar_type_decoder)
        self.aa_nf = len(self.pocket_type_decoder)
        self.x_dims = 3

        net_dynamics = EGNNDynamics(
            phar_nf=self.phar_nf,
            residue_nf=self.aa_nf,
            n_dims=self.x_dims,
            joint_nf=egnn_params.joint_nf,
            device=egnn_params.device if torch.cuda.is_available() else 'cpu',
            hidden_nf=egnn_params.hidden_nf,
            act_fn=torch.nn.SiLU(),
            n_layers=egnn_params.n_layers,
            attention=egnn_params.attention,
            tanh=egnn_params.tanh,
            norm_constant=egnn_params.norm_constant,
            inv_sublayers=egnn_params.inv_sublayers,
            sin_embedding=egnn_params.sin_embedding,
            normalization_factor=egnn_params.normalization_factor,
            aggregation_method=egnn_params.aggregation_method,
            edge_cutoff=egnn_params.__dict__.get('edge_cutoff'),
            update_pocket_coords=(self.mode == 'joint')
        )

        self.ddpm = ddpm_models[self.mode](
                dynamics=net_dynamics,
                phar_nf=self.phar_nf,
                residue_nf=self.aa_nf,
                n_dims=self.x_dims,
                timesteps=diffusion_params.diffusion_steps,
                noise_schedule=diffusion_params.diffusion_noise_schedule,
                noise_precision=diffusion_params.diffusion_noise_precision,
                loss_type=diffusion_params.diffusion_loss_type,
                norm_values=diffusion_params.normalize_factors,
                size_histogram=node_histogram,
            )

    def configure_optimizers(self):
        return torch.optim.AdamW(self.ddpm.parameters(), lr=self.lr,
                                 amsgrad=True, weight_decay=1e-12)

    def setup(self, stage: Optional[str] = None):
        if stage == 'fit':
            self.train_dataset = ProcessedLigandPharPocketDataset(
                Path(self.datadir, 'train.npz'))
            self.val_dataset = ProcessedLigandPharPocketDataset(
                Path(self.datadir, 'val.npz'))
        elif stage == 'test':
            self.test_dataset = ProcessedLigandPharPocketDataset(
                Path(self.datadir, 'test.npz'))
        else:
            raise NotImplementedError

    def train_dataloader(self):
        return DataLoader(self.train_dataset, self.batch_size, shuffle=True,
                          num_workers=self.num_workers,
                          collate_fn=self.train_dataset.collate_fn)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, self.batch_size, shuffle=False,
                          num_workers=self.num_workers,
                          collate_fn=self.val_dataset.collate_fn)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, self.batch_size, shuffle=False,
                          num_workers=self.num_workers,
                          collate_fn=self.test_dataset.collate_fn)

    def get_phar_and_pocket(self, data):
        phar = {
            'x': data['phar_coords'].to(self.device, FLOAT_TYPE),
            'one_hot': data['phar_one_hot'].to(self.device, FLOAT_TYPE),
            'size': data['num_phar_atoms'].to(self.device, INT_TYPE),
            'mask': data['phar_mask'].to(self.device, INT_TYPE)
        }

        pocket = {
            'x': data['pocket_c_alpha'].to(self.device, FLOAT_TYPE),
            'one_hot': data['pocket_one_hot'].to(self.device, FLOAT_TYPE),
            'size': data['num_pocket_nodes'].to(self.device, INT_TYPE),
            'mask': data['pocket_mask'].to(self.device, INT_TYPE)
        }
        return phar, pocket

    def forward(self, data):
        phar, pocket = self.get_phar_and_pocket(data)

        # Note: \mathcal{L} terms in the paper represent log-likelihoods while
        # our loss terms are a negative(!) log-likelihoods
        delta_log_px, error_t_phar, error_t_pocket, SNR_weight, \
        loss_0_x_phar, loss_0_x_pocket, loss_0_h, neg_log_const_0, \
        kl_prior, log_pN, t_int, xh_phar_hat, info = \
            self.ddpm(phar, pocket, return_info=True)

        if self.loss_type == 'l2' and self.training:
            # normalize loss_t
            denom_phar = (self.x_dims + self.ddpm.phar_nf) * phar['size']
            error_t_phar = error_t_phar / denom_phar
            denom_pocket = (self.x_dims + self.ddpm.residue_nf) * pocket['size']
            error_t_pocket = error_t_pocket / denom_pocket
            loss_t = 0.5 * (error_t_phar + error_t_pocket)

            # normalize loss_0
            loss_0_x_phar = loss_0_x_phar / (self.x_dims * phar['size'])
            loss_0_x_pocket = loss_0_x_pocket / (self.x_dims * pocket['size'])
            loss_0 = loss_0_x_phar + loss_0_x_pocket + loss_0_h

        # VLB objective or evaluation step
        else:
            # Note: SNR_weight should be negative
            loss_t = -self.T * 0.5 * SNR_weight * (error_t_phar + error_t_pocket)
            loss_0 = loss_0_x_phar + loss_0_x_pocket + loss_0_h
            loss_0 = loss_0 + neg_log_const_0

        nll = loss_t + loss_0 + kl_prior

        # Correct for normalization on x.
        if not (self.loss_type == 'l2' and self.training):
            nll = nll - delta_log_px

            # Transform conditional nll into joint nll
            # Note:
            # loss = -log p(x,h|N) and log p(x,h,N) = log p(x,h|N) + log p(N)
            # Therefore, log p(x,h|N) = -loss + log p(N)
            # => loss_new = -log p(x,h,N) = loss - log p(N)
            nll = nll - log_pN

        info['error_t_phar'] = error_t_phar.mean(0)
        info['error_t_pocket'] = error_t_pocket.mean(0)
        info['SNR_weight'] = SNR_weight.mean(0)
        info['loss_0'] = loss_0.mean(0)
        info['kl_prior'] = kl_prior.mean(0)
        info['delta_log_px'] = delta_log_px.mean(0)
        info['neg_log_const_0'] = neg_log_const_0.mean(0)
        info['log_pN'] = log_pN.mean(0)
        return nll, info

    def log_metrics(self, metrics_dict, split, batch_size=None, **kwargs):
        for m, value in metrics_dict.items():
            self.log(f'{m}/{split}', value, batch_size=batch_size, **kwargs)

    def training_step(self, data, *args):
        if self.augment_noise > 0:
            raise NotImplementedError
            # Add noise eps ~ N(0, augment_noise) around points.
            eps = sample_center_gravity_zero_gaussian(x.size(), x.device)
            x = x + eps * args.augment_noise

        if self.augment_rotation:
            raise NotImplementedError
            x = utils.random_rotation(x).detach()

        nll, info = self.forward(data)
        loss = nll.mean(0)

        info['loss'] = loss
        self.log_metrics(info, 'train', batch_size=len(data['num_phar_atoms']))

        return info

    def _shared_eval(self, data, prefix, *args):
        nll, info = self.forward(data)
        loss = nll.mean(0)

        info['loss'] = loss

        # some additional info
        gamma_0 = self.ddpm.gamma(torch.zeros(1, device=self.device))
        gamma_1 = self.ddpm.gamma(torch.ones(1, device=self.device))
        log_SNR_max = -gamma_0
        log_SNR_min = -gamma_1
        info['log_SNR_max'] = log_SNR_max
        info['log_SNR_min'] = log_SNR_min

        self.log_metrics(info, prefix, batch_size=len(data['num_phar_atoms']),
                         sync_dist=True)

        return info

    def validation_step(self, data, *args):
        self._shared_eval(data, 'val', *args)

    def test_step(self, data, *args):
        self._shared_eval(data, 'test', *args)

    def validation_epoch_end(self, validation_step_outputs):

        # Perform validation on single GPU
        # TODO: sample on multiple devices if available
        if not self.trainer.is_global_zero:
            return

        if (self.current_epoch + 1) % self.eval_epochs == 0:
            tic = time()

            sampling_results = getattr(self, 'sample_and_analyze_given_pocket')(
                self.eval_params.n_eval_samples, self.val_dataset,
                batch_size=self.eval_batch_size)
            self.log_metrics(sampling_results, 'val')

            print(f'Evaluation took {time() - tic:.2f} seconds')


    def analyze_sample(self, phars, phar_types, aa_types):
        # Distribution of node types
        kl_div_atom = self.phar_type_distribution.kl_divergence(phar_types) \
            if self.phar_type_distribution is not None else -1
        kl_div_aa = self.pocket_type_distribution.kl_divergence(aa_types) \
            if self.pocket_type_distribution is not None else -1

        # Stability
        #phar_stable = 0
        #for pos, phar_type, ref in phars:
        #    pos_com = torch.mean(pos, dim=0)
        #    ref_com = torch.mean(ref, dim=0).cpu()
        #    # 计算 RMSD
        #    distance = torch.norm(pos_com - ref_com)

        #    # 判断 RMSD 是否小于 2
        #    if distance <= 2:
        #        phar_stable += 1

        #fraction_mol_stable = phar_stable / float(len(phars))

        print("kl_div_atom_types:", kl_div_atom)
        print("kl_div_residue_types:", kl_div_aa)
        #print("phar_stable:", fraction_mol_stable)

        return {
            'kl_div_atom_types': kl_div_atom,
            'kl_div_residue_types': kl_div_aa,
        }

    @torch.no_grad()
    def sample_and_analyze_given_pocket(self, n_samples, dataset=None,
                                        batch_size=None):
        print(f'Analyzing molecule stability given pockets at epoch '
              f'{self.current_epoch}...')

        batch_size = self.batch_size if batch_size is None else batch_size
        batch_size = min(batch_size, n_samples)

        # each item in molecules is a tuple (position, atom_type_encoded)
        phars = []
        phar_types = []
        aa_types = []
        for i in range(math.ceil(n_samples / batch_size)):

            n_samples_batch = min(batch_size, n_samples - len(phars))

            # Create a batch
            batch = dataset.collate_fn(
                [dataset[(i * batch_size + j) % len(dataset)]
                 for j in range(n_samples_batch)]
            )

            phar, pocket = self.get_phar_and_pocket(batch)

            num_nodes_phar = self.ddpm.size_distribution.sample_conditional(
                n1=None, n2=pocket['size'])

            xh_phar, xh_pocket, phar_mask, _ = self.ddpm.sample_given_pocket(
                pocket, num_nodes_phar)

            x = xh_phar[:, :self.x_dims].detach().cpu()
            phar_type = xh_phar[:, self.x_dims:].argmax(1).detach().cpu()

            phars.extend(list(
                zip(utils.batch_to_list(x, phar_mask),
                    utils.batch_to_list(phar_type, phar_mask),
                    utils.batch_to_list(phar['x'],phar['mask'])
            )))


            phar_types.extend(phar_type.tolist())
            aa_types.extend(
                xh_pocket[:, self.x_dims:].argmax(1).detach().cpu().tolist())

        return self.analyze_sample(phars, phar_types, aa_types)


    def generate_phars(self, pdb_file, n_samples, pocket_ids=None,
                         ref_ligand=None, num_nodes_phar=None, sanitize=False,
                         largest_frag=False, relax_iter=0, timesteps=None,
                         **kwargs):
        """
        Generate ligands given a pocket
        Args:
            pdb_file: PDB filename
            n_samples: number of samples
            pocket_ids: list of pocket residues in <chain>:<resi> format
            ref_ligand: alternative way of defining the pocket based on a
                reference ligand given in <chain>:<resi> format
            num_nodes_lig: number of ligand nodes for each sample (list of
                integers), sampled randomly if 'None'
            sanitize: whether to sanitize molecules or not
            largest_frag: only return the largest fragment
            relax_iter: number of force field optimization steps
            timesteps: number of denoising steps, use training value if None
            kwargs: additional inpainting parameters
        Returns:
            list of molecules
        """

        assert (pocket_ids is None) ^ (ref_ligand is None)

        # Load PDB
        pdb_struct = PDBParser(QUIET=True).get_structure('', pdb_file)[0]
        if pocket_ids is not None:
            # define pocket with list of residues
            residues = [
                pdb_struct[x.split(':')[0]][(' ', int(x.split(':')[1]), ' ')]
                for x in pocket_ids]

        else:
            # define pocket with reference ligand
            residues = utils.get_pocket_from_ligand(pdb_struct, ref_ligand)

        if self.pocket_representation == 'CA':
            pocket_coord = torch.tensor(np.array(
                [res['CA'].get_coord() for res in residues]),
                device=self.device, dtype=FLOAT_TYPE)
            pocket_types = torch.tensor(
                [self.pocket_type_encoder[three_to_one(res.get_resname())]
                 for res in residues], device=self.device)
        else:
            pocket_atoms = [a for res in residues for a in res.get_atoms()
                            if (a.element.capitalize() in self.pocket_type_encoder or a.element != 'H')]
            pocket_coord = torch.tensor(np.array(
                [a.get_coord() for a in pocket_atoms]),
                device=self.device, dtype=FLOAT_TYPE)
            pocket_types = torch.tensor(
                [self.pocket_type_encoder[a.element.capitalize()]
                 for a in pocket_atoms], device=self.device)

        pocket_one_hot = F.one_hot(
            pocket_types, num_classes=len(self.pocket_type_encoder)
        )

        pocket_size = torch.tensor([len(pocket_coord)] * n_samples,
                                   device=self.device, dtype=INT_TYPE)
        pocket_mask = torch.repeat_interleave(
            torch.arange(n_samples, device=self.device, dtype=INT_TYPE),
            len(pocket_coord)
        )

        pocket = {
            'x': pocket_coord.repeat(n_samples, 1),
            'one_hot': pocket_one_hot.repeat(n_samples, 1),
            'size': pocket_size,
            'mask': pocket_mask
        }

        # Pocket's center of mass
        pocket_com_before = scatter_mean(pocket['x'], pocket['mask'], dim=0)

        # Create dummy ligands
        if num_nodes_phar is None:
            num_nodes_phar = self.ddpm.size_distribution.sample_conditional(
                n1=None, n2=pocket['size'])

        # Use inpainting
        if type(self.ddpm) == EnVariationalDiffusion:
            phar_mask = utils.num_nodes_to_batch_mask(
                len(num_nodes_phar), num_nodes_phar, self.device)

            phar = {
                'x': torch.zeros((len(phar_mask), self.x_dims),
                                 device=self.device, dtype=FLOAT_TYPE),
                'one_hot': torch.zeros((len(phar_mask), self.phar_nf),
                                       device=self.device, dtype=FLOAT_TYPE),
                'size': num_nodes_phar,
                'mask': phar_mask
            }

            # Fix all pocket nodes but sample
            phar_mask_fixed = torch.zeros(len(phar_mask), device=self.device)
            pocket_mask_fixed = torch.ones(len(pocket['mask']),
                                           device=self.device)

            xh_phar, xh_pocket, phar_mask, pocket_mask = self.ddpm.inpaint(
                phar, pocket, phar_mask_fixed, pocket_mask_fixed,
                timesteps=timesteps, **kwargs)

        # Use conditional generation
        elif type(self.ddpm) == ConditionalDDPM:
            xh_phar, xh_pocket, phar_mask, pocket_mask = \
                self.ddpm.sample_given_pocket(pocket, num_nodes_phar,
                                              timesteps=timesteps)

        else:
            raise NotImplementedError

        # Move generated molecule back to the original pocket position
        pocket_com_after = scatter_mean(
            xh_pocket[:, :self.x_dims], pocket_mask, dim=0)

        xh_pocket[:, :self.x_dims] += \
            (pocket_com_before - pocket_com_after)[pocket_mask]
        xh_phar[:, :self.x_dims] += \
            (pocket_com_before - pocket_com_after)[phar_mask]

        # Build mol objects
        phar_mask = phar_mask.cpu()
        x = xh_phar[:, :self.x_dims].detach().cpu()
        phar_type = xh_phar[:, self.x_dims:].argmax(1).detach().cpu()

        phar_to_coords = {}
        for mol_pc in zip(utils.batch_to_list(x, phar_mask),
                          utils.batch_to_list(phar_type, phar_mask)):
            # 提取坐标和药效团类型
            coords_batch = mol_pc[0]
            atom_types = mol_pc[1]

            # 转换药效团类型为药效团名称
            atom_names = [self.dataset_info["phar_decoder"][x] for x in atom_types]

            # 在循环内部维护一个计数器来为每个分子命名
            molecule_counter = 1

            for atom_name, coords in zip(atom_names, coords_batch):
                # 使用分子计数器来为每个分子命名
                molecule_name = f"Molecule_{molecule_counter}"

                # 创建一个新的分子字典
                if molecule_name not in phar_to_coords:
                    phar_to_coords[molecule_name] = {}

                # 使用列表存储坐标，以处理相同药效团名称的情况
                if atom_name not in phar_to_coords[molecule_name]:
                    phar_to_coords[molecule_name][atom_name] = []

                # 将坐标添加到分子的药效团名称下
                phar_to_coords[molecule_name][atom_name].append(coords)

                # 增加分子计数器，以确保每个分子有唯一的名称
                molecule_counter += 1
        return phar_to_coords

    def configure_gradient_clipping(self, optimizer, optimizer_idx,
                                    gradient_clip_val, gradient_clip_algorithm):

        if not self.clip_grad:
            return

        # Allow gradient norm to be 150% + 2 * stdev of the recent history.
        max_grad_norm = 1.5 * self.gradnorm_queue.mean() + \
                        2 * self.gradnorm_queue.std()

        # Get current grad_norm
        params = [p for g in optimizer.param_groups for p in g['params']]
        grad_norm = utils.get_grad_norm(params)

        # Lightning will handle the gradient clipping
        self.clip_gradients(optimizer, gradient_clip_val=max_grad_norm,
                            gradient_clip_algorithm='norm')

        if float(grad_norm) > max_grad_norm:
            self.gradnorm_queue.add(float(max_grad_norm))
        else:
            self.gradnorm_queue.add(float(grad_norm))

        if float(grad_norm) > max_grad_norm:
            print(f'Clipped gradient with value {grad_norm:.1f} '
                  f'while allowed {max_grad_norm:.1f}')
