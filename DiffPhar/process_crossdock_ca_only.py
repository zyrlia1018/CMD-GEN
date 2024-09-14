from pathlib import Path
from time import time
import argparse
import shutil
import random

import matplotlib.pyplot as plt
import seaborn as sns

from tqdm import tqdm
import numpy as np

from Bio.PDB import PDBParser
from Bio.PDB.Polypeptide import three_to_one, is_aa
from rdkit import Chem
from scipy.ndimage import gaussian_filter

import torch


import constants
from constants import covalent_radii, dataset_params

import os
import random

import torch
from rdkit import Chem
from rdkit import RDConfig
from rdkit.Chem import ChemicalFeatures


dataset_info = dataset_params['crossdock']
amino_acid_dict = dataset_info['aa_encoder']
atom_dict = dataset_info['atom_encoder']
# atom_decoder = dataset_info['atom_decoder']
phar_dict = dataset_info['phar_encoder']
phar_decoder = dataset_info['phar_decoder']
num_phar_classes = 8


def convert_pharmacophore_to_one_hot(pharmacophore, num_classes=num_phar_classes):
    one_hot_array = np.zeros((len(pharmacophore), num_classes))

    for i, value in enumerate(pharmacophore):
        if 0 < value <= num_classes:
            one_hot_array[i, value - 1] = 1.0
        else:
            raise ValueError("Pharmacophore value is out of range.")

    return np.array(one_hot_array)


def process_ligand_and_pocket(pdbfile, sdffile,
                              phar_dict, dist_cutoff, ca_only):
    pdb_struct = PDBParser(QUIET=True).get_structure('', pdbfile)

    # ligand
    try:
        ligand = Chem.SDMolSupplier(str(sdffile))[0]
    except:
        raise Exception(f'cannot read sdf mol ({sdffile})')

    lig_coords = np.array([list(ligand.GetConformer(0).GetAtomPosition(idx))
                           for idx in range(ligand.GetNumAtoms())])
    # Find interacting pocket residues based on distance cutoff
    pocket_residues = []
    for residue in pdb_struct[0].get_residues():
        res_coords = np.array([a.get_coord() for a in residue.get_atoms()])
        if is_aa(residue.get_resname(), standard=True) and \
                (((res_coords[:, None, :] - lig_coords[None, :, :]) ** 2).sum(-1) ** 0.5).min() < dist_cutoff:
            pocket_residues.append(residue)

    pocket_ids = [f'{res.parent.id}:{res.id[1]}' for res in pocket_residues]

    # pharmocophore

    pharmocophore = []
    atom_index_list = []
    position = []

    fdefName = os.path.join(RDConfig.RDDataDir, 'BaseFeatures.fdef')
    factory = ChemicalFeatures.BuildFeatureFactory(fdefName)
    feats = factory.GetFeaturesForMol(ligand)
    for f in feats:
        phar = f.GetFamily()
        pos = list(f.GetPos())  # 三维的坐标信息，会自动算中心点
        atom_index = f.GetAtomIds()
        atom_index = tuple(sorted(atom_index))
        # atom_type = f.GetType()
        phar_index = phar_dict.setdefault(phar, 8)
        # pharmocophore_ = [phar_index, atom_index]  # some pharmacophore feature
        pharmocophore.append(phar_index)  # all pharmacophore features within a molecule
        atom_index_list.append(atom_index)  # atom indices of one pharmacophore feature
        position.append(pos)

    phar_coords = np.array(position)
    assert len(phar_coords) > 0, "phar_coords长度为零，请检查输入数据。"
    phar_one_hot = convert_pharmacophore_to_one_hot(pharmocophore)
    phar_data = {
        'phar_coords': phar_coords,
        'phar_one_hot': phar_one_hot,
    }

    # pocket

    if ca_only:
        try:
            pocket_one_hot = []
            full_coords = []
            for res in pocket_residues:
                for atom in res.get_atoms():
                    if atom.name == 'CA':
                        pocket_one_hot.append(np.eye(1, len(amino_acid_dict),
                                                     amino_acid_dict[three_to_one(res.get_resname())]).squeeze())
                        full_coords.append(atom.coord)
            pocket_one_hot = np.stack(pocket_one_hot)
            full_coords = np.stack(full_coords)
        except KeyError as e:
            raise KeyError(
                f'{e} not in amino acid dict ({pdbfile}, {sdffile})')
        pocket_data = {
            'pocket_ca': full_coords,
            'pocket_one_hot': pocket_one_hot,
            'pocket_ids': pocket_ids
        }
    else:
        full_atoms = np.concatenate([np.array([atom.element for atom in res.get_atoms()]) for res in pocket_residues],
                                    axis=0)
        full_coords = np.concatenate([np.array([atom.coord for atom in res.get_atoms()]) for res in pocket_residues],
                                     axis=0)
        try:
            pocket_one_hot = []
            for a in full_atoms:
                if a in amino_acid_dict:
                    atom = np.eye(1, len(amino_acid_dict), amino_acid_dict[a.capitalize()]).squeeze()
                elif a != 'H':
                    atom = np.eye(1, len(amino_acid_dict), len(amino_acid_dict)).squeeze()
                pocket_one_hot.append(atom)
            pocket_one_hot = np.stack(pocket_one_hot)
        except KeyError as e:
            raise KeyError(
                f'{e} not in atom dict ({pdbfile})')
        pocket_data = {
            'pocket_ca': full_coords,
            'pocket_one_hot': pocket_one_hot,
            'pocket_ids': pocket_ids
        }
    return phar_data, pocket_data


def get_n_nodes(phar_mask, pocket_mask, smooth_sigma=None):
    # Joint distribution of ligand's and pocket's number of nodes
    idx_phar, n_nodes_phar = np.unique(phar_mask, return_counts=True)
    idx_pocket, n_nodes_pocket = np.unique(pocket_mask, return_counts=True)
    assert np.all(idx_phar == idx_pocket)

    joint_histogram = np.zeros((np.max(n_nodes_phar) + 1,
                                np.max(n_nodes_pocket) + 1))

    for nphar, npocket in zip(n_nodes_phar, n_nodes_pocket):
        joint_histogram[nphar, npocket] += 1

    print(f'Original histogram: {np.count_nonzero(joint_histogram)}/'
          f'{joint_histogram.shape[0] * joint_histogram.shape[1]} bins filled')

    # Smooth the histogram
    if smooth_sigma is not None:
        filtered_histogram = gaussian_filter(
            joint_histogram, sigma=smooth_sigma, order=0, mode='constant',
            cval=0.0, truncate=4.0)

        print(f'Smoothed histogram: {np.count_nonzero(filtered_histogram)}/'
              f'{filtered_histogram.shape[0] * filtered_histogram.shape[1]} bins filled')

        joint_histogram = filtered_histogram

    return joint_histogram


def get_type_histograms(phar_one_hot, pocket_one_hot, phar_encoder, aa_encoder):
    phar_decoder = list(phar_encoder.keys())
    phar_counts = {k: 0 for k in phar_encoder.keys()}
    for a in [phar_decoder[x] for x in phar_one_hot.argmax(1)]:
        phar_counts[a] += 1

    aa_decoder = list(aa_encoder.keys())
    aa_counts = {k: 0 for k in aa_encoder.keys()}
    for r in [aa_decoder[x] for x in pocket_one_hot.argmax(1)]:
        aa_counts[r] += 1

    return phar_counts, aa_counts


def saveall(filename, pdb_and_mol_ids,
            phar_coords, phar_one_hot, phar_mask,
            pocket_c_alpha, pocket_one_hot, pocket_mask):
    np.savez(filename,
             names=pdb_and_mol_ids,
             phar_coords=phar_coords,
             phar_one_hot=phar_one_hot,
             phar_mask=phar_mask,
             pocket_c_alpha=pocket_c_alpha,
             pocket_one_hot=pocket_one_hot,
             pocket_mask=pocket_mask
             )
    return True


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('basedir', type=Path)
    parser.add_argument('--outdir', type=Path, default=None)
    parser.add_argument('--no_H', action='store_true')
    parser.add_argument('--ca_only', action='store_true')
    parser.add_argument('--dist_cutoff', type=float, default=8.0)
    parser.add_argument('--random_seed', type=int, default=42)
    args = parser.parse_args()

    datadir = args.basedir / 'crossdocked_pocket10/'

    if args.ca_only:
        dataset_info = dataset_params['crossdock']

    # Make output directory
    if args.outdir is None:
        suffix = '_crossdock' if 'H' in atom_dict else '_crossdock_noH'
        suffix += '_ca_only_temp' if args.ca_only else '_full_temp'
        processed_dir = Path(args.basedir, f'processed{suffix}')
    else:
        processed_dir = args.outdir

    processed_dir.mkdir(exist_ok=True, parents=True)

    # Read data split
    split_path = Path(args.basedir, 'split_by_name.pt')
    data_split = torch.load(split_path)

    # There is no validation set, copy 300 training examples (the validation set
    # is not very important in this application)
    # Note: before we had a data leak but it should not matter too much as most
    # metrics monitored during training are independent of the pockets
    data_split['val'] = random.sample(data_split['train'], 300)

    n_train_before = len(data_split['train'])
    n_val_before = len(data_split['val'])
    n_test_before = len(data_split['test'])

    failed_save = []

    n_samples_after = {}
    for split in data_split.keys():
        phar_coords = []
        phar_one_hot = []
        phar_mask = []
        pocket_c_alpha = []
        pocket_one_hot = []
        pocket_mask = []
        pdb_and_mol_ids = []
        count_protein = []
        count_phar = []
        count_total = []
        count = 0

        pdb_sdf_dir = processed_dir / split
        pdb_sdf_dir.mkdir(exist_ok=True)

        tic = time()
        num_failed = 0
        pbar = tqdm(data_split[split])
        pbar.set_description(f'#failed: {num_failed}')
        for pocket_fn, lig_phar_fn in pbar:

            sdffile = datadir / f'{lig_phar_fn}'
            pdbfile = datadir / f'{pocket_fn}'

            try:
                struct_copy = PDBParser(QUIET=True).get_structure('', pdbfile)
            except:
                num_failed += 1
                failed_save.append((pocket_fn, lig_phar_fn))
                print(failed_save[-1])
                pbar.set_description(f'#failed: {num_failed}')
                continue

            try:
                phar_data, pocket_data = process_ligand_and_pocket(
                    pdbfile, sdffile,
                    phar_dict=phar_dict, dist_cutoff=args.dist_cutoff, ca_only=args.ca_only)
            except (KeyError, AssertionError, FileNotFoundError, IndexError,
                    ValueError) as e:
                print(type(e).__name__, e, pocket_fn, lig_phar_fn)
                num_failed += 1
                pbar.set_description(f'#failed: {num_failed}')
                continue

            pdb_and_mol_ids.append(f"{pocket_fn}_{lig_phar_fn}")

            phar_coords.append(phar_data['phar_coords'])
            phar_one_hot.append(phar_data['phar_one_hot'])
            phar_mask.append(count * np.ones(len(phar_data['phar_coords'])))

            pocket_c_alpha.append(pocket_data['pocket_ca'])
            pocket_one_hot.append(pocket_data['pocket_one_hot'])
            pocket_mask.append(count * np.ones(len(pocket_data['pocket_ca'])))
            count_protein.append(pocket_data['pocket_ca'].shape[0])
            count_phar.append(phar_data['phar_coords'].shape[0])
            count_total.append(pocket_data['pocket_ca'].shape[0] + phar_data['phar_coords'].shape[0])
            count += 1

            if split in {'val', 'test'}:
                # Copy PDB file
                new_rec_name = Path(pdbfile).stem.replace('_', '-')
                pdb_file_out = Path(pdb_sdf_dir, f"{new_rec_name}.pdb")
                shutil.copy(pdbfile, pdb_file_out)

                # Copy SDF file
                new_lig_name = new_rec_name + '_' + Path(sdffile).stem.replace('_', '-')
                sdf_file_out = Path(pdb_sdf_dir, f'{new_lig_name}.sdf')
                shutil.copy(sdffile, sdf_file_out)

                # specify pocket residues
                with open(Path(pdb_sdf_dir, f'{new_lig_name}.txt'), 'w') as f:
                    f.write(' '.join(pocket_data['pocket_ids']))

        #print(phar_coords)
        phar_coords = np.concatenate(phar_coords, axis=0)
        phar_one_hot = np.concatenate(phar_one_hot, axis=0)
        phar_mask = np.concatenate(phar_mask, axis=0)

        pocket_c_alpha = np.concatenate(pocket_c_alpha, axis=0)
        pocket_one_hot = np.concatenate(pocket_one_hot, axis=0)
        pocket_mask = np.concatenate(pocket_mask, axis=0)

        saveall(processed_dir / f'{split}.npz', pdb_and_mol_ids,
                phar_coords, phar_one_hot, phar_mask,
                pocket_c_alpha, pocket_one_hot, pocket_mask)

        n_samples_after[split] = len(pdb_and_mol_ids)
        print(f"Processing {split} set took {(time() - tic) / 60.0:.2f} minutes")

    # --------------------------------------------------------------------------
    # Compute statistics & additional information
    # --------------------------------------------------------------------------
    with np.load(processed_dir / 'train.npz', allow_pickle=True) as data:

        phar_mask = data['phar_mask']
        phar_coords = data['phar_coords']
        phar_one_hot = data['phar_one_hot']

        pocket_mask = data['pocket_mask']
        pocket_one_hot = data['pocket_one_hot']


    # Joint histogram of number of ligand and pocket nodes
    n_nodes = get_n_nodes(phar_mask, pocket_mask, smooth_sigma=1.0)
    np.save(Path(processed_dir, 'size_distribution.npy'), n_nodes)

    # Get histograms of ligand and pocket node types
    phar_hist, aa_hist = get_type_histograms(phar_one_hot, pocket_one_hot,
                                             phar_dict, amino_acid_dict)

    # Create summary string
    summary_string = '# SUMMARY\n\n'
    summary_string += '# Before processing\n'
    summary_string += f'num_samples train: {n_train_before}\n'
    summary_string += f'num_samples val: {n_val_before}\n'
    summary_string += f'num_samples test: {n_test_before}\n\n'
    summary_string += '# After processing\n'
    summary_string += f"num_samples train: {n_samples_after['train']}\n"
    summary_string += f"num_samples val: {n_samples_after['val']}\n"
    summary_string += f"num_samples test: {n_samples_after['test']}\n\n"
    summary_string += '# Info\n'
    summary_string += f"'phar_encoder': {phar_dict}\n"
    summary_string += f"'phar_decoder': {list(phar_dict.keys())}\n"
    summary_string += f"'aa_encoder': {amino_acid_dict}\n"
    summary_string += f"'aa_decoder': {list(amino_acid_dict.keys())}\n"
    summary_string += f"'phar_hist': {phar_hist}\n"
    summary_string += f"'aa_hist': {aa_hist}\n"
    summary_string += f"'n_nodes': {n_nodes.tolist()}\n"

    sns.distplot(count_protein)
    plt.savefig(processed_dir / 'protein_size_distribution.png')
    plt.clf()

    sns.distplot(count_phar)
    plt.savefig(processed_dir / 'lig_phar_size_distribution.png')
    plt.clf()

    sns.distplot(count_total)
    plt.savefig(processed_dir / 'total_size_distribution.png')
    plt.clf()

    # Write summary to text file
    with open(processed_dir / 'summary.txt', 'w') as f:
        f.write(summary_string)

    # Print summary
    print(summary_string)

    print(failed_save)
