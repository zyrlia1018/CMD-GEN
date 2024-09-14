import argparse
import warnings
from pathlib import Path
from time import time
import numpy as np

import torch
from rdkit import Chem
from tqdm import tqdm

from lightning_modules import PharPocketDDPM
from analysis.molecule_builder import process_molecule
import utils

from rdkit import Chem
from rdkit import RDConfig
from rdkit.Chem import ChemicalFeatures
import os
from constants import covalent_radii, dataset_params

dataset_info = dataset_params['crossdock_full']
amino_acid_dict = dataset_info['aa_encoder']
atom_dict = dataset_info['atom_encoder']
# atom_decoder = dataset_info['atom_decoder']
phar_dict = dataset_info['phar_encoder']
phar_decoder = dataset_info['phar_decoder']
num_phar_classes = 8


MAXITER = 10
MAXNTRIES = 3


def get_type_histograms(phar_one_hot, phar_encoder):
    phar_decoder = list(phar_encoder.keys())
    phar_counts = {k: 0 for k in phar_encoder.keys()}
    for a in [phar_decoder[x] for x in phar_one_hot.argmax(1)]:
        phar_counts[a] += 1


    return phar_counts

def convert_pharmacophore_to_one_hot(pharmacophore, num_classes=num_phar_classes):
    one_hot_array = np.zeros((len(pharmacophore), num_classes))

    for i, value in enumerate(pharmacophore):
        if 0 <= value <= num_classes:
            one_hot_array[i, value - 1] = 1.0
        else:
            raise ValueError("Pharmacophore value is out of range.")

    return np.array(one_hot_array)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('checkpoint', type=Path)
    parser.add_argument('--test_dir', type=Path)
    parser.add_argument('--test_list', type=Path, default=None)
    parser.add_argument('--all_frags', action='store_true')
    parser.add_argument('--sanitize', action='store_true')
    parser.add_argument('--relax', action='store_true')
    parser.add_argument('--fix_n_nodes', action='store_true')
    parser.add_argument('--batch_size', type=int, default=15)
    parser.add_argument('--resamplings', type=int, default=1)
    parser.add_argument('--jump_length', type=int, default=1)
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'


    # Load model
    model = PharPocketDDPM.load_from_checkpoint(
        args.checkpoint, map_location=device)
    model = model.to(device)

    test_files = list(args.test_dir.glob('[!.]*.sdf'))
    if args.test_list is not None:
        with open(args.test_list, 'r') as f:
            test_list = set(f.read().split(','))
        test_files = [x for x in test_files if x.stem in test_list]

    pbar = tqdm(test_files)

    ref_com_distances = []
    ref_max_phar_distances = []
    ref_num_gen = []
    ref_phar_onehot = []
    com_distances = []
    max_phar_distances = []
    num_gen = []
    gen_phar_onehot = []

    for sdf_file in pbar:
        ligand_name = sdf_file.stem

        pdb_name, pocket_id, *suffix = ligand_name.split('_')
        pdb_file = Path(sdf_file.parent, f"{pdb_name}.pdb")
        txt_file = Path(sdf_file.parent, f"{ligand_name}.txt")
        
        for n_try in range(MAXNTRIES):
            t_pocket_start = time()

            with open(txt_file, 'r') as f:
                resi_list = f.read().split()


            if args.fix_n_nodes:
                # some ligands (e.g. 6JWS_bio1_PT1:A:801) could not be read with sanitize=True
                suppl = Chem.SDMolSupplier(str(sdf_file), sanitize=True)[0]
                num_nodes_lig = suppl.GetNumAtoms()
                conf = suppl.GetConformer()
                centroid = conf.GetPositions().mean(axis=0)
            else:
                num_nodes_lig = None
                suppl = Chem.SDMolSupplier(str(sdf_file), sanitize=True)[0]
                conf = suppl.GetConformer()
                centroid = conf.GetPositions().mean(axis=0)

                ref_pharmocophore = []
                ref_position = []

                fdefName = os.path.join(RDConfig.RDDataDir, 'BaseFeatures.fdef')
                factory = ChemicalFeatures.BuildFeatureFactory(fdefName)
                feats = factory.GetFeaturesForMol(suppl)

                for f in feats:
                    phar = f.GetFamily()
                    pos = list(f.GetPos())  # 三维的坐标信息，会自动算中心点
                    # atom_type = f.GetType()
                    phar_index = phar_dict.setdefault(phar, 7)
                    # pharmocophore_ = [phar_index, atom_index]  # some pharmacophore feature
                    ref_pharmocophore.append(phar_index)  # all pharmacophore features within a molecule
                    ref_position.append(pos)
                ref_phar_one_hot = convert_pharmacophore_to_one_hot(ref_pharmocophore)
                ref_phar_onehot.append(ref_phar_one_hot)
                ref_phar_coords = np.array(ref_position)

                ref_max_distance = 0
                for i in range(len(ref_phar_coords)):
                    for j in range(i + 1, len(ref_phar_coords)):
                        ref_distance = np.linalg.norm(ref_phar_coords[i] - ref_phar_coords[j])
                        if ref_distance > ref_max_distance:
                            ref_max_distance = ref_distance

                ref_num_phar_gen = len(ref_phar_coords)
                com_ref_phar_coords = np.mean(ref_phar_coords, axis=0)
                ref_com_distance = np.linalg.norm(centroid - com_ref_phar_coords)

                ref_com_distances.append(ref_com_distance)
                ref_max_phar_distances.append(ref_max_distance)
                ref_num_gen.append(ref_num_phar_gen)


                num_nodes_lig_inflated = None if num_nodes_lig is None else \
                    torch.ones(args.batch_size, dtype=int) * num_nodes_lig

                # Turn all filters off first
                phar_to_coords = model.generate_phars(
                    pdb_file, args.batch_size, resi_list,
                    num_nodes_phar=num_nodes_lig_inflated, sanitize=False,
                    largest_frag=False, relax_iter=0,
                    resamplings=args.resamplings, jump_length=args.jump_length)

                for molecule_name, molecule_data in phar_to_coords.items():

                    molecule_centroids = {}  # 用于存储每个分子内药效团的质心
                    all_phar_coords = []  # 用于存储所有药效团的坐标a
                    all_gen_pharmocophore = []
                                      
                    for phar_name in molecule_data.keys():
                        gen_phar_index = phar_dict.setdefault(phar_name, 7)
                        all_gen_pharmocophore.append(gen_phar_index)
                    gen_phar_one_hot = convert_pharmacophore_to_one_hot(all_gen_pharmocophore)
                    
                     
                    for phar_coords in molecule_data.values():
                        all_phar_coords.extend(phar_coords)
                   
                    # 生成的药效团的个数
                    num_phar_gen = len(all_phar_coords)
                    # 计算每个分子的药效团质心
                    molecule_centroid = np.mean(all_phar_coords, axis=0)
                    # 计算生成药效团与参考分子质心之间的距离
                    com_distance = np.linalg.norm(centroid - molecule_centroid)
                    # 计算每个分子内药效团间的最远距离
                    max_distance = 0
                    for i in range(len(all_phar_coords)):
                        for j in range(i + 1, len(all_phar_coords)):
                            distance = np.linalg.norm(all_phar_coords[i] - all_phar_coords[j])
                            if distance > max_distance:
                                max_distance = distance
                  
                    com_distances.append(com_distance)
                    max_phar_distances.append(max_distance)
                    num_gen.append(num_phar_gen)
                    gen_phar_onehot.extend(gen_phar_one_hot)
    
    ref_phar_onehot = np.concatenate(ref_phar_onehot, axis=0)
    phar2_dict = {'Aromatic': 0, 'Hydrophobe': 1, 'PosIonizable': 2, 'NegIonizable': 3, 'Acceptor': 4, 'Donor': 5, 'LumpedHydrophobe': 6, 'others': 7}
    ref_phar_hist = get_type_histograms(ref_phar_onehot, phar2_dict)
    gen_phar_hist = get_type_histograms(np.array(gen_phar_onehot), phar2_dict)
    
    print(ref_phar_hist)
    print(gen_phar_hist)
     
    #print(ref_com_distances,
    #ref_max_phar_distances,
    #ref_num_gen,
    #com_distances,
    #max_phar_distances,
    #num_gen)
    # 假设以下是您的数据，这里用随机数代替
    ref_com_distances = np.array(ref_com_distances)
    com_distances = np.array(com_distances)
    ref_num_gen = np.array(ref_num_gen)
    num_gen = np.array(num_gen)
    ref_max_phar_distances = np.array(ref_max_phar_distances)
    max_phar_distances = np.array(max_phar_distances)

    np.savez('data_full_15.npz',
         ref_com_distances=ref_com_distances,
         com_distances=com_distances,
         ref_num_gen=ref_num_gen,
         num_gen=num_gen,
         ref_max_phar_distances=ref_max_phar_distances,
         max_phar_distances=max_phar_distances)
