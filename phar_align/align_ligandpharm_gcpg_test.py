from rdkit import Geometry
from rdkit.Chem.Pharm3D import Pharmacophore as rdkitPharmacophore

from operator import itemgetter
import numpy as np

from rdkit import Geometry
from rdkit import Chem
from rdkit.Chem import rdMolTransforms
from rdkit.Numerics import rdAlignment

import os
from rdkit import RDConfig
from rdkit.Chem.Pharm3D import EmbedLib
from rdkit.Chem import ChemicalFeatures, rdDistGeom
fdefName = os.path.join(RDConfig.RDDataDir, 'BaseFeatures.fdef')
factory = ChemicalFeatures.BuildFeatureFactory(fdefName)

import argparse
from pathlib import Path


def pharmacophore_to_rdkit(pharmacophoric_points):
    """ Transform a list of pharmacophoric points to an rdkit pharmacophore.

        Returns
        -------
        pharmacophore : rdkit.Chem.Pharm3D.Pharmacophore
            The rdkit pharmacophore.

    """
    rdkit_element_name = {
        "aromatic ring": "Aromatic",
        "hydrophobicity": "Hydrophobe",
        "hb acceptor": "Acceptor",
        "hb donor": "Donor",
        "positive charge": "PosIonizable",
        "negative charge": "NegIonizable",
    }

    points = []
    radii = []

    for element in pharmacophoric_points:
        feat_name = rdkit_element_name[element["feature_name"]]
        center = Geometry.Point3D(element["center"][0], element["center"][1], element["center"][2])
        points.append(ChemicalFeatures.FreeChemicalFeature(
            feat_name,
            center
        ))
        radius = element["radius"]
        radii.append(radius)

    pharmacophore = rdkitPharmacophore.Pharmacophore(points)
    # Apply the radius of each point to the bound matrix of the pharmacophore
    for ii in range(len(radii)):
        for jj in range(ii + 1, len(radii)):
            sum_radii = radii[ii] + radii[jj]
            pharmacophore.setLowerBound(ii, jj, max(pharmacophore.getLowerBound(ii, jj) - sum_radii, 0))
            pharmacophore.setUpperBound(ii, jj, pharmacophore.getUpperBound(ii, jj) + sum_radii)

    return pharmacophore



def feature_mappings(ligand, pharmacophore):
    """ Maps the chemical features in the given ligand to those of
        the pharmacophore.

        Parameters
        ----------
        ligand : rdkit.Mol
        pharmacophore : rdkit.Chem.Pharm3D.Pharmacophore

        Returns
        -------
        list [rdkit.MolChemicalFeature]

    """
    can_match, all_matches = EmbedLib.MatchPharmacophoreToMol(
        ligand, factory, pharmacophore)
    if can_match:
        bounds_matrix = rdDistGeom.GetMoleculeBoundsMatrix(ligand)
        failed, _, matched_features, _ = EmbedLib.MatchPharmacophore(
            all_matches, bounds_matrix, pharmacophore, useDownsampling=True)
        return matched_features
    return []



def align_pharmacophores(ref_pharma, probe_pharma):
    """ Align two pharmacophores against each other.

        Parameters
        ----------
        ref_pharma : np.ndarray
            Coordinates of the pharmacophoric points of the reference pharmacophore.
            Shape (n_points, 3)

        probe_pharma : np.ndarray
            Coordinates of the pharmacophoric points of probe pharmacophore.
            Shape (n_points, 3)

        Returns
        -------
        rmsd : float
            The root mean square deviation of the alignment.

        trans_mat : np.ndarray
            The transformation matrix. This matrix should be applied to the confomer of
            the probe_pharmacophore to obtain its updated positions.
    """
    ssd, trans_mat = rdAlignment.GetAlignmentTransform(ref_pharma, probe_pharma)
    rmsd = np.sqrt(ssd / ref_pharma.shape[0])
    return rmsd, trans_mat


def align_ligand_to_pharmacophore(ligand, atom_match, pharmacophore, num_gen):
    """ Align a ligand to a pharmacophore.

        Parameters
        ----------
        ligand : rdkit.Mol
        atom_match : list[list[int]]
        pharmacophore : rdkit.Chem.Pharm3D.Pharmacophore
        num_gen : [int] <10

        Returns
        -------
        rdkit.Mol
            The molecule with an aligned conformer.
        float
            The rmsd of the alignment.
    """

    try:
        _, embeddings, _ = EmbedLib.EmbedPharmacophore(ligand, atom_match, pharmacophore, count=10)
    except KeyError:
        return

    rmsd_list = _transform_embeddings(pharmacophore, embeddings, atom_match)
    if len(rmsd_list) == 0:
        return

    best_fit_indices = sorted(enumerate(rmsd_list), key=itemgetter(1))[:num_gen]
    embeddings = [embeddings[i] for i,_ in best_fit_indices]
    rmsd_list = [rmsd_list[i] for i,_ in best_fit_indices]
    return embeddings, rmsd_list


def _transform_embeddings(pharmacophore, embeddings, atom_match):
    """ Transform embeddings. Performs the alignment of the conformers of
        a molecule to the pharmacophore.

        Parameters
        ----------
        pharmacophore: rdkit.Chem.Pharm3D.Pharmacophore
            A pharmacophore object.

        embeddings: list[rdkit.Mol]
            List of molecules (the same molecule) with a single conformer.

        atom_match: list[list[int]]
            A nested list of atoms ids that match the pharmacophore.

        Returns
        -------
        rmsd_list: list[float]
            List of sum of RMSD values for the alignments.

    """
    align_ref = [f.GetPos() for f in pharmacophore.getFeatures()]
    rmsd_list = []
    for embedding in embeddings:
        conformer = embedding.GetConformer()
        rmsd, transform_matrix = _transform_matrix(align_ref, conformer, atom_match)
        # Transform the coordinates of the conformer
        rdMolTransforms.TransformConformer(conformer, transform_matrix)
        rmsd_list.append(rmsd)

    return rmsd_list


def _transform_matrix(align_ref, conformer_embed, atom_match):
    """ Get the transformation matrix of a conformer that is aligned to
        a pharmacophore.

        Parameters
        ----------
        align_ref: list[rdkit.Geometry.Point3D]
            list of pharmacophore reference points for the alignment.

        conformer_embed: rdkit.Conformer
            The conformer embedding.

        atom_match: list[list]
            Nested list of atoms ids that match the pharmacophore.

        Returns
        -------
        rmsd: float
            RMSD value of the alignment.

        transform_matrix: numpy.ndarray; shape(4, 4)
            The transform matrix.
    """
    align_probe = []  # probe points to align to reference points
    for match_ids in atom_match:
        # Calculate the centroid of the feature in case it has multiple atoms
        dummy_point = Geometry.Point3D(0.0, 0.0, 0.0)
        for id_ in match_ids:
            dummy_point += conformer_embed.GetAtomPosition(id_)
        dummy_point /= len(match_ids)
        align_probe.append(dummy_point)

    # Note This function returns the sum of squared distance (SSD) not the RMSD RMSD = sqrt(SSD/numPoints)
    ssd, transform_matrix = rdAlignment.GetAlignmentTransform(align_ref, align_probe)
    rmsd = np.sqrt(ssd / len(align_ref))
    return rmsd, transform_matrix



#
# lig = Chem.MolFromSmiles('CN(C)CN1CCCC1=Cc1c[nH]c(=O)c2ccccc12')
#
# # lig = Chem.SDMolSupplier("test_pi3k.sdf")[0]
# #
# # "aromatic ring": "Aromatic",
# # "hydrophobicity": "Hydrophobe",
# # "hb acceptor": "Acceptor",
# # "hb donor": "Donor",
# # "positive charge": "PosIonizable",
# # "negative charge": "NegIonizable",
#
# pharmacophoric_points = [
#     {
#         "feature_name": "hb acceptor",
#         "center": (15.41, 45.17, 4.46),
#         "radius": 1.0
#     },
#     {
#         "feature_name": "positive charge",
#         "center": (9.10, 40.86, 7.29),
#         "radius": 1.0
#     },
#     {
#         "feature_name": "hydrophobicity",
#         "center": (13.40, 42.07, 7.04),
#         "radius": 1.0
#     },
#     {
#         "feature_name": "hydrophobicity",
#         "center": (15.95, 43.04, 7.27),
#         "radius": 1.0
#     },
#     {
#         "feature_name": "positive charge",
#         "center": (11.29, 41.25, 7.05),
#         "radius": 1.0
#     },
#     {
#         "feature_name": "aromatic ring",
#         "center": (15.64, 43.80, 6.18),
#         "radius": 1.0
#     },
#     {
#         "feature_name": "hb donor",
#         "center": (14.16, 43.75, 5.28),
#         "radius": 1.0
#     },
# ]
#
# import itertools
# def generate_subsets_with_tolerance(pharmacophoric_points, tolerance):
#     all_subsets = []
#     all_subsets.append(pharmacophoric_points)
#
#     for subset in itertools.combinations(pharmacophoric_points, len(pharmacophoric_points) - tolerance):
#         all_subsets.append(list(subset))
#
#     return all_subsets
#
# tolerance=1
# num_gen=1
#
# if tolerance==0:
#     pharma = pharmacophore_to_rdkit(pharmacophoric_points)
#
#     mappings = feature_mappings(lig, pharma)
#     if len(mappings) > 0:
#         atom_match = [list(x.GetAtomIds()) for x in mappings]
#         lig = Chem.AddHs(lig)
#         result = align_ligand_to_pharmacophore(lig, atom_match, pharma, num_gen=num_gen)
#         if result is not None:
#             aligned_mols, rmsd = result
# else:
#     aligned_mols = []
#     phar_points_subsets = generate_subsets_with_tolerance(pharmacophoric_points, tolerance)
#     for phar_group in phar_points_subsets:
#         try:
#             pharma = pharmacophore_to_rdkit(phar_group)
#             mappings = feature_mappings(lig, pharma)
#             if len(mappings) > 0:
#                 atom_match = [list(x.GetAtomIds()) for x in mappings]
#                 lig = Chem.AddHs(lig)
#                 result = align_ligand_to_pharmacophore(lig, atom_match, pharma, num_gen=num_gen)
#                 if result is not None:
#                     aligned_mol, rmsd = result
#                     aligned_mols.extend(aligned_mol)
#                 print("can align in this phar group " + str(phar_group))
#         except:
#             pass
#     assert aligned_mols, "Fail to align into current pharmacophore"
#
# import os
#
# # 创建 SDF 文件夹
# sdf_folder = "SDF"
# os.makedirs(sdf_folder, exist_ok=True)
#
# # 保存每个构象为单独的 SDF 文件
# for i, mol in enumerate(aligned_mols):
#     # 创建一个 SDWriter 对象以保存分子到 SDF 文件
#     sdf_filename = os.path.join(sdf_folder, f"aligned_molecule_{i+1}.sdf")
#     sdf_writer = Chem.SDWriter(sdf_filename)
#
#     # 将 aligned_mol 分子写入 SDF 文件
#     sdf_writer.write(mol)
#
#     # 关闭 SDF 文件
#     sdf_writer.close()


import os
import itertools
from rdkit import Chem
import shutil
from rdkit.Chem import AllChem
import torch

def compute_RMSD(a, b):
    return torch.sqrt((((a - b) ** 2).sum(axis=-1)).mean())

def generate_coords_np(mol):
    # Remove hydrogen atoms
    mol = Chem.RemoveHs(mol)

    AllChem.EmbedMolecule(mol)
    conf = mol.GetConformer()
    coords = [conf.GetAtomPosition(i) for i in range(mol.GetNumAtoms())]
    return torch.tensor([(pos.x, pos.y, pos.z) for pos in coords])

def generate_subsets_with_tolerance(pharmacophoric_points, tolerance):
    all_subsets = []
    all_subsets.append(pharmacophoric_points)

    for subset in itertools.combinations(pharmacophoric_points, len(pharmacophoric_points) - tolerance):
        all_subsets.append(list(subset))

    return all_subsets

def align_ligand_to_pharmacophore_with_tolerance(pdb_filename, pharmacophoric_points, tolerance, num_gen, name):
    lig = Chem.MolFromPDBFile(pdb_filename)

    if tolerance == 0:
        pharma = pharmacophore_to_rdkit(pharmacophoric_points)

        mappings = feature_mappings(lig, pharma)

        assert len(mappings) == 0, "Unable to completely align into the current pharmacophore, try to increase tolerance"

        if len(mappings) > 0:
            atom_match = [list(x.GetAtomIds()) for x in mappings]
            lig = Chem.AddHs(lig)
            result = align_ligand_to_pharmacophore(lig, atom_match, pharma, num_gen=num_gen)
            if result is not None:
                aligned_mols, rmsd = result
    else:
        aligned_mols = []
        phar_points_subsets = generate_subsets_with_tolerance(pharmacophoric_points, tolerance)
        for phar_group in phar_points_subsets:
            try:
                pharma = pharmacophore_to_rdkit(phar_group)
                mappings = feature_mappings(lig, pharma)
                if len(mappings) > 0:
                    atom_match = [list(x.GetAtomIds()) for x in mappings]
                    lig = Chem.AddHs(lig)
                    result = align_ligand_to_pharmacophore(lig, atom_match, pharma, num_gen=num_gen)
                    if result is not None:
                        aligned_mol, rmsd = result
                        aligned_mols.extend(aligned_mol)
                    print("can align in this phar group " + str(phar_group))
            except:
                pass
        #assert aligned_mols, "Fail to align into current pharmacophore"

    # Create SDF folder
    sdf_folder = name
    if os.path.exists(sdf_folder):
        shutil.rmtree(sdf_folder)
    os.makedirs(sdf_folder, exist_ok=True)

    # Save each conformer as a separate SDF file
    min_rmsd = float('inf')
    for i, mol in enumerate(aligned_mols):
        # Create an SDWriter object to save the molecule to an SDF file
        mol = Chem.RemoveHs(mol)
        sdf_filename = os.path.join(sdf_folder, f"aligned_molecule_{i+1}.sdf")
        sdf_writer = Chem.SDWriter(sdf_filename)
        sdf_writer.write(mol)
        # Close the SDF file
        sdf_writer.close()


        coords1_np = generate_coords_np(mol)
        coords2_np = generate_coords_np(lig)

        gen_rmsd = compute_RMSD(coords1_np, coords2_np)
        if gen_rmsd < min_rmsd:
            min_rmsd = gen_rmsd

    return min_rmsd





def load_phar_file(file_path: Path):
    load_file_fn = {'.posp': load_pp_file}.get(file_path.suffix, None)

    if load_file_fn is None:
        raise ValueError(f'Invalid file path: "{file_path}"!')

    return load_file_fn(file_path)

def load_pp_file(file_path: Path):
    # 字符串标识符到新的命名的映射
    phar_mapping = {
        'AROM': 'aromatic ring',
        'HYBL': 'hydrophobicity',
        'POSC': 'positive charge',
        'HACC': 'hb acceptor',
        'HDON': 'hb donor',
        'LHYBL': 'hydrophobicity',
        'UNKONWN': None  # 未知类型，不要该药效团点
    }

    # 读取.posp文件
    pharmacophoric_points = []

    # 使用read_text读取文件内容，并按行和空格拆分
    for line in file_path.read_text().strip().split('\n'):
        types, x, y, z = line.strip().split(' ')
        new_phar_type = phar_mapping.get(types, None)

        if new_phar_type is not None:
            # 将新格式的信息添加到列表中
            point = {
                "feature_name": new_phar_type,
                "center": (float(x), float(y), float(z)),
                "radius": 1
            }
            pharmacophoric_points.append(point)

    return pharmacophoric_points


def read_pdb_files(folder_path):
    folder_path = Path(folder_path)
    assert folder_path.is_dir(), "Input path should be a directory."

    pdb_files = list(folder_path.glob('*.pdb'))

    # Use os.path.join to construct file paths
    pdb_files = [os.path.join(folder_path, file) for file in pdb_files]

    return pdb_files


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--smi', type=str, help='the molecule prepared for association with the pharmacophore aglin')

    parser.add_argument('--lig_path', type=Path, help='the input smi file path. ends with `.sdf` or `.pdb` will be processed')

    parser.add_argument('--input_path', type=Path, help='the input file path. If it is a directory, then every file '
                                                      'ends with `.posp` will be processed')
    parser.add_argument('--output_dir', type=Path, help='the output directory')


    parser.add_argument('--phar_tolerance', type=int, default=0, help='The degree of matching with the pharmacophore, '
                                                                      'which defaults to 0, must match completely otherwise an error will be reported. '
                                                                      'The user can adjust the tolerance by adjusting the tolerance. ')

    parser.add_argument('--num_gen', type=int, default=1, help='The number of conformations generated by each group of pharmacophore')


    args = parser.parse_args()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    if args.input_path.is_dir():
        phar_files = list(args.input_path.glob('*.posp'))
    else:
        assert args.input_path.suffix in ('.posp')
        phar_files = [args.input_path]

    pharm_groups = []
    for file in phar_files:
        pharm_group = load_phar_file(file)
        pharm_groups.append(pharm_group)

    rmsd = []
    for i, pharm_group in enumerate(pharm_groups):
        if args.smi is not None:
            align_ligand_to_pharmacophore_with_tolerance(args.smi, pharm_group, args.phar_tolerance,
                                                         args.num_gen, name=f"lig_phar{i}")
        if args.lig_path is not None:
            file_path = args.lig_path


            pdb_lig_path_list = read_pdb_files(args.lig_path)


            for j, lig_file in enumerate(pdb_lig_path_list):
                try:
                    file_name = os.path.splitext(os.path.basename(lig_file))[0]
                    min_rmsd = align_ligand_to_pharmacophore_with_tolerance(lig_file, pharm_group, args.phar_tolerance,
                                                                 args.num_gen, name=f"lig{file_name}_phar{i}")
                    rmsd.append(min_rmsd)
                except:
                    pass

    print(rmsd)

    # Remove 'inf' values from the list
    filtered_rmsd_values = [float(i) for i in rmsd if i != float('inf')]
    output_file_path = os.path.join(args.output_dir, 'rmsd_values.npy')

    # Save the NumPy array
    if filtered_rmsd_values:
        np.save(output_file_path, np.array(filtered_rmsd_values))
        print("NumPy array saved successfully.")
    else:
        print("No valid values to save.")















