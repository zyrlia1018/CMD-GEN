import argparse
from pathlib import Path
import json
import torch

import utils
from lightning_modules import PharPocketDDPM


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('checkpoint', type=Path)
    parser.add_argument('--pdbfile', type=str)
    parser.add_argument('--resi_list', type=str, nargs='+', default=None)
    parser.add_argument('--ref_ligand', type=str, default=None)
    parser.add_argument('--outdir', type=Path)
    parser.add_argument('--n_samples', type=int, default=20)  # 每个药效团的个数
    parser.add_argument('--num_nodes_phar', type=int, default=3) # 采样的个数
    parser.add_argument('--all_frags', action='store_true')
    parser.add_argument('--sanitize', action='store_true')
    parser.add_argument('--relax', action='store_true')
    parser.add_argument('--resamplings', type=int, default=10)
    parser.add_argument('--jump_length', type=int, default=1)
    parser.add_argument('--timesteps', type=int, default=None)
    args = parser.parse_args()

    pdb_id = Path(args.pdbfile).stem

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Load model
    model = PharPocketDDPM.load_from_checkpoint(
        args.checkpoint, map_location=device)
    model = model.to(device)

    if args.num_nodes_phar is not None:
        num_nodes_phar = torch.ones(args.n_samples, dtype=int) * \
                        args.num_nodes_phar
    else:
        num_nodes_phar = None

    phar_to_coords = model.generate_phars(
        args.pdbfile, args.n_samples, args.resi_list, args.ref_ligand,
        num_nodes_phar, args.sanitize, largest_frag=not args.all_frags,
        relax_iter=(200 if args.relax else 0),
        resamplings=args.resamplings, jump_length=args.jump_length,
        timesteps=args.timesteps)
     
    # 转换张量为列表
    phar_to_coords_no_tensor = {}
    for molecule_name, features in phar_to_coords.items():
        phar_to_coords_no_tensor[molecule_name] = {}
        for feature_type, coordinates_list in features.items():
            phar_to_coords_no_tensor[molecule_name][feature_type] = [coord.tolist() for coord in coordinates_list]

    # 保存数据到文件
    output_file_path = Path('phar_to_coords_no_tensor_PI3K_dul.json')
    with open(output_file_path, 'w') as f:
        # 添加转换为 NumPy 数组的步骤
        json.dump(phar_to_coords_no_tensor, f, default=lambda x: x.tolist() if isinstance(x, np.ndarray) else x)

      
    # 保存数据到文件
    # Make SDF files
    print(phar_to_coords)
    # utils.write_sdf_file(Path(args.outdir, f'{pdb_id}_mol.sdf'), pharmocophore)
