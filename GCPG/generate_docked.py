import argparse
import pickle
from pathlib import Path

import dgl
import rdkit
import torch
from rdkit import RDLogger
from tqdm.auto import tqdm

from model.gcpg import GCPG
from utils.file_utils import load_phar_file
from utils.utils import seed_torch

RDLogger.DisableLog('rdApp.*')

def load_model(model_path, tokenizer_path):
    with open(tokenizer_path, 'rb') as f:
        tokenizer = pickle.load(f)

    model_params = {
        "max_len": 128,
        "cond_dim": 7,
        "pp_v_dim": 7 + 1,
        "pp_e_dim": 1,
        "pp_encoder_n_layer": 4,
        "hidden_dim": 384,
        "n_layers": 8,
        "ff_dim": 1024,
        "n_head": 8,
    }

    model = GCPG(model_params, tokenizer)
    states = torch.load(model_path, map_location='cpu')
    print(model.load_state_dict(states['model'], strict=False))

    return model, tokenizer

def format_smiles(smiles):
    mol = rdkit.Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    smiles = rdkit.Chem.MolToSmiles(mol, isomericSmiles=True)

    return smiles


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('input_path', type=Path, help='the input file path. If it is a directory, then every file '
                                                      'ends with `.edgep` or `.posp` will be processed')
    parser.add_argument('output_dir', type=Path, help='the output directory')
    parser.add_argument('model_path', type=Path, help='the weights file (xxx.pth)')
    parser.add_argument('tokenizer_path', type=Path, help='the saved tokenizer (tokenizer.pkl)')

    parser.add_argument('--n_mol', type=int, default=200, help='number of generated molecules for each '
                                                                 'pharmacophore file')
    parser.add_argument('--device', type=str, default='cpu', help='`cpu` or `cuda`')
    parser.add_argument('--filter', action='store_true', help='whether to save only the unique valid molecules')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--seed', type=int, default=-1)

    args = parser.parse_args()

    if args.seed != -1:
        seed_torch(args.seed)

    if args.input_path.is_dir():
        files = list(args.input_path.glob('*.posp')) + list(args.input_path.glob('*.edgep'))
    else:
        assert args.input_path.suffix in ('.edgep', '.posp')
        files = [args.input_path]

    args.output_dir.mkdir(parents=False, exist_ok=True)

    model, tokenizer = load_model(args.model_path, args.tokenizer_path)

    model.eval()
    model.to(args.device)

    # 遍历文件列表
    for file in files:
        output_path = args.output_dir / f'{file.stem}_result.txt'
        g = load_phar_file(file)
        g_batch = [g] * args.batch_size
        g_batch = dgl.batch(g_batch).to(args.device)
        n_epoch = (args.n_mol + args.batch_size - 1) // args.batch_size

        res_total = []
        MW_min, MW_max, MW_step = 0, 0, 1
        logP_min, logP_max, logP_step = 0, 0, 1
        QED_min, QED_max, QED_step = 0, 0, 1
        SAS_min, SAS_max, SAS_step = 0, 0, 1
        RotaNumBonds_min, RotaNumBonds_max, RotaNumBonds_step = 0, 0, 1
        Score_min, Score_max, Score_step = -14, -14, 1
        Smi_min, Smi_max, Smi_step = 0, 0, 1

        for MW in torch.arange(MW_min, MW_max + 1, MW_step, dtype=torch.float32):
            for logP in torch.arange(logP_min, logP_max + 0.1, logP_step, dtype=torch.float32):
                for QED in torch.arange(QED_min, QED_max + 0.1, QED_step, dtype=torch.float32):
                    for SAS in torch.arange(SAS_min, SAS_max + 0.1, SAS_step, dtype=torch.float32):
                        for RotaNumBonds in range(RotaNumBonds_min, RotaNumBonds_max + 1, RotaNumBonds_step):
                            for Score in range(Score_min, Score_max + 1, Score_step):
                                for Smi in range(Smi_min, Smi_max + 1, Smi_step):
                                    MW_tensor = torch.tensor([MW], dtype=torch.float32).repeat(1, args.batch_size)
                                    logP_tensor = torch.tensor([logP], dtype=torch.float32).repeat(1, args.batch_size)
                                    QED_tensor = torch.tensor([QED], dtype=torch.float32).repeat(1, args.batch_size)
                                    SAS_tensor = torch.tensor([SAS], dtype=torch.float32).repeat(1, args.batch_size)
                                    RotaNumBonds_tensor = torch.tensor([RotaNumBonds], dtype=torch.float32).repeat(1,
                                                                                                                   args.batch_size)
                                    Score_tensor = torch.tensor([Score], dtype=torch.float32).repeat(1, args.batch_size)
                                    Smi_tensor = torch.tensor([Smi], dtype=torch.float32).repeat(1,
                                                                                                       args.batch_size)

                                    conditions = [MW_tensor, logP_tensor, QED_tensor, SAS_tensor, 
                                                  RotaNumBonds_tensor, Score_tensor, Smi_tensor]
                        
                                    res = []                     
                
                                    for i in tqdm(range(n_epoch)):
                                        res.extend(
                                            tokenizer.get_text(model.generate(pp_graphs=g_batch, conditions=conditions)))
                
                                    res = res[:args.n_mol]
                                    res_total.extend(res)

        if args.filter:
            res_total = [format_smiles(i) for i in res_total]
            res_total = [i for i in res_total if i]
            res_total = list(set(res_total))

        output_path.write_text('\n'.join(res_total))

    print('done')

