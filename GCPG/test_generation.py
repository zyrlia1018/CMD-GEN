import argparse
import pickle
from pathlib import Path
import time

import dgl
import rdkit
import torch
from rdkit import RDLogger
from tqdm.auto import tqdm

from model.gcpg import GCPG
from utils.file_utils import load_phar_file
from utils.utils import AverageMeter, timeSince, seed_torch
from utils.dataset import Tokenizer, SemiSmilesDataset

import torch
import torch.multiprocessing
import torch.cuda.amp as amp
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau

from multiprocessing import Pool, TimeoutError
from multiprocessing.dummy import Pool as ThreadPool
import numpy as np


RDLogger.DisableLog('rdApp.*')

MODEL_SETTINGS = {
    # default
    'rs_mapping': {'non_vae': False, 'remove_pp_dis': False, 'in': 'rs', 'out': 'rs'},
    # others
    'cs_mapping': {'in': 'cs', 'out': 'cs'},
    'non_vae': {'non_vae': True},
    'remove_pp_dis': {'remove_pp_dis': True},
}

class CFG:
    fp16 = False  # whether to train with mixed precision (may have some bugs)

    # GENERAL SETTING
    print_freq = 200  # log frequency
    num_workers = 20

    # TRAINING
    init_lr = 3e-4
    weight_decay = 1e-6
    min_lr = 1e-6  # for CosineAnnealingLR scheduler
    T_max = 4  # for CosineAnnealingLR scheduler

    max_grad_norm = 5

    epochs = 64
    batch_size = 128
    gradient_accumulation_steps = 1
    valid_batch_size = 512
    valid_size = None  # can be used to set a fixed size validation dataset
    # we generated some molecules during training to track metrics like Validity
    gen_size = 2048  # number of pharmacophore graphs used to generate molecules during training
    gen_repeat = 2  # number of generated molecules for each input
    # the total number of generated molecules each time is `gen_size`*`gen_repeat`

    seed = 42  # random seed
    n_fold = 20  # k-fold validation
    valid_fold = 0  # which fold is used to as the validation dataset

    #n_device = len(os.environ['CUDA_VISIBLE_DEVICES'].split(','))  # not used

    save_freq = 4  # save model every `save_freq` epochs
    skip_gen_test = 12  # skip saving and track Validity for `skip_gen_test` epochs

    # settings for reloading model and continue training
    init_epoch = 0  # 16
    reload_path = None  # './output/chembl_test/rs_mapping/fold0_epoch16.pth'
    reload_ignore = []


if CFG.init_epoch > 0:
    CFG.init_epoch -= 1



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

@torch.no_grad()
def test_generate(valid_loader, model, random_sampling=False):
    from utils.match_eval import get_match_score

    # switch to eval mode
    model.eval()

    start = end = time.time()

    res = []
    pp_graph_list = []

    for step, batch_data in tqdm(enumerate(valid_loader), disable=disable_tqdm, total=len(valid_loader)):
        inputs, input_mask, pp_graphs, mappings, targets, MW, logP, QED, SAS, \
        HBA, HBD, RotaNumBonds, Score, Smi,*others = [i.to('cuda:0') for i in batch_data]

        conditions = [MW, logP, QED, SAS, RotaNumBonds, Score, Smi]
        predictions = model.generate(pp_graphs=pp_graphs, conditions=conditions, random_sample=random_sampling)
        res.extend(tokenizer.get_text(predictions))
        pp_graph_list.extend(dgl.unbatch(pp_graphs.to('cpu')))

    match_score = get_match_score(pp_graph_list, res, n_workers=CFG.num_workers, timeout=10)
    #print(match_score)
    with Pool(CFG.num_workers) as pool:
        v_smiles = pool.map(format_smiles, res)

    valid_smiles = [i for i in v_smiles if i is not None]
    s_valid_smiles = set(valid_smiles)
    uniqueness = len(s_valid_smiles) / len(valid_smiles)
    novelty = len(s_valid_smiles - all_smiles) / len(s_valid_smiles)

    timeout_count = 0
    exceptions = 0
    error = 0
    for i in match_score:
        timeout_count += i == -2
        exceptions += i == -3
        error += i == -1
    valid_match_score = [i for i in match_score if i >= 0]
    np.save('valid_match_score_gcpg_egat.npy', np.array(valid_match_score))
    end = time.time()

    print(
          f'Time: {end - start} '
          f'Match Score: {np.mean(valid_match_score):.4f} '
          f'Validity: {(len(valid_smiles) / len(res)):.4f} '
          f'Uniqueness: {uniqueness:.4f} '
          f'Novelty: {novelty:.4f} '
          f'TimeoutCount: {timeout_count} '
          f'Exceptions: {exceptions} '
          )

    return np.mean(valid_match_score)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('model_path', type=Path, help='the weights file (xxx.pth)')
    parser.add_argument('tokenizer_path', type=Path, help='the saved tokenizer (tokenizer.pkl)')
    parser.add_argument('--device', type=str, default='cpu', help='`cpu` or `cuda`')
    parser.add_argument('--model_type', choices=['rs_mapping', 'cs_mapping', 'non_vae', 'remove_pp_dis'],
                        default='rs_mapping')
    parser.add_argument('--show_progressbar', action='store_true')

    args = parser.parse_args()

    disable_tqdm = not args.show_progressbar
    model, tokenizer = load_model(args.model_path, args.tokenizer_path)

    model.eval()
    model.to(args.device)

    with open('/home/data/zou/MSMG/msmg/data_prepare/filtered_train_data.pickle', 'rb') as f:
        train_properties = pickle.load(f)
        train_smiles = train_properties["smiles"]
        # train_MW = train_properties["MW"]
        # train_logP = train_properties["logP"]
        # train_qed = train_properties["qed"]
        # train_SAS = train_properties["SAS"]
        # train_HBA = train_properties["HBA"]
        # train_HBD = train_properties["HBD"]
        # train_RotaNumBonds = train_properties["RotaNumBonds"]
        # train_Score = train_properties["score"]
        # train_Strain = train_properties["strain"]
    with open('/home/data/zou/MSMG/msmg/data_prepare/filtered_valid_data.pickle', 'rb') as f:
        valid_properties = pickle.load(f)
        valid_smiles = valid_properties["smiles"]
        # valid_MW = valid_properties["MW"]
        # valid_logP = valid_properties["logP"]
        # valid_qed = valid_properties["qed"]
        # valid_SAS = valid_properties["SAS"]
        # valid_HBA = valid_properties["HBA"]
        # valid_HBD = valid_properties["HBD"]
        # valid_RotaNumBonds = valid_properties["RotaNumBonds"]
        # valid_Score = valid_properties["score"]
        # valid_Strain = valid_properties["strain"]
    with open('/home/data/zou/MSMG/msmg/data_prepare/filtered_test_data.pickle', 'rb') as f:
        test_properties = pickle.load(f)
        test_smiles = test_properties["smiles"]
        test_MW = test_properties["MW"]
        test_logP = test_properties["logP"]
        test_qed = test_properties["qed"]
        test_SAS = test_properties["SAS"]
        test_HBA = test_properties["HBA"]
        test_HBD = test_properties["HBD"]
        test_RotaNumBonds = test_properties["RotaNumBonds"]
        test_Score = test_properties["score"]
        test_Strain = test_properties["strain"]
    
    import random
    # Set the random seed for reproducibility
    random_seed = 226
    random.seed(random_seed)

    # Combine the data into a list of tuples for easy shuffling
    combined_data = list(zip(test_smiles, test_MW, test_logP, test_qed, test_SAS, test_HBA, test_HBD, test_RotaNumBonds, test_Score, test_Strain))

    # Shuffle the combined data
    random.shuffle(combined_data)

    # Take the first 1000 items after shuffling
    selected_data = combined_data[:1000]

    # Unpack the selected data into separate lists
    test_smiles, test_MW, test_logP, test_qed, test_SAS, test_HBA, test_HBD, test_RotaNumBonds, test_Score, test_Strain = map(list,zip(*selected_data))


    all_smiles = set(train_smiles + valid_smiles + test_smiles)

    use_random_input_smiles = MODEL_SETTINGS[args.model_type].setdefault('in', 'rs') == 'rs'
    use_random_target_smiles = MODEL_SETTINGS[args.model_type].setdefault('out', 'rs') == 'rs'

    gen_dataset = SemiSmilesDataset(test_smiles, test_MW, test_logP, test_qed, test_SAS,
                                    test_HBA, test_HBD, test_RotaNumBonds, test_Score,
                                    test_Strain, tokenizer,
                                    use_random_input_smiles, use_random_target_smiles)

    gen_loader = DataLoader(gen_dataset,
                            batch_size=CFG.valid_batch_size,
                            shuffle=False,
                            num_workers=CFG.num_workers,
                            pin_memory=True,
                            drop_last=False,
                            collate_fn=gen_dataset.collate_fn)

    mean_match_score = test_generate(gen_loader, model)

    print('done')

