

# Create a data loader that has the data objects for EGNN, VIT, TrfmDecoder

from qm9 import dataset
from qm9.models import EGNN
import torch

from torch import nn, optim
from torch.nn import functional as F

import argparse
from qm9 import utils as qm9_utils
import utils
import json
import numpy as np
import pickle
from qm9.data.utils import _get_species, initialize_datasets
from torch.utils.data import DataLoader, Dataset
from qm9.data.dataset import ProcessedDataset
from qm9.data.prepare import prepare_dataset
from torch.utils.data import DataLoader
from qm9.data.utils import initialize_datasets
from qm9.args import init_argparse
from qm9.data.collate import collate_fn
import pickle
from tqdm.auto import tqdm

class dummy_dataset(Dataset):
    def __init__(self, data_list,  num_species, max_charge ):
        self.data_list = data_list
        self.max_charge = max_charge
        self.num_species = num_species
        
    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, item):
        return self.data_list[item]
    
def combine_datasets(datafiles):
    datasets = {}
    for split, datafile in datafiles.items():
        datasets[split] = pickle.load(open(datafile, 'rb'))

    keys = [list(data.keys()) for data in datasets.values()]
    assert all([key == keys[0] for key in keys]), 'Datasets must have same set of keys!'

    all_species = _get_species(datasets, ignore_check=False)

    num_pts = {'train': datasets['train']['index'].shape[0],
            'test': datasets['test']['index'].shape[0], 
            'valid': datasets['val']['index'].shape[0]}
    
    datasets = {split: ProcessedDataset(data, num_pts=num_pts.get(
        split, -1), included_species=all_species, subtract_thermo=False) for split, data in datasets.items()}
    
    ls = []
    for i, data in enumerate(datasets['train']):
        ls.append(data)    
    for i, data in enumerate(datasets['test']):
        ls.append(data)
    for i, data in enumerate(datasets['val']):
        ls.append(data)
    full_dataset = dummy_dataset(ls, 
                                 datasets['train'].num_species, 
                                 datasets['train'].max_charge)
    return full_dataset       

def id_data_map(path):
    qm9_broad_ir = pickle.load(open(path, 'rb'))
    smiles_id_map = {}
    for id, row in tqdm(qm9_broad_ir.iterrows()):
        smiles_id_map[int(row['ID'].split('_')[1])] = row['SMILES']
        
    id_ir_map = {}
    for id, row in tqdm(qm9_broad_ir.iterrows()):
        id_ir_map[int(row['ID'].split('_')[1])] = row['IR_Data']

    del qm9_broad_ir
    return smiles_id_map, id_ir_map

# use the smiles-transformer build_vocab and build_dataset for generating
# qm9_corpus.txt and qm9_vocab.pkl

from build_vocab import WordVocab
import argparse
import pandas as pd
import numpy as np
from sklearn import preprocessing
import numpy as np

def normalize_data(id_ir_map, type="unit", individual_norm=True):
    if individual_norm:
        print("Normalizing each spectrum individually", flush=True)
        for i in id_ir_map:
            ir = id_ir_map[i]
            ir = np.array(ir)
            ir = ir/ir.sum()
            id_ir_map[i] = ir
            
    irs = []
    for i in id_ir_map:
        irs.append(id_ir_map[i])
        
    irs = np.array(irs)
    min_max_scaler = preprocessing.MinMaxScaler()
    standard_scaler = preprocessing.StandardScaler()

    irs_minmax = min_max_scaler.fit_transform(irs)
    irs_standard = standard_scaler.fit_transform(irs)
    irs_unitnorm = preprocessing.normalize(irs, norm='l2')

    new_dict_minmax = {}
    new_dict_standard = {}
    new_dict_unitnorm = {}
    # new_dict_original = {}
    for i, j in enumerate(id_ir_map):
        new_dict_minmax[j] = irs_minmax[i]
        new_dict_standard[j] = irs_standard[i]
        new_dict_unitnorm[j] = irs_unitnorm[i]
    
    if type == "minmax":
        return new_dict_minmax, min_max_scaler
    elif type == "unit":
        return new_dict_unitnorm, standard_scaler
    else:
        print("Not normalizing", flush=True)
        return id_ir_map, None
    
from dataset import Randomizer
import random
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

from enumerator import SmilesEnumerator
from utils_decoder import split

def set_up_causal_mask(seq_len):
    mask = (torch.triu(torch.ones(seq_len, seq_len)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    mask.requires_grad = False
    return mask

def set_up_padding_mask(max_len, no_of_words):
    tgt_padding_mask = torch.ones([max_len, ])
    tgt_padding_mask[:no_of_words] = 0.0
    tgt_padding_mask = tgt_padding_mask.bool()
    return tgt_padding_mask

class ParentDataset(Dataset):
    def __init__(self, 
                 clip_dataset, 
                 max_charge,
                 num_species,
                 smiles_id_map, 
                 vocab, 
                 ir_dict_norm, 
                 seq_len=70,
                 transform=Randomizer()
                 ):
        self.clip_dataset = clip_dataset
        self.smiles_id_map = smiles_id_map
        self.ir_dict_norm = ir_dict_norm
        self.vocab = vocab
        self.seq_len = seq_len
        self.transform = transform
        self.max_charge = max_charge
        self.num_species = num_species
        if self.transform:
            print("SMILES WILL BE RANDOMIZED")
        
    def __len__(self):
        return len(self.clip_dataset)
    
    def __getitem__(self, item):
        data = self.clip_dataset[item]
        
        sm = self.smiles_id_map[data['index'].item()]
        
        if self.transform:
            sm = self.transform(sm)
        
        ir = self.ir_dict_norm[data['index'].item()]
        
        content = [self.vocab.stoi.get(token, self.vocab.unk_index) for token in sm]
        X = [self.vocab.sos_index] + content + [self.vocab.eos_index]
        
        inp_tokens = X[:-1].copy()
        tgt_tokens = X[1:].copy()
        
        sample_size = len(inp_tokens)
        
        tgt_padding_mask = set_up_padding_mask(self.seq_len, sample_size)
        
        padding = [self.vocab.pad_index]*(self.seq_len - sample_size)
        X.extend(padding)
        inp_tokens.extend(padding)
        tgt_tokens.extend(padding)
        
        return {
            "index": data['index'],
            # "decoder_inp":torch.tensor(inp_tokens),
            # "decoder_tgt":torch.tensor(tgt_tokens),
            "IR":torch.tensor(ir),
            # "tgt_padding_mask":tgt_padding_mask,
            "num_atoms": data['num_atoms'],
            "charges":data["charges"],
            "positions" : data['positions'],
            "one_hot" : data['one_hot'],
            "smiles": torch.tensor(X)
        }
        
def CreateDataloaders( dataset, sizes = [0.8, 0.1, 0.1], batch_size=128, num_workers=16, shuffle=True):
    
    # train_size = int(len(dataset)*sizes[0])
    # test_size = int(len(dataset)*(sizes[1]))
    # val_size = len(dataset) - train_size - test_size
    # train_size = 97792
    # val_size = 12288
    train_size = 100000
    val_size = 20000
    test_size = len(dataset) - train_size - val_size
    
    train , test, val = torch.utils.data.random_split(dataset, [train_size, test_size, val_size])
    
    datasets = {'train':train, 
                'test': test,
                'val':val}
    
    dataloaders = {split: DataLoader(dataset,
                                batch_size=batch_size,
                                 shuffle=shuffle ,
                                 num_workers=num_workers,
                                 collate_fn=collate_fn)
                                 for split, dataset in datasets.items()}
    return dataloaders            

def prepare_data(config):
    vocab = WordVocab.load_vocab(config['data']['vocab_path'])

    PAD = vocab.pad_index # 0 
    UNK = vocab.unk_index # 1
    EOS = vocab.eos_index # 2
    SOS = vocab.sos_index # 3
    MASK = vocab.mask_index # 5

    smiles_id_map, ir_id_map = id_data_map(config['data']['qm9_broad_ir_path'])
    new_dict_norm, scaler = normalize_data(id_ir_map=ir_id_map, type=config['data']['normalization'])
    full_dataset = combine_datasets(datafiles=config['data']['datafiles'])
    final_dataset = ParentDataset(clip_dataset=full_dataset,
                                  max_charge=full_dataset.max_charge,
                                  num_species=full_dataset.num_species,
                                  smiles_id_map=smiles_id_map, 
                                  ir_dict_norm=new_dict_norm, 
                                  seq_len=config['data']['seq_len'],
                                  vocab=vocab,
                                  transform=Randomizer()
                                  )
    dataloaders = CreateDataloaders(final_dataset,
                                    sizes=config['data']['splits'], 
                                    batch_size=config['data']['batch_size'],
                                    num_workers=config['data']['num_workers'],
                                    shuffle=config['data']['shuffle']
                                    )
    
    return dataloaders, full_dataset.max_charge, full_dataset.num_species, scaler

#====================USAGE==================#

# config = {}
# datafiles = {
#     'train': '/home2/kanakala.ganesh/ir_data/raw_train.pickle',
#     'test':  '/home2/kanakala.ganesh/ir_data/raw_test.pickle',
#     'val':   '/home2/kanakala.ganesh/ir_data/raw_val.pickle'
# }
# config['data'] = {"qm9_broad_ir_path":'/home2/kanakala.ganesh/ir_data/qm9_broad_ir.pkl',
#                   "vocab_path":'/home2/kanakala.ganesh/CLIP_PART_1/data/qm9_vocab.pkl',
#                   "datafiles" : datafiles,
#                   "normalization" : "unit",
#                   "shuffle": True,
#                   "batch_size":128,
#                   "seq_len":70,
#                   "splits":[0.9, 0.1, 0.1],
#                   "num_workers":16
#                 }

# dataloaders , mc, ns = prepare_data(config)

# for i, data in enumerate(dataloaders['train']):
#     for k in data:
#         print(k, data[k].shape)
#     break

# outputs 

# index torch.Size([128])
# decoder_inp torch.Size([128, 70])
# decoder_tgt torch.Size([128, 70])
# IR torch.Size([128, 1801])
# tgt_padding_mask torch.Size([128, 70])
# num_atoms torch.Size([128])
# charges torch.Size([128, 29])
# positions torch.Size([128, 29, 3])
# atom_mask torch.Size([128, 29])
# edge_mask torch.Size([107648, 1])
