
import torch
from torch import nn
import pickle
from qm9 import utils as qm9_utils
from models.vit import ViT
from qm9.models import EGNN
import numpy as np

device = torch.device("cuda")
dtype = torch.float32

from models.decoder import LatentToMol

def set_up_causal_mask(seq_len):
    mask = (torch.triu(torch.ones(seq_len, seq_len)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    mask.requires_grad = False
    return mask


class CLIP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.vocab = pickle.load(open(config['data']['vocab_path'], 'rb'))
        self.temperature = config['train']['temperature']
        self.max_charge = config['data']['max_charge']
        self.num_species = config['data']['num_species']
        
        self.Molecule_Encoder = EGNN(
             in_node_nf = self.config['molecule_encoder']['in_node_nf'], 
             in_edge_nf = self.config['molecule_encoder']['in_edge_nf'], 
             hidden_nf = self.config['molecule_encoder']['hidden_nf'], 
             device = torch.device(self.config['molecule_encoder']['device']), 
             n_layers = self.config['molecule_encoder']['n_layers'], 
             coords_weight = self.config['molecule_encoder']['coords_weight'],
             attention = self.config['molecule_encoder']['attention'], 
             node_attr = self.config['molecule_encoder']['node_attr'],
            output_size = self.config['molecule_encoder']['output_size'],
        )
        
        self.Spectra_Encoder = ViT(
            patch_size = self.config['spectra_encoder']['patch_size'], 
            num_layers = self.config['spectra_encoder']['num_layers'], 
            h_dim = self.config['spectra_encoder']['h_dim'], 
            num_heads = self.config['spectra_encoder']['num_heads'], 
            output_size = self.config['spectra_encoder']['output_size'], 
            d_ff=self.config['spectra_encoder']['d_ff'], 
            max_time_steps=self.config['spectra_encoder']['max_time_steps'], 
            use_clf_token=self.config['spectra_encoder']['use_clf_token'],
            dropout = self.config['spectra_encoder']['dropout'],
            dropout_emb = self.config['spectra_encoder']['dropout_emb']   
        )
        
        self.smiles_decoder = LatentToMol(
            in_size=self.config['molecule_decoder']['latent_size'],
            hidden_size=self.config['molecule_decoder']['hidden_size'], 
            n_layers=self.config['molecule_decoder']['n_layers'], 
            n_heads = self.config['molecule_decoder']['n_heads'],
            seq_len=self.config['data']['seq_len'], 
            vocab = self.vocab)
        
        self.logit_scale = nn.Parameter(torch.ones([]) * self.temperature)
        
    
    def forward_mol(self, data):
        batch_size = self.config['data']['batch_size']
        batch_size, n_nodes, _ = data['positions'].size()
        atom_positions = data['positions'].view(batch_size * n_nodes, -1).to(device, dtype)
        atom_mask = data['atom_mask'].view(batch_size * n_nodes, -1).to(device, dtype)
        edge_mask = data['edge_mask'].to(device, dtype)
        one_hot = data['one_hot'].to(device, dtype)
        charges = data['charges'].to(device, dtype)
        
        charge_scale = 9
    
        nodes = qm9_utils.preprocess_input(one_hot, 
                                    charges,
                                    2,
                                    charge_scale,
                                    device)

        nodes = nodes.view(batch_size * n_nodes, -1)
        edges = qm9_utils.get_adj_matrix(n_nodes, batch_size, device)
        
        mol_features = self.Molecule_Encoder(h0=nodes, 
             x=atom_positions, 
             edges=edges, 
             edge_attr=None, 
             node_mask=atom_mask, 
             edge_mask=edge_mask,
             n_nodes=n_nodes)
        
        mol_features = mol_features / mol_features.norm(dim=1, keepdim=True)
        
        return mol_features
    
    def forward_spec(self, data):
        spectra = data['IR'].to(device, dtype)
        spectra = torch.unsqueeze(spectra, 1)
        spectra = torch.unsqueeze(spectra, 1)
        
        spectra_features = self.Spectra_Encoder(spectra)
        spectra_features = spectra_features / spectra_features.norm(dim=1, keepdim=True)
        
        return spectra_features
    
    def forward_decoder(self, data, spec_latents):
        smi = data['smiles'].to(device)[:,:-1]        
        pred = self.smiles_decoder(spec_latents, smi)
        return pred
        
    def forward(self, data):
        logits_scale = self.logit_scale.exp()
        
        mol_latents = self.forward_mol(data)
        spec_latents = self.forward_spec(data)
        
        mean = 0
        std = 0.005
        noise = torch.tensor(np.random.normal(mean, std, spec_latents.size()), dtype=torch.float).to(device)
        
        # spec_latents += noise
        
        smile_preds = self.forward_decoder(data, spec_latents)
        
        # smile_preds = self.forward_decoder(data, mol_latents)
        return mol_latents, spec_latents, smile_preds, logits_scale, data['index'] 
        
        
        
        