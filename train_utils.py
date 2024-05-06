
from PrepareData import prepare_data

import torch
from torch import nn, optim, Tensor
from torch.nn import functional as F
import pickle 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import rdkit
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import Draw
import seaborn as sns
from architecture import CLIP
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.float32

import os
import yaml

import wandb
import time

def freeze_molecule_encoder(model):
    for name, param in model.named_parameters():
        if "Molecule_Encoder" in name:
            param.requires_grad = False
        print(name, param.requires_grad)

def unfreeze_molecule_encoder(model):
    for name, param in model.named_parameters():
        if "Molecule_Encoder" in name:
            param.requires_grad = True
        print(name, param.requires_grad)
        
def freeze_spectra_encoder(model):
    for name, param in model.named_parameters():
        if "Spectra_Encoder" in name:
            param.requires_grad = False
        print(name, param.requires_grad)

def unfreeze_spectra_encoder(model):
    for name, param in model.named_parameters():
        if "Spectra_Encoder" in name:
            param.requires_grad = True
        print(name, param.requires_grad)
        
def freeze_smiles_decoder(model):
    for name, param in model.named_parameters():
        if "smiles_decoder" in name:
            param.requires_grad = False
        print(name, param.requires_grad)

def unfreeze_smiles_decoder(model):
    for name, param in model.named_parameters():
        if "smiles_decoder" in name:
            param.requires_grad = True
        print(name, param.requires_grad)
          
def nt_xent_loss(out_1, out_2, temperature):
    out = torch.cat([out_1, out_2], dim=0)
    n_samples = len(out)

    # Full similarity matrix
    cov = torch.mm(out, out.t().contiguous())
    sim = torch.exp(cov / temperature)

    mask = ~torch.eye(n_samples, device=sim.device).bool()
    neg = sim.masked_select(mask).view(n_samples, -1).sum(dim=-1)

    # Positive similarity
    pos = torch.exp(torch.sum(out_1 * out_2, dim=-1) / temperature)
    pos = torch.cat([pos, pos], dim=0)

    loss = -torch.log(pos / neg).mean()
    return loss

class CombinedLoss(nn.Module):
    # under construction 
    def __init__(self, vocab, temperature=1, threshold=0.8, type="default"):
        super().__init__()
        self.temperature = temperature
        self.threshold = threshold
        self.vocab = vocab
        self.type = type
    
    def forward(self, mol_features, spectra_features, logit_scale, smile_ypred, data):
        # spectra = spectra.squeeze(1)
        # spectra = spectra.squeeze(1)
        # print(logit_scale)
        # print(mol_features.shape)
        # print(spectra_features.shape)
        if self.type == "nt_xent":
            clip_loss = nt_xent_loss(mol_features, spectra_features, self.temperature)
        
        elif self.type == "default":
            logits = logit_scale[0] *  mol_features @ spectra_features.t() 
            targets = torch.diag(torch.ones(spectra_features.shape[0])).to(device)
            clip_loss = (F.cross_entropy(logits, targets) + 
                        F.cross_entropy(logits.t(), targets.t())
                        ) / 2
        
        smile_y = data['smiles'].to(device)[:,1:]
        smile_yprob = F.log_softmax(smile_ypred, dim=2)
        # reconstruction_loss = F.nll_loss(smile_yprob.view(-1, len(self.vocab)),
        #                                 smile_y.view(-1))
        reconstruction_loss = F.cross_entropy(smile_ypred.view(-1, len(self.vocab)),
                                              smile_y.contiguous().view(-1))
        total_loss = clip_loss + reconstruction_loss
        
        return total_loss, clip_loss, reconstruction_loss

def train_one_epoch( config, model, dataloader, epoch, optimizer, loss_fn, focus="clip_loss"):
    
    running_loss = []
    model.to(device)
    model.train()
    max_charge = config['data']['max_charge']
    num_species = config['data']['num_species']
    for i, data in enumerate(dataloader):
        
        optimizer.zero_grad()
        data = {k: v.to(device) for k, v in data.items()}
        
        mol_latents, spec_latents, smile_preds, logit_scale, ids = model(data)
        total_loss, clip_loss, reconstruction_loss = loss_fn(mol_latents, spec_latents, logit_scale, smile_preds, data)
        
        if focus == "total_loss":
            total_loss.backward()
        elif focus == "clip_loss":
            clip_loss.backward()
        elif focus == "reconstruction_loss":
            reconstruction_loss.backward()
            
        optimizer.step()
        nn.utils.clip_grad_value_(model.parameters(), clip_value=1.0)

        print( 'Training Epoch: {} | iteration: {}/{} | Loss: {}'.format(epoch, i, len(dataloader), total_loss.item() ), end='\r')
        running_loss.append([total_loss.detach().item(), clip_loss.detach().item(), reconstruction_loss.detach().item()])
        del total_loss, clip_loss, reconstruction_loss, mol_latents, spec_latents, smile_preds, logit_scale
        
    running_loss = np.array(running_loss)
    return np.mean(running_loss, axis= 0)

def validate(config, model, dataloader, epoch, optimizer, loss_fn):
    
    running_loss = []
    # model.to(device)
    model.eval()
    max_charge = config['data']['max_charge']
    num_species = config['data']['num_species']
    
    with torch.no_grad():
        for i, data in enumerate(dataloader):      
            data = {k: v.to(device) for k, v in data.items()}
            mol_latents, spec_latents, smile_preds, logit_scale, ids = model(data)
            total_loss, clip_loss, reconstruction_loss = loss_fn(mol_latents, spec_latents, logit_scale, smile_preds, data)
        
            print( 'Validation Epoch: {} | iteration: {}/{} | Loss: {}'.format(epoch, i, len(dataloader), total_loss.detach().item() ), end='\r')
            running_loss.append([total_loss.detach().item(), clip_loss.detach().item(), reconstruction_loss.detach().item()])
            del total_loss, clip_loss, reconstruction_loss, mol_latents, spec_latents, smile_preds, logit_scale
        
    running_loss = np.array(running_loss)
    plt.clf()
    df = pd.DataFrame(running_loss, columns=['ttl_loss', 'clip_loss', 'recon_loss'])
    plot = sns.histplot(df, kde=True, bins=50)
    wandb.log({"Validation Loss Distribution":wandb.Image(plot)}, step=epoch)
    plt.clf()
    del plot
    model.train()
    return np.mean(running_loss, axis = 0)


def save_model(model, config, logs, name):
    
    path_dir = config['train']['checkpoint_dir']          
    if not os.path.exists(path_dir):
        os.mkdir(path_dir)
    model_path = path_dir + '/' + name + '.pth'
    config_path = path_dir + '/config.yaml'
    logs_path = path_dir + '/logs.pickle'
    
    torch.save(model.state_dict(), model_path)
    
    with open(config_path,'w') as yaml_file:
        yaml.dump(dict(config), yaml_file)
    with open(logs_path, 'wb') as file:
        pickle.dump(logs, file)
        
    print("Saved to {}".format(path_dir))
    
def load_model(path_to_dir, type="best_total"):
    files = os.listdir(path_to_dir)
    for file in files:
        if type +'.pth' in file:      
            model_path = path_to_dir + '/' + file
        if '.yaml' in file:
            config_path = path_to_dir + '/' + file
    with open(config_path,'r') as f:
        config = yaml.full_load(f)
        
    model = CLIP(config)
    model.to(device)
    model = torch.nn.parallel.DataParallel(model)
    model.load_state_dict(torch.load(model_path))
    return model

def update_logs_and_checkpoints(config, model, tl, vl, epoch, logs):
    logs['train_total_loss'].append(tl[0])
    logs['train_clip_loss'].append(tl[1])
    logs['train_recon_loss'].append(tl[2])
    
    logs['val_total_loss'].append(vl[0])
    logs['val_clip_loss'].append(vl[1])
    logs['val_recon_loss'].append(vl[2])
    
    if vl[0] < logs['best_total_loss']:
        logs['best_total_loss'] = vl[0]
        logs['best_epoch'] = epoch
        save_model(model, config, logs, 'best_total')

    if vl[1] < logs['best_clip_loss']:
        logs['best_clip_loss'] = vl[1]
        logs['best_clip_epoch'] = epoch 
        save_model(model, config, logs, 'best_clip')
           
    if vl[2] < logs['best_recon_loss']:
        logs['best_recon_loss'] = vl[2]
        logs['best_recon_epoch'] = epoch
        save_model(model, config, logs, 'best_recon')
              
    save_model(model, config, logs, 'best_latest')
    # for key in logs:
    #     if not isinstance(logs[key], list):
    #         wandb.log({key:logs[key]}, step=epoch)  
    return logs

def print_status(logs, time=None):
    train_total_loss = logs['train_total_loss'][-1]
    val_total_loss = logs['train_total_loss'][-1]
    print("Latest Train_Loss: {}, Latest Val_Loss: {}".format( train_total_loss, val_total_loss))
    print("Best Test_Loss: {}, Best Epoch: {}".format( logs['best_total_loss'],logs['best_epoch']))
    print("=============== Time: {}========================".format(time))
  
def train_clip(config, model, dataloaders, optimizer, loss_fn, logs, start=0, num_epochs=50 ):
    for epoch in range(start,num_epochs):
        start = time.time()
        tl = train_one_epoch(config, model, dataloaders['train'], epoch, optimizer, loss_fn , focus="clip_loss")
        vl = validate(config, model, dataloaders['val'], epoch, optimizer, loss_fn )
        logs = update_logs_and_checkpoints(config, model, tl, vl, epoch, logs)
        end = time.time()
        
        wandb.log(
            {
                'epoch': epoch,
                'train_total_loss':tl[0],
                'train_clip_loss':tl[1],
                'train_recon_loss':tl[2],    
                'val_total_loss':vl[0],
                'val_clip_loss':vl[1],
                'val_recon_loss':vl[2],
            },
            step = epoch
        )
        if epoch % 50 == 0:
            clip_performance(config, model, dataloaders, epoch)
        elif epoch > 450 and epoch % 10 == 0:
            clip_performance(config, model, dataloaders, epoch)
            
        print_status(logs, end-start)
    return logs

def train_recon(config, model, dataloaders, optimizer, loss_fn, logs, start=0, num_epochs=50 ):
    for epoch in range(start, num_epochs):
        start = time.time()
        tl = train_one_epoch(config, model, dataloaders['train'], epoch, optimizer, loss_fn , focus="reconstruction_loss")
        vl = validate(config, model, dataloaders['val'], epoch, optimizer, loss_fn )
        logs = update_logs_and_checkpoints(config, model, tl, vl, epoch, logs)
        end = time.time()
        
        wandb.log(
            {
                'epoch': epoch,
                'train_total_loss':tl[0],
                'train_clip_loss':tl[1],
                'train_recon_loss':tl[2],    
                'val_total_loss':vl[0],
                'val_clip_loss':vl[1],
                'val_recon_loss':vl[2],
            },
            step = epoch
        )
        if epoch % 50 == 0:
            clip_performance(config, model, dataloaders, epoch)
        elif epoch > 450 and epoch % 10 == 0:
            clip_performance(config, model, dataloaders, epoch)
        print_status(logs, end-start)
    return logs

def train_total(config, model, dataloaders, optimizer, loss_fn, logs, start=0, num_epochs=50 ):
    for epoch in range(start, num_epochs):
        start = time.time()
        tl = train_one_epoch(config, model, dataloaders['train'], epoch, optimizer, loss_fn , focus="total_loss")
        vl = validate(config, model, dataloaders['val'], epoch, optimizer, loss_fn )
        logs = update_logs_and_checkpoints(config, model, tl, vl, epoch, logs)
        end = time.time()
        
        wandb.log(
            {
                'epoch': epoch,
                'train_total_loss':tl[0],
                'train_clip_loss':tl[1],
                'train_recon_loss':tl[2],    
                'val_total_loss':vl[0],
                'val_clip_loss':vl[1],
                'val_recon_loss':vl[2],
            },
            step = epoch
        )
        if epoch % 50 == 0:
            clip_performance(config, model, dataloaders, epoch)
        elif epoch > 450 and epoch % 10 == 0:
            clip_performance(config, model, dataloaders, epoch)
        print_status(logs, end-start)
    return logs

def top_scores(mat1, mat2):
    """
    mat1 is the first mat
    mat2 is the second mat
    d = []
    for i in mat1:
        for j in mat2:
            closest between i and j
            d.append(rank)
    """
    hits = []
    tops = [1,2,3,4,5, 7, 10]
    score = [0] * (len(tops))

    for i in range(mat1.shape[0]):
        sims = mat2 @ mat1[i].t()
        for k in range(len(tops)):
            max_sims, ids = torch.topk(sims, tops[k])
            if (i) in ids:
                score[k] += 1

    for i in range(len(tops)):
        score[i] = score[i] / mat1.shape[0]

    return np.array(tops), np.array(score ) * 100

# %matplotlib inline
def distance_distribution(molmat, specmat):
    sims = molmat @ specmat.t()
    diagonals = torch.diagonal(sims, 0).cpu().numpy()
    sims = np.random.choice(sims.view(-1).cpu().numpy(), len(diagonals))
    vals = np.concatenate((sims, diagonals), axis=0)
    pairs = ["pairs"] * len(diagonals)
    nonpairs = ["others"] * len(sims)
    df = pd.DataFrame()
    df['distance'] = vals
    df['labels'] = nonpairs + pairs
    plt.clf()
    plot = sns.histplot(df, x='distance', hue='labels', kde=True, bins=50)
    del sims, diagonals,  vals, df
    return plot

def distance_mat(molmat, specmat):
    sims = specmat[:1000].cpu() @ molmat.t()[:,:1000].cpu()
    plt.clf()
    img = sns.heatmap(data=sims, annot=None)
    del sims
    return img

PAD = 0
UNK = 1
EOS = 2
SOS = 3
MASK = 4

class Sampler():
    def __init__(self, model, vocab):
        self.model = model
        self.vocab = vocab
        self.max_len = 40
    def sample(self, embed, greedy_decode=False):
        embed = embed.unsqueeze(0).to(device)
        self.model.eval()
        sample_tensor = torch.zeros((1,self.max_len), dtype=torch.int64).to(device)
        sample_tensor[0,0] = SOS
        with torch.no_grad():
            for i in range(0,self.max_len-1):
                tensor = sample_tensor[:,:i+1]
                logits = self.model.forward(embed, tensor)[:,-1,:]
                probabilities = F.softmax(logits, dim=1)
                sampled_char = torch.multinomial(probabilities,1).item()
                if greedy_decode:
                    sampled_char = torch.argmax(probabilities)
                    
                sample_tensor[0,i+1] = sampled_char
                if sampled_char == EOS:
                    break
            smiles = ""
            chars = self.vocab.from_seq(sample_tensor[0])
            for char in chars:
                if char != "<pad>" and char != "<eos>" and char != "<sos>" and char != "<unk>":
                    smiles += char
            
        return smiles
  
    def sample_multi(self, n, embed, greedy_decode=False):
        smiles_list = []
        for i in range(n):
            smiles_list.append(self.sample(embed, greedy_decode))
        return smiles_list
    
from rdkit import Chem
from rdkit.Chem import RDConfig
from rdkit.Chem import AllChem
from rdkit.Chem import Draw
from rdkit.Chem.Draw import rdDepictor, rdMolDraw2D
opts = Draw.DrawingOptions()
Draw.SetComicMode(opts)

def calculate_decoder_accuracy( model, dataloaders, k=1):
    with torch.no_grad():
        pred_smiles_list = []
        og_smiles_list = []
        count = 0
        sampler = Sampler(model.module.smiles_decoder, model.module.vocab)
        
        for i, data in tqdm(enumerate(dataloaders['val'])):
            data = {k: v.to(device) for k, v in data.items()}
            spec_latents = model.module.forward_spec(data)
            for spec, og in zip(spec_latents, data['smiles'] ):
                ls = sampler.sample_multi(k,spec,greedy_decode=True)
                generated_smiles = []
                for smi in ls:
                    try:
                        generated_smiles.append(Chem.CanonSmiles(smi))
                    except:
                        pass
                og_smile = ""
                chars = model.module.vocab.from_seq(og)
                for char in chars:
                    if char != "<pad>" and char != "<eos>" and char != "<sos>" and char != "<unk>":
                        og_smile += char
                try:
                    og_smile = Chem.CanonSmiles(og_smile)
                except:
                    og_smile=None
                    
                if og_smile is not None and og_smile in generated_smiles:
                    count += 1
                
                og_smiles_list.append(og_smile)
                pred_smiles_list.append(generated_smiles)
            print("No of Hits : ",count / len(og_smiles_list))

        return count / len(og_smiles_list)
    
def decoder_performance(config, model, dataloaders, epoch):
    with torch.no_grad():
        for i, data in enumerate(dataloaders["val"]):
            data = {k: v.to(device) for k, v in data.items()}
            break
        spec_latent = model.module.forward_spec(data)
        one_hot = data['smiles'].to(device)
        
        arr = np.array([])
        vocab = pickle.load(open(config['data']['vocab_path'], 'rb'))
        for i in range(4) :
            index = torch.randint(0,spec_latent.shape[0], (1,))[0].item()
            sampler = Sampler(model.module.smiles_decoder, model.module.vocab)
            smiles_list = sampler.sample_multi(10, spec_latent[index])
            parsed_mols = np.array([Chem.MolFromSmiles(s) for s in smiles_list])
            print("Percentage invalid", (parsed_mols == None).sum()/len(parsed_mols))

            og_smiles = ""
            chars = model.module.vocab.from_seq(one_hot[index])
            for char in chars:
                if char != "<pad>" and char != "<eos>" and char != "<sos>" and char != "<unk>":
                    og_smiles += char

            og_mol = Chem.MolFromSmiles(og_smiles)
            all_mols = np.array([og_mol] + list(parsed_mols))
            arr = np.concatenate((arr, all_mols), axis=0)
        img = Draw.MolsToGridImage(arr, molsPerRow=11, returnPNG=False)
        wandb.log({"Spectra Conditioned smiles decoding": wandb.Image(img)}, step=epoch)
        
        index = torch.randint(0,spec_latent.shape[0], (1,))[0].item()
        sampler = Sampler(model.module.smiles_decoder, vocab)
        smiles_list = sampler.sample_multi(100, spec_latent[index])
        parsed_mols = np.array([Chem.MolFromSmiles(s) for s in smiles_list])
        print("Percentage invalid", (parsed_mols == None).sum()/len(parsed_mols))
        print("==============================")
        wandb.log({"Invalid Molecules percentage": (parsed_mols == None).sum()/len(parsed_mols)}, step=epoch)
        return img    

def clip_performance(config, model, dataloaders, epoch):
    
    # model.to(device)
    model.eval()
    with torch.no_grad():
    
        plt.clf()
        img = decoder_performance(config, model, dataloaders, epoch)
        plt.clf()
        
        max_charge = config['data']['max_charge']
        num_species = config['data']['num_species']

        molembeds = []
        specembeds = []
        val_ids = []

        for i, data in tqdm(enumerate(dataloaders['val'])):    
            data = {k: v.to(device) for k, v in data.items()}
            mol_latents, spec_latents, smile_preds, logit_scale, ids = model(data)
            molembeds.append(mol_latents.detach().cpu())
            specembeds.append(spec_latents.detach().cpu())
            val_ids.append(ids.detach().cpu())
        del mol_latents, spec_latents, smile_preds, logit_scale, ids

        test_molembeds = torch.cat(molembeds, 0)
        test_specembeds = torch.cat(specembeds, 0)
        
        val_ids = torch.cat(val_ids, 0)
        pickle.dump(val_ids, open(config['train']['checkpoint_dir'] + '/val_ids.pickle', 'wb'))
        
        molembeds = []
        specembeds = []
        
        for i, data in tqdm(enumerate(dataloaders['train'])):    
            data = {k: v.to(device) for k, v in data.items()}
            mol_latents, spec_latents, smile_preds, logit_scale, ids = model(data)
            molembeds.append(mol_latents.detach().cpu())
                # specembeds.append(spec_latents.detach().cpu())
        del mol_latents, spec_latents, smile_preds, logit_scale, ids
        
        train_molembeds = torch.cat(molembeds, 0)
        # train_specembeds = torch.cat(specembeds, 0)
        
        all_molembeds = torch.cat(( test_molembeds, train_molembeds), axis = 0)
        del train_molembeds
        
        tops, scores = top_scores(test_specembeds, all_molembeds)
        del all_molembeds
        
        for k, acc in zip(tops, scores):
            # print("Full Top {} Spec".format(k), acc)
            wandb.log({"Full Top {} Spec".format(k): acc}, step=epoch)
            

        tops, scores = top_scores(test_specembeds, test_molembeds )
        for k, acc in zip(tops, scores):
            # print("Test Top {} Spec".format(k), acc)
            wandb.log({"Test Top {} Spec".format(k): acc}, step=epoch)

        # wandb.log({'Distance Distribution Train': distance_distribution(train_molembeds, train_specembeds)}, step=epoch) 
        # del train_molembeds, train_specembeds
        
        # print("===================================================================================","HERE", decoder_acc, decoder_validity,)
        wandb.log({
                'Distance Distribution Test': wandb.Image(distance_distribution(test_molembeds, test_specembeds)),
                'Similarity Matrix Test':wandb.Image(distance_mat(test_specembeds, test_molembeds)),
                'Decoding Accuracy': calculate_decoder_accuracy(model, dataloaders,k=1)
                }, step=epoch)
        

        del test_molembeds, test_specembeds

        model.train()
        plt.clf()
