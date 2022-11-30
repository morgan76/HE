# -*- coding: utf-8 -*-
import sys

import numpy as np
import pandas as pd
import tqdm
import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from configuration import config as experiment_config
from torch.autograd import Variable
from data import CondTripletDset
from model.CS_Tripletnet import CS_Tripletnet
from model.csn import ConditionalSimNet
from model.models import EmbedNet
from parser import get_parser
from eval_segmentation import eval_segmentation
import utils
from losses import TripletLoss_margins


def train_epoch(model, loss_func, device, train_loader, optimizer, disable=False):
    model.train()
    train_losses = []

    for batch_idx, (data1, data2, data3, c) in enumerate(tqdm.tqdm(train_loader,
                                                        disable=disable)):    
        data1, data2, data3, c = data1.to(device), data2.to(device), data3.to(device), c.to(device)     
        data1, data2, data3, c = Variable(data1), Variable(data2), Variable(data3), Variable(c)   
        embedded_x, embedded_y, embedded_z = model(data1, data2, data3, c)
        loss = loss_func(embedded_x, embedded_z, embedded_y, c)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_losses.append(loss.item())
            
    return np.mean(train_losses)


def validate_using_triplets(model, loss_func, device, valid_loader, epoch,
                              disable=False):
    model.eval()
    with torch.no_grad():
        loss_valid = []
        for batch_idx, (data1, data2, data3, c) in enumerate(tqdm.tqdm(valid_loader, disable=disable)):
            data1, data2, data3, c = data1.to(device), data2.to(device), data3.to(device), c.to(device)
            data1, data2, data3, c = Variable(data1), Variable(data2), Variable(data3), Variable(c)
            embedded_x, embedded_y, embedded_z = model(data1, data2, data3, c)
            valid_loss =  loss_func(embedded_x, embedded_z, embedded_y, c).data.item()
            loss_valid.append(valid_loss)
    return np.mean(loss_valid)

def validate_using_segmentation(tracklist, embedding_net, device, config,
                                  epoch, disable=False, name='valid'):
    results_window_1 = []
    results_window_3 = []
    recalls_3 = []
    precision_3 = []
    embedding_net.eval()
    with torch.no_grad():
        for idx, track in enumerate(tqdm.tqdm(tracklist, disable=disable)):
            #try:
            results1, results2, results3 = eval_segmentation(track, embedding_net, config, device, experiment_config.feat_id, return_data=False)
            results_window_1.append(results1['F1'])
            results_window_3.append(results3['F3'])
            recalls_3.append(results3['R3'])
            precision_3.append(results3['P3'])
            #except:
            #    pass
    F1 = np.mean(results_window_1)
    F3 = np.mean(results_window_3)
    R3 = np.mean(recalls_3)
    P3 = np.mean(precision_3)
    return F1, F3, R3, P3

def train_model(exp_config):
    # Configuration
    print(exp_config)

    use_cuda = torch.cuda.is_available() and not exp_config.no_cuda
    print('Using GPU:', use_cuda)
    # Create 'models' folder if it does not exist
    exp_config.checkpoints_dir.mkdir(exist_ok=True, parents=True)
    exp_config.models_dir.mkdir(exist_ok=True, parents=True)

    # Seed for reproductible experiments
    torch.manual_seed(exp_config.seed)
    torch.cuda.manual_seed(exp_config.seed)
    np.random.seed(exp_config.seed)

    # Dataset
    train_data = CondTripletDset(exp_config, split ='train')
    valid_data = CondTripletDset(exp_config, split ='valid')

    dataloader_kwargs = {'num_workers': exp_config.nb_workers,
                        'pin_memory': True} if use_cuda else {}

    device = torch.device('cuda' if use_cuda else 'cpu')
    
    # Data loaders
    train_loader = DataLoader(train_data, batch_size=exp_config.batch_size, shuffle=False, **dataloader_kwargs)
    valid_loader = DataLoader(valid_data, batch_size=exp_config.batch_size, shuffle=False, **dataloader_kwargs)


    # Loss function
    loss_func = TripletLoss_margins()

    # Network
    model_path = exp_config.models_dir.joinpath(exp_config.model_name+'.pt')

    if not exp_config.resume:
        print('New model {}.pt will be created'.format(exp_config.model_name))

        
        model = EmbedNet(exp_config).to(device)
        csn_model = ConditionalSimNet(model, n_conditions=exp_config.n_conditions, 
        embedding_size=128, learnedmask=exp_config.learnedmask, prein=exp_config.prein)
        global mask_var
        mask_var = csn_model.masks.weight
        embedding_net = CS_Tripletnet(csn_model)
        embedding_net.to(device)
    
    else:
        model_temp = EmbedNet(exp_config)
        csn_model_temp = ConditionalSimNet(model_temp, n_conditions=exp_config.n_conditions, 
        embedding_size=128, learnedmask=exp_config.learnedmask, prein=exp_config.prein)
        embedding_net = CS_Tripletnet(csn_model_temp)
        model_path = os.path.join(exp_config.models_dir, exp_config.model_name + "." + 'pt')
        embedding_net.load_state_dict(torch.load(model_path)['state_dict']) 
        embedding_net.to(device) 
    
    
    # Optimizer 
    parameters = filter(lambda p: p.requires_grad, embedding_net.parameters())
    optimizer = optim.RMSprop(parameters, lr=exp_config.learning_rate, alpha=0.99, eps=1e-08, weight_decay=0, momentum=0, centered=False)

    
    t = tqdm.trange(1, exp_config.epochs + 1, disable=exp_config.quiet, file=sys.stdout)

    best_F3 = 0
    patience_max = exp_config.patience
    patience = 0
    
    # Training
    for epoch in t:
        t.set_description('Training Epoch')
        # Training
        train_loss = train_epoch(embedding_net, loss_func, device,
                    train_loader, optimizer, disable=exp_config.quiet)

        # Validate using a fixed set of triplets of the valid part
        valid_loss = validate_using_triplets(embedding_net, loss_func, device,
                                valid_loader, epoch, disable=exp_config.quiet)
        
        F1, F3, R3, P3 = validate_using_segmentation(valid_data.tracklist, embedding_net,
                                    device, exp_config, epoch,
                                    disable=exp_config.quiet)
   
        print("Epoch {} Training Loss = {}".format(epoch, train_loss), "Valid Loss = {}".format(valid_loss))

        if (F3 > best_F3) :

            print("\nEpoch {}: best model yet, with a F3 score of {} for a window"
                " of 3 seconds (best was {})\n".format(epoch, F3, best_F3,))

            utils.save_model(exp_config.models_dir, exp_config, epoch,
                        embedding_net, optimizer)
                
            best_F3 = F3
            patience = 0
                
        else:
                
            print("\nEpoch {}: Current F3={}, whereas best is {}\n".format(epoch, F3, best_F3))
            patience += 1
            if patience >= patience_max:
                optimizer.param_groups[0]['lr'] *= 0.5
                print('Changing learning rate: lr =', optimizer.param_groups[0]['lr'])
                patience = 0

def update_config_with_args(args):
    variables = vars(args)
    for var in variables:
        if variables[var] is not None:
            setattr(experiment_config, var, variables[var])


if __name__ == '__main__':
    parser = get_parser()
    args, _ = parser.parse_known_args()
    update_config_with_args(args)