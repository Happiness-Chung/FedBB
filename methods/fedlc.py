import torch
from methods.base import Base_Client, Base_Server
import math
import numpy as np
import torch.nn as nn
import logging
from torch.multiprocessing import current_process
import matplotlib.pyplot as plt
import os
from datetime import datetime

class FedLC_Loss():

    def __init__(self, tau , pos_freq, device):

        self.tau = tau
        self.device = device
        self.label_distrib = torch.Tensor(pos_freq).to(self.device)
        for i in range(len(self.label_distrib)):
            for j in range(len(self.label_distrib[i])):
                self.label_distrib[i][j] = max(1e-8, self.label_distrib[i][j])
        print(self.label_distrib)
        self.epsilon = 1e-10

    def __call__(self, client_idx, logit, y):

        logit = logit.to(self.device)
        loss = 0
    
        cal_logit = torch.exp(
            logit
            - (
                self.tau
                * torch.pow(self.label_distrib[client_idx], -1 / 4)
                .unsqueeze(0)
                .expand((logit.shape[0], -1))
            )
        )
        
        y_logit = torch.gather(cal_logit, dim=-1, index=y.unsqueeze(1))
        loss = -torch.log(y_logit / cal_logit.sum(dim=-1, keepdim=True) + self.epsilon)
        
        return loss.sum() / logit.shape[0]

class Client(Base_Client):

    def __init__(self, client_dict, args):
        
        super().__init__(client_dict, args)

        self.model = self.model_type(self.num_classes).to(self.device)
        self.client_pos_freq = client_dict['clients_pos']
        self.client_neg_freq = client_dict['clients_neg']
        
        self.criterion = FedLC_Loss(self.args.tau, self.client_pos_freq, self.device)
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.args.lr, momentum=0.9, weight_decay=self.args.wd, nesterov=True)

    def run(self, received_info): 
        # recieved info : a server model weights(OrderedDict)
        # one globally merged model's parameter
        client_results = []
        for client_idx in self.client_map[self.round]: # round is the index of communication round
            self.load_client_state_dict(received_info) 
            self.train_dataloader = self.train_data[client_idx] # among dataloader, pick one
            self.test_dataloader = self.test_data[client_idx]
            if self.args.client_sample < 1.0 and self.train_dataloader._iterator is not None and self.train_dataloader._iterator._shutdown:
                self.train_dataloader._iterator = self.train_dataloader._get_iterator()
            self.client_index = client_idx
            num_samples = len(self.train_dataloader)*self.args.batch_size
            weights = self.train(client_idx)
            acc = self.test(client_idx)
            client_results.append({'weights':weights, 'num_samples':num_samples,'acc':acc, 'client_index':self.client_index})
            if self.args.client_sample < 1.0 and self.train_dataloader._iterator is not None:
                self.train_dataloader._iterator._shutdown_workers()

        self.round += 1
        return client_results # clients' number of weights 

    def train(self, client_idx):
        # train the local model
        self.model.to(self.device)
        self.model.train()
        epoch_loss = []
        for epoch in range(self.args.epochs):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.train_dataloader):
                images, labels = images.to(self.device), labels.to(self.device)
                self.optimizer.zero_grad()

                if 'NIH' in self.dir or 'CheXpert' in self.dir:
                    out = self.model(images)  
                    loss = self.criterion(client_idx, out, labels.type(torch.FloatTensor).to(self.device))
                else:
                    log_probs = self.model(images)
                    loss = self.criterion(client_idx, log_probs.to(self.device), labels.type(torch.LongTensor).to(self.device)) ####
                
                loss.backward()
                self.optimizer.step()
                batch_loss.append(loss.item())
            if len(batch_loss) > 0:
                epoch_loss.append(sum(batch_loss) / len(batch_loss))
                logging.info('(client {}. Local Training Epoch: {} \tLoss: {:.6f}  Thread {}  Map {}'.format(self.client_index,
                                                                            epoch, sum(epoch_loss) / len(epoch_loss), current_process()._identity[0], self.client_map[self.round]))
        weights = self.model.cpu().state_dict()
        return weights
        
class Server(Base_Server):
    def __init__(self,server_dict, args):
        super().__init__(server_dict, args)
        self.model = self.model_type(self.num_classes)
        self.imbalance_weights = server_dict['imbalances']
        self.gamma = args.gamma

    def operations(self, client_info):
        client_info.sort(key=lambda tup: tup['client_index'])
        client_sd = [c['weights'] for c in client_info] # clients' number of weights
        ################################################################################################
        cw = [c['num_samples']/sum([x['num_samples'] for x in client_info]) for c in client_info]
        ssd = self.model.state_dict()
        
        for key in ssd:
            ssd[key] = sum([sd[key]*cw[i] for i, sd in enumerate(client_sd)])
        self.model.load_state_dict(ssd)
        if self.args.save_client:
            for client in client_info:
                torch.save(client['weights'], '{}/client_{}.pt'.format(self.save_path, client['client_index']))
        return [self.model.cpu().state_dict() for x in range(self.args.thread_number)] 
