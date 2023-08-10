'''
Code credit to https://github.com/QinbinLi/MOON
for thier implementation of FedProx.
'''

import torch
import logging
from methods.base import Base_Client, Base_Server
import copy
from torch.multiprocessing import current_process
import numpy as np
import torch.nn as nn
import math

class PNB_loss():

    def __init__(self, dataset, pos_freq, neg_freq):
        self.beta = 0.9999
        self.alpha = 1
        self.mu = 5.0
        self.dataset = dataset
        self.pos_freq = np.array(pos_freq)
        self.neg_freq = np.array(neg_freq)
        self.pos_weights = self.get_inverse_effective_number(self.beta, self.pos_freq)
        self.neg_weights = self.get_inverse_effective_number(self.beta, self.neg_freq)       
        
        #temp
        self.total = self.pos_weights + self.neg_weights
        self.pos_weights = (self.pos_weights / self.total)
        self.pos_weights = np.nan_to_num(self.pos_weights)
        self.neg_weights = self.neg_weights / self.total

    def get_inverse_effective_number(self, beta, freq): # beta is same for all classes
        sons = np.array(freq) / self.alpha # scaling factor
        for c in range(len(freq)):
            for i in range(len(freq[0])):
                if sons[c][i] == 0:
                    sons[c][i] = 1
                sons[c][i] = math.pow(beta,sons[c][i])
        sons = np.array(sons)
        En =  (1 - beta) / (1 - sons)
        En[np.isnan(En)] = En.max()
        return En # the form of vector

    def __call__(self, client_idx, y_pred, y_true, epsilon=1e-7):
        """
        Return weighted loss value. 

        Args:
            y_true (Tensor): Tensor of true labels, size is (num_examples, num_classes)
            y_pred (Tensor): Tensor of predicted labels, size is (num_examples, num_classes)
            pos_weights : (client_num, batch_size, num_classes)
            neg_weights : (client_num, batch_size, num_classes)
        Returns:
            loss (Float): overall scalar loss summed across all classes
        """
        # initialize loss to zero
        loss = 0.0
        sigmoid = nn.Sigmoid()
        
        if self.dataset == 'NIH' or self.dataset == 'CheXpert':
            for i in range(len(self.pos_weights[0])): # This length should be the class
                # for each class, add average weighted loss for that class 
                loss_pos =  -1 * torch.mean(self.pos_weights[client_idx][i] * y_true[:, i] * torch.log(sigmoid(y_pred[:, i]) + epsilon))
                loss_neg =  -1 * torch.mean(self.neg_weights[client_idx][i] * (1 - y_true[:, i]) * torch.log(1 -sigmoid( y_pred[:, i]) + epsilon))
                loss += self.mu * self.pos_weights[client_idx][i] * (loss_pos + loss_neg)
        else : 
            for i in range(len(y_true)):
                loss_pos =  -1 * (torch.log(y_pred[i][y_true[i]] + epsilon))
                loss += self.mu * self.pos_weights[client_idx][y_true[i]] * loss_pos
            loss /= len(y_true)
        return loss

class Client(Base_Client):
    def __init__(self, client_dict, args):
        super().__init__(client_dict, args)
        self.model = self.model_type(self.num_classes).to(self.device)
        self.harmony = client_dict['harmony']
        self.client_pos_freq = client_dict['clients_pos']
        self.client_neg_freq = client_dict['clients_neg']
        if 'NIH' in self.dir or 'CheXpert' in self.dir:
            if self.harmony == 'n':
                self.criterion = torch.nn.BCEWithLogitsLoss().to(self.device)
            else:
                self.criterion = PNB_loss(self.args.dataset, self.client_pos_freq, self.client_neg_freq)
        else:
            if self.harmony == 'n':
                self.criterion = torch.nn.CrossEntropyLoss().to(self.device)
            else:
                self.criterion = PNB_loss(self.args.dataset, self.client_pos_freq, self.client_neg_freq)
            
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
            acc = self.test()
            client_results.append({'weights':weights, 'num_samples':num_samples,'acc':acc, 'client_index':self.client_index})
            if self.args.client_sample < 1.0 and self.train_dataloader._iterator is not None:
                self.train_dataloader._iterator._shutdown_workers()

        self.round += 1
        return client_results # clients' number of weights 

    def train(self, client_idx):
        # train the local model
        self.model.to(self.device)
        global_weight_collector = copy.deepcopy(list(self.model.parameters()))
        self.model.train()
        epoch_loss = []
        for epoch in range(self.args.epochs):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.train_dataloader):
                # logging.info(images.shape)
                images, labels = images.to(self.device), labels.to(self.device)
                self.optimizer.zero_grad()
                if 'NIH' in self.dir or 'CheXpert' in self.dir:
                    if self.harmony == 'n':
                        log_probs = self.model(images)
                        loss = self.criterion(log_probs, labels.type(torch.FloatTensor).to(self.device))
                    else:
                        log_probs = self.model(images)
                        loss = self.criterion(client_idx, log_probs, labels.type(torch.FloatTensor).to(self.device))
                else:
                    if self.harmony == 'n':
                        log_probs = self.model(images)
                        loss = self.criterion(log_probs, labels.type(torch.LongTensor).to(self.device))
                    else:
                        log_probs = self.model(images)
                        loss = self.criterion(client_idx, torch.softmax(log_probs, dim=1), labels.type(torch.LongTensor).to(self.device)) ####
                ############
                # for fedprox
                fed_prox_reg = 0.0
                for param_index, param in enumerate(self.model.parameters()):
                    fed_prox_reg += ((self.args.mu / 2) * torch.norm((param - global_weight_collector[param_index])) ** 2)
                loss = loss + fed_prox_reg
                ########
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
        self.harmony = server_dict['harmony']
        self.imbalance_weights = server_dict['imbalances']
        self.gamma = args.gamma
    
    def operations(self, client_info):
        client_info.sort(key=lambda tup: tup['client_index']) 
        client_sd = [c['weights'] for c in client_info] # clients' number of weights
        ################################################################################################
        if self.harmony == 'y':
            gamma = self.gamma
            cw1 = self.imbalance_weights
            cw2 = [c['num_samples']/sum([x['num_samples'] for x in client_info]) for c in client_info]
            cw1 = np.array(cw1)
            cw2 = np.array(cw2)
            cw = gamma * cw1 + (1 - gamma) * cw2
            print("Clients weight: ", cw)
        else:
            cw = [c['num_samples']/sum([x['num_samples'] for x in client_info]) for c in client_info]

        ssd = self.model.state_dict()
        for key in ssd:
            ssd[key] = sum([sd[key]*cw[i] for i, sd in enumerate(client_sd)])
        self.model.load_state_dict(ssd)
        if self.args.save_client:
            for client in client_info:
                torch.save(client['weights'], '{}/client_{}.pt'.format(self.save_path, client['client_index']))
        return [self.model.cpu().state_dict() for x in range(self.args.thread_number)] # thread의 갯수만큼 server의 모델을 복사해서 반환
