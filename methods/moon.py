'''
Code credit to https://github.com/QinbinLi/MOON
for implementation of thier method, MOON.
'''

import torch
import logging
from methods.base import Base_Client, Base_Server
from torch.multiprocessing import current_process
import numpy as np
from sklearn.metrics import roc_auc_score
import os
import torch.nn as nn
import math
from datetime import datetime

global result_dir 
now = datetime.now()
result_dir = os.getcwd() + "/Results/{}_{}H".format(now.date(), str(now.hour))
model_dir = result_dir + "/models"


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
                if freq[c][i] == 0:
                    freq[c][i] = 1
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
                # loss = (1 / self.neg_weights[i]) * loss * 0.05
        else : 
            for i in range(len(y_true)):
                loss_pos =  -1 * (torch.log(y_pred[i][y_true[i]] + epsilon))
                loss += self.mu * self.pos_weights[client_idx][y_true[i]] * loss_pos
                # self.pos_weights[client_idx][y_true[i]] * 
            loss /= len(y_true)
        return loss

class Client(Base_Client):
    def __init__(self, client_dict, args):
        super().__init__(client_dict, args)
        self.model = self.model_type(self.num_classes, KD=True, projection=True)
        self.prev_model = self.model_type(self.num_classes, KD=True, projection=True)
        # self.prev_model.load_state_dict(self.model.state_dict())
        self.global_model = self.model_type(self.num_classes, KD=True, projection=True)
        self.harmony = client_dict['harmony']
        self.client_pos_freq = client_dict['clients_pos']
        self.client_neg_freq = client_dict['clients_neg']
        if 'NIH' in self.dir or 'ChexPert' in self.dir:
            if self.harmony == 'n':
                self.criterion1 = torch.nn.BCEWithLogitsLoss().to(self.device)
            else:
                self.criterion1 = PNB_loss(self.args.dataset, self.client_pos_freq, self.client_neg_freq)
            self.criterion2 = torch.nn.CrossEntropyLoss().to(self.device)
        else:
            if self.harmony == 'n':
                self.criterion1 = torch.nn.CrossEntropyLoss().to(self.device)
            else:
                self.criterion1 = PNB_loss(self.args.dataset, self.client_pos_freq, self.client_neg_freq)
            self.criterion2 = torch.nn.CrossEntropyLoss().to(self.device)
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.args.lr, momentum=0.9, weight_decay=self.args.wd, nesterov=True)
        self.cos = torch.nn.CosineSimilarity(dim=-1)
        self.temp = 0.5

    def run(self, received_info):
        client_results = []
        self.global_model.load_state_dict(received_info['global'])
        for client_idx in self.client_map[self.round]:
            self.prev_model.load_state_dict(received_info['prev'][client_idx])
            self.load_client_state_dict(received_info['global'])
            self.train_dataloader = self.train_data[client_idx]
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
        return client_results

    def train(self, client_idx):
        # train the local model
        self.model.to(self.device)
        self.global_model.to(self.device)
        self.prev_model.to(self.device)
        self.model.train()
        epoch_loss = []
        for epoch in range(self.args.epochs):
            batch_loss = []
            for batch_idx, (x, target) in enumerate(self.train_dataloader):
                # logging.info(x.shape)
                x= x.to(self.device)
                self.optimizer.zero_grad()
                #####
                pro1, out = self.model(x)
                pro2, _ = self.global_model(x)

                posi = self.cos(pro1, pro2)
                logits = posi.reshape(-1,1)

                pro3, _ = self.prev_model(x)
                nega = self.cos(pro1, pro3)
                logits = torch.cat((logits, nega.reshape(-1,1)), dim=1)

                logits /= self.temp
                labels = torch.zeros(x.size(0)).to(self.device).long()

                if 'NIH' in self.dir or 'CheXpert' in self.dir:
                    if self.harmony == 'n':
                        loss1 = self.criterion1(out, target.type(torch.FloatTensor).to(self.device))
                    else:
                        loss1 = self.criterion1(client_idx, out, target.type(torch.FloatTensor).to(self.device))
                else:
                    if self.harmony == 'n':
                        loss1 = self.criterion1(out, target.type(torch.LongTensor).to(self.device))
                    else:
                        loss1 = self.criterion1(client_idx, torch.softmax(out, dim=1), target.type(torch.LongTensor).to(self.device))


                loss2 = self.args.mu * self.criterion2(logits, labels)

                # loss1 = self.criterion(out, target)
                
                loss = loss1 + loss2
                #####
                loss.backward()
                self.optimizer.step()
                batch_loss.append(loss.item())
            if len(batch_loss) > 0:
                epoch_loss.append(sum(batch_loss) / len(batch_loss))
                logging.info('(client {}. Local Training Epoch: {} \tLoss: {:.6f}  Thread {}  Map {}'.format(self.client_index,
                    epoch, sum(epoch_loss) / len(epoch_loss), current_process()._identity[0], self.client_map[self.round]))
        weights = self.model.cpu().state_dict()
        self.prev_model.load_state_dict(weights) ##
        return weights

    def test(self, client_idx):
        self.model.to(self.device)
        self.model.eval()
        sigmoid = torch.nn.Sigmoid()
        test_correct = 0.0
        test_sample_number = 0.0

        val_loader_examples_num = len(self.test_dataloader.dataset)
        probs = np.zeros((val_loader_examples_num, self.num_classes), dtype = np.float32)
        gt    = np.zeros((val_loader_examples_num, self.num_classes), dtype = np.float32)
        k=0

        with torch.no_grad():
            for batch_idx, (x, target) in enumerate(self.test_dataloader):
                x = x.to(self.device)
                target = target.to(self.device)

                _, out = self.model(x)
                if 'NIH' in self.dir or 'CheXpert' in self.dir:
                    probs[k: k + out.shape[0], :] = out.cpu()
                    gt[   k: k + out.shape[0], :] = target.cpu()
                    k += out.shape[0] 
                    preds = np.round(sigmoid(out).cpu().detach().numpy())
                    targets = target.cpu().detach().numpy()
                    test_sample_number += len(targets)*self.num_classes
                    test_correct += (preds == targets).sum()
                else:
                    _, predicted = torch.max(out, 1)
                    correct = predicted.eq(target).sum()
                    test_correct += correct.item()
                    test_sample_number += target.size(0)

            acc = (test_correct / test_sample_number)*100
            if self.args.dataset == 'NIH' or self.args.dataset == 'CheXpert':
                try:
                    auc = roc_auc_score(gt, probs)
                except:
                    auc = 0
                logging.info("************* Client {} AUC = {:.2f},  Acc = {:.2f}**************".format(self.client_index, auc, acc))
                f = open(result_dir + "/performance{}.txt".format(client_idx), "a")
                f.write(str(auc) + "\n")
                f.close()
                return auc
            else:
                logging.info("************* Client {} Acc = {:.2f} ***************************".format(self.client_index, acc))
                f = open(result_dir + "/performance{}.txt".format(client_idx), "a")
                f.write(str(acc) + "\n")
                f.close()
                return acc

class Server(Base_Server):
    def __init__(self,server_dict, args):
        super().__init__(server_dict, args)
        self.model = self.model_type(self.num_classes, KD=True, projection=True)
        self.prev_models = {x:self.model.cpu().state_dict() for x in range(self.args.client_number)}
        self.harmony = server_dict['harmony']
        self.imbalance_weights = server_dict['imbalances']

        global result_dir 
        self.result_dir = result_dir
        os.mkdir(self.result_dir)
        self.model_dir = model_dir
        os.mkdir(self.model_dir)
        c = open(self.result_dir + "/config.txt", "w")
        c.write("method: {}, dataset: {}, partition_alpha (delta): {}, longtail: {}, ibf: {},  beta: {}, gamma: {}, mu: {}, tau: {}, comm_round: {}, local_epoch: {}, num_of_client: {}, random seed: {}".format(self.args.method, self.args.dataset, str(self.args.partition_alpha),str(self.args.longtail), str(self.args.ibf), str(self.args.beta), str(self.args.gamma) ,str(self.args.mu), str(self.args.tau), str(self.args.comm_round), str(self.args.epochs), str(self.args.client_number), str(self.args.seed)))
        open(self.result_dir + "/overall_performance.txt", "w")
        for i in range(args.client_number):
            open(self.result_dir + "/performance{}.txt".format(i), "w")
    
    def operations(self, client_info):
        client_info.sort(key=lambda tup: tup['client_index']) 
        client_sd = [c['weights'] for c in client_info] # clients' number of weights
        ################################################################################################
        if self.harmony == 'y':
            gamma = 1
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


    def run(self, received_info):
        server_outputs = self.operations(received_info)
        acc = self.test()
        self.log_info(received_info, acc)
        self.round += 1
        if acc > self.acc:
            torch.save(self.model.state_dict(), '{}/{}.pt'.format(self.save_path, 'server'))
            self.acc = acc
        for x in received_info:
            self.prev_models[x['client_index']] = x['weights']
        server_outputs = [{'global':g, 'prev':self.prev_models} for g in server_outputs]
        acc_path = '{}/logs/{}_{}_harmony_acc.txt'.format(os.getcwd(), self.args.dataset,self.args.method)
        f = open(acc_path, 'a')
        f.write(str(acc) + '\n')
        f.close()
        return server_outputs

    def start(self):
        return [{'global':self.model.cpu().state_dict(), 'prev':self.prev_models} for x in range(self.args.thread_number)]
    
    def test(self):
        self.model.to(self.device)
        self.model.eval()

        test_correct = 0.0
        test_loss = 0.0
        test_sample_number = 0.0
        sigmoid = torch.nn.Sigmoid()
        val_loader_examples_num = len(self.test_data.dataset)
        probs = np.zeros((val_loader_examples_num, self.num_classes), dtype = np.float32)
        gt    = np.zeros((val_loader_examples_num, self.num_classes), dtype = np.float32)
        k=0

        with torch.no_grad():
            for batch_idx, (x, target) in enumerate(self.test_data):
                x = x.to(self.device)
                target = target.to(self.device)
                _, out = self.model(x)

                if 'NIH' in self.dir or 'CheXpert' in self.dir:
                    probs[k: k + out.shape[0], :] = out.cpu()
                    gt[   k: k + out.shape[0], :] = target.cpu()
                    k += out.shape[0] 
                    preds = np.round(sigmoid(out).cpu().detach().numpy())
                    targets = target.cpu().detach().numpy()
                    test_sample_number += len(targets)*self.num_classes
                    test_correct += (preds == targets).sum()
                else:
                    _, predicted = torch.max(out, 1)
                    correct = predicted.eq(target).sum()
                    test_correct += correct.item()
                    # test_loss += loss.item() * target.size(0)
                    test_sample_number += target.size(0)

            acc = (test_correct / test_sample_number)*100
            if self.args.dataset == 'NIH' or self.args.dataset == 'CheXpert':
                auc = roc_auc_score(gt, probs)
                logging.info("***** Server AUC = {:.4f} ,Acc = {:.4f} *********************************************************************".format(auc, acc))
                f = open(result_dir + "/overall_performance.txt", "a")
                f.write(str(auc) + "\n")
                f.close()
                return auc
            else:
                logging.info("***** Server Acc = {:.4f} *********************************************************************".format(acc))
                f = open(result_dir + "/overall_performance.txt", "a")
                f.write(str(acc) + "\n")
                f.close()
                return acc