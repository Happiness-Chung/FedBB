'''
Code is based on
https://github.com/taoyang1122/GradAug,
https://github.com/taoyang1122/MutualNet.
Also, Lipschitz related functions are from
https://github.com/42Shawn/LONDON/tree/master
'''

import random
import torch
import torch.nn.functional as F

import logging
from methods.base import Base_Client, Base_Server
import torch.nn.functional as F
import models.ComputePostBN as pbn
from torch.multiprocessing import current_process
import numpy as np
import random
from sklearn.metrics import roc_auc_score
from datetime import datetime
import os

global result_dir 
now = datetime.now()
result_dir = os.getcwd() + "/Results/{}_{}H".format(now.date(), str(now.hour))
model_dir = result_dir + "/models"

class Client(Base_Client):
    def __init__(self, client_dict, args):
        super().__init__(client_dict, args)
        self.model = self.model_type(self.num_classes).to(self.device)
        self.criterion = torch.nn.CrossEntropyLoss().to(self.device)
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.args.lr, momentum=0.9, weight_decay=self.args.wd, nesterov=True)
        self.width_range = client_dict['width_range']
        self.resolutions = client_dict['resolutions']
        self.num_sub = args.num_subnets-1

    def train(self):
        # train the local model
        self.model.to(self.device)
        self.model.train()
        epoch_loss = []
        for epoch in range(self.args.epochs):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.train_dataloader):
                labels = labels.type(torch.LongTensor)
                images, labels = images.to(self.device, non_blocking=True), labels.to(self.device, non_blocking=True)
                self.optimizer.zero_grad()
               
                self.model.apply(lambda m: setattr(m, 'width_mult', self.width_range[-1]))
                t_feats, t_out = self.model.extract_feature(images)

                if 'NIH' in self.dir or 'CheXpert' in self.dir:
                    loss = self.criterion(t_out, labels.type(torch.FloatTensor).to(self.device))
                else:
                    loss = self.criterion(t_out, labels.type(torch.LongTensor).to(self.device))

                loss.backward()
                loss_CE = loss.item()
                self.model.apply(lambda m: setattr(m, 'width_mult', self.width_range[0]))
                s_feats = self.model.reuse_feature(t_feats[-2].detach())
                
                # Lipschitz loss
                TM_s = torch.bmm(self.transmitting_matrix(s_feats[-2], s_feats[-1]), self.transmitting_matrix(s_feats[-2], s_feats[-1]).transpose(2,1))
                TM_t = torch.bmm(self.transmitting_matrix(t_feats[-2].detach(), t_feats[-1].detach()), self.transmitting_matrix(t_feats[-2].detach(), t_feats[-1].detach()).transpose(2,1))
                loss = F.mse_loss(self.top_eigenvalue(K=TM_s), self.top_eigenvalue(K=TM_t))
                loss = self.args.mu*(loss_CE/loss.item())*loss
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 10.0)
                self.optimizer.step()
                batch_loss.append(loss.item())
            if len(batch_loss) > 0:
                epoch_loss.append(sum(batch_loss) / len(batch_loss))
                logging.info('(client {}. Local Training Epoch: {} \tLoss: {:.6f}  Thread {}  Map {}'.format(self.client_index,
                                                                            epoch, sum(epoch_loss) / len(epoch_loss), current_process()._identity[0], self.client_map[self.round]))
        weights = self.model.cpu().state_dict()
        return weights

    def transmitting_matrix(self, fm1, fm2):
        if fm1.size(2) > fm2.size(2):
            fm1 = F.adaptive_avg_pool2d(fm1, (fm2.size(2), fm2.size(3)))

        fm1 = fm1.view(fm1.size(0), fm1.size(1), -1)
        fm2 = fm2.view(fm2.size(0), fm2.size(1), -1).transpose(1, 2)

        fsp = torch.bmm(fm1, fm2) / fm1.size(2)
        return fsp

    def top_eigenvalue(self, K, n_power_iterations=10, dim=1):
        v = torch.ones(K.shape[0], K.shape[1], 1).to(self.device)
        for _ in range(n_power_iterations):
            m = torch.bmm(K, v)
            n = torch.norm(m, dim=1).unsqueeze(1)
            v = m / n

        top_eigenvalue = torch.sqrt(n / torch.norm(v, dim=1).unsqueeze(1))
        return top_eigenvalue

    def test(self, client_idx):
        self.model.to(self.device)
        self.model.eval()
        test_correct = 0.0
        test_sample_number = 0.0
        sigmoid = torch.nn.Sigmoid()
        val_loader_examples_num = len(self.test_dataloader.dataset)
        probs = np.zeros((val_loader_examples_num, self.num_classes), dtype = np.float32)
        gt    = np.zeros((val_loader_examples_num, self.num_classes), dtype = np.float32)
        k=0
        with torch.no_grad():
            ###
            self.model.apply(lambda m: setattr(m, 'width_mult', self.width_range[-1]))
            self.model = pbn.ComputeBN(self.model, self.train_dataloader, self.resolutions[0], self.device)
            ###
            for batch_idx, (x, target) in enumerate(self.test_dataloader):
                target = target.type(torch.LongTensor)
                x = x.to(self.device)
                target = target.to(self.device)

                out = self.model(x)
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
                logging.info("************* Client {} Acc = {:.2f} **************".format(self.client_index, acc))
                f = open(result_dir + "/performance{}.txt".format(client_idx), "a")
                f.write(str(acc) + "\n")
                f.close()
                return acc

class Server(Base_Server):
    def __init__(self,server_dict, args):
        super().__init__(server_dict, args)
        self.model = self.model_type(self.num_classes)

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
            ###
            self.model.apply(lambda m: setattr(m, 'width_mult', 1.0))
            ###
            for batch_idx, (x, target) in enumerate(self.test_data):
                target = target.type(torch.LongTensor)
                x = x.to(self.device)
                target = target.to(self.device)

                out = self.model(x)
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