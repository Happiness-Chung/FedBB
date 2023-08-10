'''
Federated Dataset Loading and Partitioning
Code based on https://github.com/FedML-AI/FedML
'''
import glob, os, sys, pdb, time
import pandas as pd
import numpy as np
import cv2
import pickle
from torch.utils.data import Dataset
from tqdm import tqdm
import torch
import torchvision.transforms as transforms
from imageio import imread
from PIL import Image, ImageOps
import torchvision.models as models 
import torch.nn as nn
from matplotlib import pyplot as plt
import csv
import pandas as pd
import PIL.Image as pilimg
import random
import logging

import numpy as np
from numpy.core.fromnumeric import mean
import torch.utils.data as data
import torchvision.transforms as transforms
from data_preprocessing import config
from data_preprocessing.datasets import CIFAR_truncated, ImageFolder_custom

logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.INFO)

def distribute_indices(length, alpha):

    ratios = np.round(np.random.dirichlet(np.repeat(alpha, 5))*length).astype(int)
    indices = list(range(length))
    random.shuffle(indices)

    ### Client num should be greater than 
    if sum(ratios) > length:
        ratios[4] -= (sum(ratios) - length)
    else:
        ratios[4] += (length - sum(ratios))

    indices0 = []
    indices1 = []
    indices2 = []
    indices3 = []
    indices4 = []

    for i in range(0,ratios[0]):
        indices0.append(indices[i])
    for i in range(ratios[0],ratios[0] + ratios[1]):
        indices1.append(indices[i])
    for i in range(ratios[0] + ratios[1],ratios[0] + ratios[1] + ratios[2]):
        indices2.append(indices[i])
    for i in range(ratios[0] + ratios[1] + ratios[2],ratios[0] + ratios[1] + ratios[2] + ratios[3]):
        indices3.append(indices[i])
    for i in range(ratios[0] + ratios[1] + ratios[2] + ratios[3], length):
        indices4.append(indices[i])

    indices = [indices0,indices1,indices2,indices3,indices4]

    return indices

def check_version(cifar_version):
    if cifar_version not in ['10', '100', '20']:
        raise ValueError('cifar version must be one of 10, 20, 100.')

def img_num(cifar_version):
    check_version(cifar_version)
    dt = {'10': 5000, '100': 500, '20': 2500}
    return dt[cifar_version]

def get_img_num_per_cls(cifar_version, imb_factor=0.1):
    """
    Get a list of image numbers for each class, given cifar version
    Num of imgs follows emponential distribution
    img max: 5000 / 500 * e^(-lambda * 0);
    img min: 5000 / 500 * e^(-lambda * int(cifar_version - 1))
    exp(-lambda * (int(cifar_version) - 1)) = img_max / img_min
    args:
      cifar_version: str, '10', '100', '20'
      imb_factor: float, imbalance factor: img_min/img_max,
        None if geting default cifar data number
    output:
      img_num_per_cls: a list of number of images per class
    """
    cls_num = int(cifar_version)
    img_max = img_num(cifar_version)
    if imb_factor is None:
        return [img_max] * cls_num
    img_num_per_cls = []
    for cls_idx in range(cls_num):
        num = img_max * (imb_factor**(cls_idx / (cls_num - 1.0)))
        img_num_per_cls.append(int(num))
    return img_num_per_cls

class NIHTrainDataset(Dataset):
    def __init__(self,c_num, data_dir, transform = None, indices=None):
        
        self.data_dir = data_dir
        self.transform = transform
        self.df = self.get_df()
        self.make_pkl_dir(config.pkl_dir_path)
        self.the_chosen = indices
        
        self.disease_cnt = [0]*14
        self.all_classes = ['Cardiomegaly','Emphysema','Effusion','Hernia','Infiltration','Mass','Nodule','Atelectasis','Pneumothorax','Pleural_Thickening','Pneumonia','Fibrosis','Edema','Consolidation', 'No Finding']
        
        if not os.path.exists(os.path.join(config.pkl_dir_path, config.train_val_df_pkl_path)):

            self.train_val_df = self.get_train_val_df()
            # pickle dump the train_val_df
            with open(os.path.join(config.pkl_dir_path, config.train_val_df_pkl_path), 'wb') as handle:
                pickle.dump(self.train_val_df, handle, protocol = pickle.HIGHEST_PROTOCOL)
        else:
            # pickle load the train_val_df
            with open(os.path.join(config.pkl_dir_path, config.train_val_df_pkl_path), 'rb') as handle:
                self.train_val_df = pickle.load(handle)
        
        self.new_df = self.train_val_df.iloc[self.the_chosen, :] # this is the sampled train_val data
    
        if not os.path.exists(os.path.join(config.pkl_dir_path, config.disease_classes_pkl_path)):
            # pickle dump the classes list
            with open(os.path.join(config.pkl_dir_path, config.disease_classes_pkl_path), 'wb') as handle:
                pickle.dump(self.all_classes, handle, protocol = pickle.HIGHEST_PROTOCOL)
                print('\n{}: dumped'.format(config.disease_classes_pkl_path))
        else:
            pass

        for i in range(len(self.new_df)):
            row = self.new_df.iloc[i, :]
            labels = str.split(row['Finding Labels'], '|')
            for lab in labels:  
                lab_idx = self.all_classes.index(lab)
                if lab_idx == 14: # No Finding
                    continue
                self.disease_cnt[lab_idx] += 1

        self.total_ds_cnt = np.array(self.disease_cnt)
        # Normalize the imbalance
        self.imbalance = 0
        difference_cnt = self.total_ds_cnt - self.total_ds_cnt.mean()
        for i in range(len(difference_cnt)):
            difference_cnt[i] = difference_cnt[i] * difference_cnt[i]        
        for i in range(len(difference_cnt)):
            difference_cnt[i] = difference_cnt[i] / difference_cnt.sum()
        # Calculate the level of imbalnce
        difference_cnt -= difference_cnt.mean()
        for i in range(len(difference_cnt)):
            difference_cnt[i] = (difference_cnt[i] * difference_cnt[i])
    
        self.imbalance = 1 / difference_cnt.sum()

    def get_ds_cnt(self, c_num):

        raw_pos_freq = self.total_ds_cnt
        raw_neg_freq = self.total_ds_cnt.sum() - self.total_ds_cnt

        return raw_pos_freq, raw_neg_freq
            
    def compute_class_freqs(self):
        # total number of patients (rows)
        labels = self.train_val_df ## What is the shape of this???
        N = labels.shape[0]
        positive_frequencies = (labels.sum(axis = 0))/N
        negative_frequencies = 1.0 - positive_frequencies
    
        return positive_frequencies, negative_frequencies

    def make_pkl_dir(self, pkl_dir_path):
        if not os.path.exists(pkl_dir_path):
            os.mkdir(pkl_dir_path)

    def get_train_val_df(self):

        # get the list of train_val data 
        train_val_list = self.get_train_val_list()
        print("train_va_list: ",len(train_val_list))

        train_val_df = pd.DataFrame()
        print('\nbuilding train_val_df...')
        for i in tqdm(range(self.df.shape[0])):
            filename  = os.path.basename(self.df.iloc[i,0])
            if filename in train_val_list:
                train_val_df = train_val_df.append(self.df.iloc[i:i+1, :])
        return train_val_df

    def __getitem__(self, index):

        self.all_classes = ['Cardiomegaly','Emphysema','Effusion','Hernia','Infiltration','Mass','Nodule','Atelectasis','Pneumothorax','Pleural_Thickening','Pneumonia','Fibrosis','Edema','Consolidation', 'No Finding']
        row = self.new_df.iloc[index, :]
        # img = cv2.imread(row['image_links'])
        img = Image.open(row['image_links'])
        labels = str.split(row['Finding Labels'], '|')
        target = torch.zeros(len(self.all_classes))
        new_target = torch.zeros(len(self.all_classes) - 1)
        for lab in labels:
            lab_idx = self.all_classes.index(lab)
            target[lab_idx] = 1            
        if self.transform is not None:
            img = self.transform(img)

        return img, target[:14]
       
    def get_df(self):
        csv_path = os.path.join(self.data_dir, 'Data_Entry_2017.csv')
        all_xray_df = pd.read_csv(csv_path)

        df = pd.DataFrame()        
        df['image_links'] = [x for x in glob.glob(os.path.join(self.data_dir, 'images*', '*', '*.png'))]

        df['Image Index'] = df['image_links'].apply(lambda x : x[len(x)-16:len(x)])
        merged_df = df.merge(all_xray_df, how = 'inner', on = ['Image Index'])
        merged_df = merged_df[['image_links','Finding Labels']]
        return merged_df
    
    def get_train_val_list(self):
        f = open("data/NIH/train_val_list(original)", 'r')
        train_val_list = str.split(f.read(), '\n')
        return train_val_list

    def __len__(self):
        return len(self.the_chosen)
    
    def get_name(self):
        return 'NIH'

    def get_class_cnt(self):
        return 14

class NIHTestDataset(Dataset):

    def __init__(self, data_dir, transform = None):
        self.data_dir = data_dir
        self.transform = transform
        # full dataframe including train_val and test set
        self.df = self.get_df()
        self.make_pkl_dir(config.pkl_dir_path)
        self.disease_cnt = [0]*14
        self.all_classes = ['Cardiomegaly','Emphysema','Effusion','Hernia','Infiltration','Mass','Nodule','Atelectasis','Pneumothorax','Pleural_Thickening','Pneumonia','Fibrosis','Edema','Consolidation', 'No Finding']

        # loading the classes list
        with open(os.path.join(config.pkl_dir_path, config.disease_classes_pkl_path), 'rb') as handle:
            self.all_classes = pickle.load(handle) 
        # get test_df
        if not os.path.exists(os.path.join(config.pkl_dir_path, config.test_df_pkl_path)):
            self.test_df = self.get_test_df()
            with open(os.path.join(config.pkl_dir_path, config.test_df_pkl_path), 'wb') as handle:
                pickle.dump(self.test_df, handle, protocol = pickle.HIGHEST_PROTOCOL)
            print('\n{}: dumped'.format(config.test_df_pkl_path))
        else:
            # pickle load the test_df
            with open(os.path.join(config.pkl_dir_path, config.test_df_pkl_path), 'rb') as handle:
                self.test_df = pickle.load(handle)

        for i in range(len(self.test_df)):
            row = self.test_df.iloc[i, :]
            labels = str.split(row['Finding Labels'], '|')
            for lab in labels:  
                lab_idx = self.all_classes.index(lab)
                if lab_idx == 14: # No Finding
                    continue
                self.disease_cnt[lab_idx] += 1

    def get_ds_cnt(self):
        return self.disease_cnt

    def __getitem__(self, index):
        row = self.test_df.iloc[index, :]
        # img = cv2.imread(row['image_links'])
        img = Image.open(row['image_links'])
        labels = str.split(row['Finding Labels'], '|')
        target = torch.zeros(len(self.all_classes)) # 15
        for lab in labels:
            lab_idx = self.all_classes.index(lab)
            target[lab_idx] = 1     
        if self.transform is not None:
            img = self.transform(img)
        return img, target[:14]

    def make_pkl_dir(self, pkl_dir_path):
        if not os.path.exists(pkl_dir_path):
            os.mkdir(pkl_dir_path)

    def get_df(self):
        csv_path = os.path.join(self.data_dir, 'Data_Entry_2017.csv')
        all_xray_df = pd.read_csv(csv_path)
        df = pd.DataFrame()        
        df['image_links'] = [x for x in glob.glob(os.path.join(self.data_dir, 'images*', '*', '*.png'))]
        df['Image Index'] = df['image_links'].apply(lambda x : x[len(x)-16:len(x)])
        merged_df = df.merge(all_xray_df, how = 'inner', on = ['Image Index'])
        merged_df = merged_df[['image_links','Finding Labels']]
        return merged_df

    def get_test_df(self):
        # get the list of test data 
        test_list = self.get_test_list()
        test_df = pd.DataFrame()
        for i in tqdm(range(self.df.shape[0])):
            filename  = os.path.basename(self.df.iloc[i,0])
            if filename in test_list:
                test_df = test_df.append(self.df.iloc[i:i+1, :])
        return test_df

    def get_test_list(self):
        f = open( os.path.join('data/NIH', 'test_list.txt'), 'r')
        test_list = str.split(f.read(), '\n')
        return test_list

    def __len__(self):
        return len(self.test_df)

class ChexpertTrainDataset(Dataset):

    def __init__(self,c_num, transform = None, indices = None):
        
        csv_path = "data/CheXpert-v1.0-small/selected_train.csv"
        self.dir = "data/"
        self.transform = transform

        self.all_data = pd.read_csv(csv_path)
        self.selecte_data = self.all_data.iloc[indices, :]
        self.class_num = 10
        self.all_classes = ['Enlarged Cardiomediastinum', 'Cardiomegaly', 'Lung Opacity', 'Lung Lesion', 'Edema', 'Consolidation', 'Pneumonia', 'Atelectasis', 'Pneumothorax', 'Fracture']
        
        self.total_ds_cnt = self.get_total_cnt()
        self.total_ds_cnt = np.array(self.total_ds_cnt)
        # Normalize the imbalance
        self.imbalance = 0
        difference_cnt = self.total_ds_cnt - self.total_ds_cnt.mean()
        for i in range(len(difference_cnt)):
            difference_cnt[i] = difference_cnt[i] * difference_cnt[i]        
        for i in range(len(difference_cnt)):
            difference_cnt[i] = difference_cnt[i] / difference_cnt.sum()
        # Calculate the level of imbalnce
        difference_cnt -= difference_cnt.mean()
        for i in range(len(difference_cnt)):
            difference_cnt[i] = (difference_cnt[i] * difference_cnt[i])
    
        self.imbalance = 1 / difference_cnt.sum()

    def __getitem__(self, index):

        row = self.selecte_data.iloc[index, :]
        # img = cv2.imread(self.dir + row['Path'])
        img = pilimg.open(self.dir + row['Path'])
        label = torch.FloatTensor(row[2:])
        gray_img = self.transform(img)
        return torch.cat([gray_img,gray_img,gray_img], dim = 0), label

    def __len__(self):
        return len(self.selecte_data)

    def get_total_cnt(self):
        total_ds_cnt = [0] * self.class_num
        for i in range(len(self.selecte_data)):
            row = self.selecte_data.iloc[i, 2:]
            for j in range(len(row)):
                total_ds_cnt[j] += int(row[j])
        return total_ds_cnt

    def get_ds_cnt(self):

        raw_pos_freq = self.total_ds_cnt
        raw_neg_freq = self.total_ds_cnt.sum() - self.total_ds_cnt

        return raw_pos_freq, raw_neg_freq

    def get_name(self):
        return 'CheXpert'

    def get_class_cnt(self):
        return 10

class ChexpertTestDataset(Dataset):

    def __init__(self, transform = None):
        
        csv_path = "data/CheXpert-v1.0-small/selected_test.csv"
        self.dir = "data/"
        self.transform = transform

        self.all_data = pd.read_csv(csv_path)
        self.selecte_data = self.all_data.iloc[:, :]
        self.class_num = 10

    def __getitem__(self, index):

        row = self.selecte_data.iloc[index, :]
        img = pilimg.open(self.dir + row['Path'])
        label = torch.FloatTensor(row[2:])
        gray_img = self.transform(img)

        return torch.cat([gray_img,gray_img,gray_img], dim = 0), label

    def get_ds_cnt(self):
        total_ds_cnt = [0] * self.class_num
        for i in range(len(self.selecte_data)):
            row = self.selecte_data.iloc[i, 2:]
            for j in range(len(row)):
                total_ds_cnt[j] += int(row[j])
        return total_ds_cnt

    def __len__(self):
        return len(self.selecte_data)


def record_net_data_stats(y_train, net_dataidx_map):
    net_cls_counts = {}

    for net_i, dataidx in net_dataidx_map.items():
        unq, unq_cnt = np.unique(y_train[dataidx], return_counts=True)
        tmp = {unq[i]: unq_cnt[i] for i in range(len(unq))}
        net_cls_counts[net_i] = tmp
    logging.debug('Data statistics: %s' % str(net_cls_counts))
    return net_cls_counts

def _data_transforms_cifar(datadir):
    if "cifar100" in datadir:
        CIFAR_MEAN = [0.5071, 0.4865, 0.4409]
        CIFAR_STD = [0.2673, 0.2564, 0.2762]
    else:
        CIFAR_MEAN = [0.49139968, 0.48215827, 0.44653124]
        CIFAR_STD = [0.24703233, 0.24348505, 0.26158768]

    train_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])

    valid_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])

    return train_transform, valid_transform

def _data_transforms_imagenet(datadir):
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    crop_scale = 0.08
    jitter_param = 0.4
    image_size = 224
    image_resize = 256

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(image_size, scale=(crop_scale, 1.0)),
        transforms.ColorJitter(
            brightness=jitter_param, contrast=jitter_param,
            saturation=jitter_param),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])
    valid_transform = transforms.Compose([
        transforms.Resize(image_resize),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

    return train_transform, valid_transform

def _data_transforms_NIH():
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
    transform = transforms.Compose([
                                    transforms.Resize([150,150]),
                                    transforms.ToTensor(),
                                    normalize])
    return transform

def _data_transforms_ChexPert():
    normalize = transforms.Normalize(mean=[0.485],
                                 std=[0.229])
    transform = transforms.Compose([
                                    transforms.Resize([150,150]),
                                    transforms.ToTensor(),
                                    normalize])
    return transform

def load_data(datadir):
    if 'cifar' in datadir:
        train_transform, test_transform = _data_transforms_cifar(datadir)
        dl_obj = CIFAR_truncated
    else:
        train_transform, test_transform = _data_transforms_imagenet(datadir)
        dl_obj = ImageFolder_custom
    train_ds = dl_obj(datadir, train=True, download=True, transform=train_transform)
    test_ds = dl_obj(datadir, train=False, download=True, transform=test_transform)

    y_train, y_test = train_ds.target, test_ds.target

    return (y_train, y_test)

def partition_data(datadir, partition, n_nets, alpha):
    logging.info("*********partition data***************")
    

    if partition == "homo":
        total_num = n_train
        idxs = np.random.permutation(total_num)
        batch_idxs = np.array_split(idxs, n_nets)
        net_dataidx_map = {i: batch_idxs[i] for i in range(n_nets)}

    elif partition == "hetero":
        net_dataidx_map = {}
        min_size = 0
        idx_batch = [[] for _ in range(n_nets)] # n_nuts : the number of clients
        client_pos_proportions = []
        client_pos_freq = []
        client_neg_proportions = []
        client_neg_freq = []
        client_imbalances = []
        # [[], [], [], [], [], [], [], [], [], []] # the number of clients
        # for each class in the dataset
        if 'NIH' in datadir or 'CheXpert' in datadir:
            N = 86336
            idx_k = np.array(list(range(N)))
            np.random.shuffle(idx_k)
            while min_size < 10:
                proportions = np.random.dirichlet(np.repeat(alpha, n_nets))
                proportions = np.array([p * (len(idx_j) < N / n_nets) for p, idx_j in zip(proportions, idx_batch)])
                proportions = proportions / proportions.sum()
                
                proportions = (np.cumsum(proportions) * N).astype(int)[:-1]
                idx_batch = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch, np.split(idx_k, proportions))]
                min_size = min([len(idx_j) for idx_j in idx_batch])
            
            for j in range(n_nets):
                np.random.shuffle(idx_batch[j]) # shuffle once more
                net_dataidx_map[j] = idx_batch[j]
                
            return net_dataidx_map
        
        elif 'cifar100' in datadir:
            long_tail = get_img_num_per_cls('100')
            print("Long Tail: ", long_tail)
            y_train, y_test = load_data(datadir)
            n_train = y_train.shape[0]
            n_test = y_test.shape[0]
            class_num = len(np.unique(y_train))
            min_size = 0
            K = class_num
            N = n_train
            logging.info("N = " + str(N))
            class_proportions = []
            class_freq = []
            
            while min_size < 10:
                proportionss = np.random.dirichlet(np.repeat(alpha, n_nets))
                idx_batch = [[] for _ in range(n_nets)]
                class_proportions = []
                class_freq = []
                class_neg_proportions = []
                class_neg_freq = []
                
                for k in range(K): # partition for the class k
                    idx_k = np.where(y_train == k)[0]
                    np.random.shuffle(idx_k)
                    idx_k = idx_k[:long_tail[k]]

                    proportionss = np.random.dirichlet(np.repeat(alpha, n_nets))
                    proportions = proportionss / proportionss.sum()

                    class_proportions.append(proportions)
                    class_freq.append((proportions * len(idx_k)).astype(int))
                    class_neg_proportions.append(1 - proportions)
                    class_neg_freq.append(((1 - proportions) * len(idx_k)).astype(int))
                    
                    min_data = min((proportions * len(idx_k)).astype(int))
                    
                    # Same as above
                    proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
                    # cumsum : cumulative summation
                    idx_batch = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch, np.split(idx_k, proportions))]

                    # fivide idx_k according to the proportion
                    min_size = min([len(idx_j) for idx_j in idx_batch])

            client_pos_freq = np.array(class_freq)
            client_pos_freq = client_pos_freq.T
            client_neg_freq = np.array(class_neg_freq)
            client_neg_freq = client_neg_freq.T

            # Get clients' degree of data imbalances.
            for i in range(n_nets):
                difference_cnt = client_pos_freq[i] - client_pos_freq[i].mean()
                for i in range(len(difference_cnt)):
                    difference_cnt[i] = difference_cnt[i] * difference_cnt[i]        
                for i in range(len(difference_cnt)):
                    difference_cnt[i] = difference_cnt[i] / difference_cnt.sum()
                # Calculate the level of imbalnce
                difference_cnt -= difference_cnt.mean()
                for i in range(len(difference_cnt)):
                    difference_cnt[i] = (difference_cnt[i] * difference_cnt[i])
                client_imbalances.append(1 / difference_cnt.sum())

            client_imbalances = np.array(client_imbalances)
            client_imbalances =  client_imbalances / client_imbalances.sum()

            for j in range(n_nets):
                np.random.shuffle(idx_batch[j]) # shuffle once more
                net_dataidx_map[j] = idx_batch[j]

            traindata_cls_counts = record_net_data_stats(y_train, net_dataidx_map)

            # the number of class, shuffled indices, record of it
            return class_num, net_dataidx_map, traindata_cls_counts, client_pos_freq, client_neg_freq, client_imbalances

        
        elif 'cifar10' in datadir:
            long_tail = get_img_num_per_cls('10')
            print("Long tail:", long_tail)
            y_train, y_test = load_data(datadir)
            n_train = y_train.shape[0]
            n_test = y_test.shape[0]
            class_num = len(np.unique(y_train))
            min_size = 0
            K = class_num
            N = n_train
            logging.info("N = " + str(N))
            while min_size < 10:
                idx_batch = [[] for _ in range(n_nets)]
                class_proportions = []
                class_freq = []
                class_neg_proportions = []
                class_neg_freq = []
                # proportionss = np.random.dirichlet(np.repeat(alpha, n_nets))
                
                for k in range(K): # partition for the class k
                    idx_k = np.where(y_train == k)[0]
                    np.random.shuffle(idx_k)
                    proportionss = np.random.dirichlet(np.repeat(alpha, n_nets))
                    proportions = proportionss / proportionss.sum()

                    class_proportions.append(proportions)
                    class_freq.append((proportions * len(idx_k)).astype(int))
                    class_neg_proportions.append(1 - proportions)
                    class_neg_freq.append(((1 - proportions) * len(idx_k)).astype(int))
                    
                    min_data = min((proportions * len(idx_k)).astype(int))
                    
                    # Same as above
                    proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
                    # cumsum : cumulative summation
                    idx_batch = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch, np.split(idx_k, proportions))]

                    min_size = min([len(idx_j) for idx_j in idx_batch])

            client_pos_freq = np.array(class_freq)
            client_pos_freq = client_pos_freq.T
            client_neg_freq = np.array(class_neg_freq)
            client_neg_freq = client_neg_freq.T

            # Get clients' degree of data imbalances.
            for i in range(n_nets):
                difference_cnt = client_pos_freq[i] - client_pos_freq[i].mean()
                for i in range(len(difference_cnt)):
                    difference_cnt[i] = difference_cnt[i] * difference_cnt[i]        
                for i in range(len(difference_cnt)):
                    difference_cnt[i] = difference_cnt[i] / difference_cnt.sum()
                # Calculate the level of imbalnce
                difference_cnt -= difference_cnt.mean()
                for i in range(len(difference_cnt)):
                    difference_cnt[i] = (difference_cnt[i] * difference_cnt[i])
                client_imbalances.append(1 / difference_cnt.sum())

            client_imbalances = np.array(client_imbalances)
            client_imbalances =  client_imbalances / client_imbalances.sum()

            for j in range(n_nets):
                np.random.shuffle(idx_batch[j]) # shuffle once more
                net_dataidx_map[j] = idx_batch[j]

            traindata_cls_counts = record_net_data_stats(y_train, net_dataidx_map)

            # the number of class, shuffled indices, record of it
            return class_num, net_dataidx_map, traindata_cls_counts, client_pos_freq, client_neg_freq, client_imbalances

# for centralized training
def get_dataloader(datadir, train_bs, test_bs, dataidxs=None):
    ################datadir is the key to discern the dataset#######################
    if 'cifar' in datadir:
        train_transform, test_transform = _data_transforms_cifar(datadir)
        dl_obj = CIFAR_truncated
        workers=0
        persist=False
    else:
        train_transform, test_transform = _data_transforms_imagenet(datadir)
        dl_obj = ImageFolder_custom
        workers=8
        persist=True

    train_ds = dl_obj(datadir, dataidxs=dataidxs, train=True, transform=train_transform, download=True)
    test_ds = dl_obj(datadir, train=False, transform=test_transform, download=True)
    
    train_dl = data.DataLoader(dataset=train_ds, batch_size=train_bs, shuffle=True, drop_last=True, num_workers=workers, persistent_workers=persist)
    test_dl = data.DataLoader(dataset=test_ds, batch_size=test_bs, shuffle=False, drop_last=True, num_workers=workers, persistent_workers=persist)

    return train_dl, test_dl

def load_partition_data(data_dir, partition_method, partition_alpha, client_number, batch_size):

    # get local dataset
    data_local_num_dict = dict() ### form 봐서 맞춰줘야 함
    train_data_local_dict = dict()
    test_data_local_dict = dict()
    data_imbalances = []
    train_data_global = None
    test_data_global = None
    
    if 'NIH' in data_dir:
        class_num = 14
        client_imbalances = []
        client_pos_freq = []
        client_neg_freq = []
        indices = partition_data(data_dir, partition_method, client_number, partition_alpha)
        train_data_global = torch.utils.data.DataLoader(NIHTrainDataset(0, data_dir, transform = _data_transforms_NIH(), indices=list(range(86336))), batch_size = 32, shuffle = True)
        test_data_global = torch.utils.data.DataLoader(NIHTestDataset(data_dir, transform = _data_transforms_NIH()), batch_size = 32, shuffle = not True)
        train_data_num = len(train_data_global)
        test_data_num = len(test_data_global)
        # indices = distribute_indices(length, 1, client_number)
        for i in range(client_number):
            data = NIHTrainDataset(i, data_dir, transform = _data_transforms_NIH(), indices=indices[i])
            total_ds_cnt = np.array(data.total_ds_cnt)
            client_imbalances.append(data.imbalance)
            train_percentage = 0.8
            train_dataset, val_dataset = torch.utils.data.random_split(data, [int(len(data)*train_percentage), len(data)-int(len(data)*train_percentage)])
            train_loader = torch.utils.data.DataLoader(train_dataset, batch_size = 32, shuffle = True)
            val_loader = torch.utils.data.DataLoader(val_dataset, batch_size = 32, shuffle = not True)
            client_pos_freq.append(total_ds_cnt.tolist())
            client_neg_freq.append((total_ds_cnt.sum() - total_ds_cnt).tolist())
            train_data_local_dict[i] = train_loader
            test_data_local_dict[i] = val_loader
        client_imbalances = np.array(client_imbalances)
        client_imbalances = client_imbalances / client_imbalances.sum()
        client_imbalances = client_imbalances.tolist()

    elif 'CheXpert' in data_dir:
        class_num = 10
        client_imbalances = []
        client_pos_freq = []
        client_neg_freq = []
        indices = partition_data(data_dir, partition_method, client_number, partition_alpha)
        train_data_global = torch.utils.data.DataLoader(ChexpertTrainDataset(0, transform = _data_transforms_ChexPert(), indices=list(range(86336))), batch_size = 32, shuffle = True)
        test_data_global =  torch.utils.data.DataLoader(ChexpertTestDataset(transform = _data_transforms_ChexPert()), batch_size = 32, shuffle = not True)
        train_data_num = len(train_data_global)
        test_data_num = len(test_data_global)
        # indices = distribute_indices(length, 1, client_number)
        for i in range(client_number):
            data = ChexpertTrainDataset(i, transform = _data_transforms_ChexPert(), indices=indices[i])
            client_imbalances.append(data.imbalance)
            total_ds_cnt = np.array(data.total_ds_cnt)
            client_pos_freq.append(total_ds_cnt.tolist())
            client_neg_freq.append((total_ds_cnt.sum() - total_ds_cnt).tolist())
            train_percentage = 0.8
            train_dataset, val_dataset = torch.utils.data.random_split(data, [int(len(data)*train_percentage), len(data)-int(len(data)*train_percentage)])
            train_loader = torch.utils.data.DataLoader(train_dataset, batch_size = 32, shuffle = True)
            val_loader = torch.utils.data.DataLoader(val_dataset, batch_size = 32, shuffle = not True)
            train_data_local_dict[i] = train_loader
            test_data_local_dict[i] = val_loader

        client_imbalances = np.array(client_imbalances)
        client_imbalances = client_imbalances / client_imbalances.sum()
        client_imbalances = client_imbalances.tolist()

    elif ('cifar10' in data_dir) or ('cifar100' in data_dir):
        
        class_num, net_dataidx_map, traindata_cls_counts, client_pos_freq, client_neg_freq, client_imbalances = partition_data(data_dir, partition_method, client_number, partition_alpha)
        logging.info("traindata_cls_counts = " + str(traindata_cls_counts)) # report the data
        train_data_num = sum([len(net_dataidx_map[r]) for r in range(client_number)]) # overall number of data
        
        # use traindata_cls_counts to calculate the degree of imbalance

        train_data_global, test_data_global = get_dataloader(data_dir, batch_size, batch_size) # get the global data
        logging.info("train_dl_global number = " + str(len(train_data_global)))
        logging.info("test_dl_global number = " + str(len(train_data_global)))
        test_data_num = len(test_data_global)

        for client_idx in range(client_number):
            dataidxs = net_dataidx_map[client_idx]
            print(dataidxs)
            local_data_num = len(dataidxs)
            data_local_num_dict[client_idx] = local_data_num
            logging.info("client_idx = %d, local_sample_number = %d" % (client_idx, local_data_num))

            # training batch size = 64; algorithms batch size = 32
            train_data_local, test_data_local = get_dataloader(data_dir, batch_size, batch_size, dataidxs)
            logging.info("client_idx = %d, batch_num_train_local = %d, batch_num_test_local = %d" % (
                client_idx, len(train_data_local), len(test_data_local)))
            train_data_local_dict[client_idx] = train_data_local # client_number : dataloader
            test_data_local_dict[client_idx] = test_data_local

    else :
        raise ValueError("Wrong data path!")

    return train_data_num, test_data_num, train_data_global, test_data_global, \
           data_local_num_dict, train_data_local_dict, test_data_local_dict, class_num, client_pos_freq, client_neg_freq, client_imbalances
