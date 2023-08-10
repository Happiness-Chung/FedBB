'''
Main file to set up the FL system and train
Code design inspired by https://github.com/mmendiet/FedAlign
'''
import torch
import numpy as np
import random
import data_preprocessing.data_loader as dl
import argparse
from models.resnet import resnet56, resnet18
from models.resnet_fedalign import resnet56 as resnet56_fedalign
from models.resnet_fedalign import resnet18 as resnet18_fedalign
from torch.multiprocessing import set_start_method, Queue
import logging
import os
from collections import defaultdict
import time

# methods
import methods.fedavg as fedavg
import methods.fedprox as fedprox
import methods.moon as moon
import methods.fedalign as fedalign
import methods.fedbb as fedbb
import data_preprocessing.custom_multiprocess as cm

def add_args(parser):
    # Training settings
    parser.add_argument('--method', type=str, default='fedalign', metavar='N',
                        help='Options are: fedavg, fedprox, moon, fedalign, fedbb')
    
    parser.add_argument('--harmony', type=str, default='n', metavar='N')

    parser.add_argument('--longtail', type=bool, default= False, metavar='N')

    parser.add_argument('--ibf', type =float, default= 0.1, metavar='N')
    
    parser.add_argument('--data_dir', type=str, default="C:/Users/hb/Desktop/data/CheXpert-v1.0-small",
                        help='data directory: data/cifar100, data/cifar10, "C:/Users/hb/Desktop/data/NIH", C:/Users/hb/Desktop/data/CheXpert-v1.0-small')

    parser.add_argument('--dataset', type=str, default="CheXpert",
                        help='data directory: cifar100, cifar10, NIH, CheXpert')

    parser.add_argument('--partition_method', type=str, default='hetero', metavar='N',
                        help='how to partition the dataset on local clients')

    parser.add_argument('--partition_alpha', type=float, default= 1, metavar='PA',
                        help='alpha value for Dirichlet distribution partitioning of data(default: 0.5)')

    parser.add_argument('--client_number', type=int, default=5, metavar='NN',
                        help='number of clients in the FL system')

    parser.add_argument('--batch_size', type=int, default=32, metavar='N',
                        help='input batch size for training (default: 64)')

    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')

    parser.add_argument('--wd', help='weight decay parameter;', type=float, default=0.0001)

    parser.add_argument('--epochs', type=int, default=2, metavar='EP',
                        help='how many epochs will be trained locally per round')

    parser.add_argument('--comm_round', type=int, default=30,
                        help='how many rounds of communications are conducted')

    parser.add_argument('--pretrained', action='store_true', default=False,  
                        help='test pretrained model')

    parser.add_argument('--mu', type=float, default= 0.45, metavar='MU',
                        help='mu value for various methods')
    
    parser.add_argument('--tau', type=float, default = 10, metavar='TAU',
                        help='mu value for various methods')
    
    parser.add_argument('--beta', type=float, default=0.9999999, metavar='BETA',
                        help='mu value for various methods')

    parser.add_argument('--width', type=float, default=0.25, metavar='WI',
                        help='minimum width for subnet training')

    parser.add_argument('--mult', type=float, default=0.0001, metavar='MT',
                        help='multiplier for subnet training')

    parser.add_argument('--num_subnets', type=int, default=3,
                        help='how many subnets sampled during training')

    parser.add_argument('--save_client', action='store_true', default=False,
                        help='Save client checkpoints each round')

    parser.add_argument('--thread_number', type=int, default=1, metavar='NN',
                        help='number of parallel training threads')

    parser.add_argument('--client_sample', type=float, default=1.0, metavar='MT',
                        help='Fraction of clients to sample')

    parser.add_argument('--stoch_depth', default=0.5, type=float,
                    help='stochastic depth probability')
    
    parser.add_argument('--seed', type=int, default=1996, metavar='GAMMA',
                        help='When 1 : data skewness, When 0 : data amount')

    parser.add_argument('--gamma', default=0.0, type=float,
                    help='hyperparameter gamma for mixup')
    args = parser.parse_args()

    return args

# Setup Functions
def set_random_seed(seed=1):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    ## NOTE: If you want every run to be exactly the same each time
    ##       uncomment the following lines
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False

# Helper Functions
def init_process(q, Client):
    # q is the client info
    set_random_seed()
    global client # 새롭게 클라이언트를 전역으로 선언
    # c0 is a client_dict
    # c1 is the namespace
    ci = q.get() # Queued에서 맨 앞의 원소를 하나 가져오고 remove
    client = Client(ci[0], ci[1]) 

def run_clients(received_info):
    try:
        return client.run(received_info) # give threads' number of model weight
    except KeyboardInterrupt:
        logging.info('exiting')
        return None

def allocate_clients_to_threads(args):
    mapping_dict = defaultdict(list) # default 값이 list인 dictionary
    for round in range(args.comm_round): # 각 communication round마다 thread를 할당 해 줌
        if args.client_sample<1.0: # 여러 클라이언트들 중에서 sampling하여 트레이닝을 진행하는 경우
            num_clients = int(args.client_number*args.client_sample)
            client_list = random.sample(range(args.client_number), num_clients)
        else: # 주어진 클라이언트를 모두 사용하는 경우
            num_clients = args.client_number
            client_list = list(range(num_clients))
        if num_clients % args.thread_number==0 and num_clients>0:
            clients_per_thread = int(num_clients/args.thread_number) # 클라이언트 1명당 스레드 보통 1개 할당
            for c, t in enumerate(range(0, num_clients, clients_per_thread)):
                idxs = [client_list[x] for x in range(t, t+clients_per_thread)] # 앞에서 부터 차례대로 할당
                mapping_dict[c].append(idxs)
        else:
            raise ValueError("Sampled client number not divisible by number of threads")
    return mapping_dict

if __name__ == "__main__":
    try:
     set_start_method('spawn')
    except RuntimeError:
        pass
    set_random_seed()
    # get arguments
    parser = argparse.ArgumentParser()
    args = add_args(parser)
 
    ###################################### get data
    train_data_num, test_data_num, train_data_global, test_data_global, data_local_num_dict, train_data_local_dict, test_data_local_dict,\
         class_num, client_pos_freq, client_neg_freq, client_imbalances = dl.load_partition_data(args.data_dir, args.partition_method, args.partition_alpha, args.client_number, args.batch_size)
    print(client_imbalances)
    
    # train_data_num = 50000
    # test_data_num = 312
    # train_data_global, test_data_global = global train, test dataloader
    # data_local_num_dict = each clients' number of data
    # train_data_local_dict, test_data_local_dict = each clients' train, test dataloader

    mapping_dict = allocate_clients_to_threads(args) # client에게 할당된 thread number
    print("Client allocation for the threads during commication round : ", mapping_dict)
    # {0: [[0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1]] ...
    # init method and model type
    # client를 여러개 생성하지 않았다는 특징이 있음
    if args.method=='fedavg':
        Server = fedavg.Server
        Client = fedavg.Client
        Model = resnet56 
        server_dict = {'train_data':train_data_global, 'test_data': test_data_global, 'model_type': Model, 'num_classes': class_num, 'dir': args.data_dir}
        client_dict = [{'train_data':train_data_local_dict, 'test_data': test_data_local_dict, 'device': i % torch.cuda.device_count(),
                            'client_map':mapping_dict[i], 'model_type': Model, 'num_classes': class_num, 'dir': args.data_dir} for i in range(args.thread_number)]
    elif args.method=='fedprox':
        Server = fedprox.Server
        Client = fedprox.Client
        Model = resnet56 
        server_dict = {'train_data':train_data_global, 'test_data': test_data_global, 'model_type': Model, 'num_classes': class_num, 'dir': args.data_dir, 'harmony': args.harmony, 'imbalances': client_imbalances}
        client_dict = [{'train_data':train_data_local_dict, 'test_data': test_data_local_dict, 'device': i % torch.cuda.device_count(),
                            'client_map':mapping_dict[i], 'model_type': Model, 'num_classes': class_num, 'dir': args.data_dir, 'harmony': args.harmony,
                            'clients_pos': client_pos_freq, 'clients_neg': client_neg_freq} for i in range(args.thread_number)]
    elif args.method=='moon':
        Server = moon.Server
        Client = moon.Client
        Model = resnet56 
        server_dict = {'train_data':train_data_global, 'test_data': test_data_global, 'model_type': Model, 'num_classes': class_num, 'dir': args.data_dir, 'harmony': args.harmony,'imbalances': client_imbalances}
        client_dict = [{'train_data':train_data_local_dict, 'test_data': test_data_local_dict, 'device': i % torch.cuda.device_count(),
                            'client_map':mapping_dict[i], 'model_type': Model, 'num_classes': class_num, 'dir': args.data_dir, 'harmony': args.harmony,
                            'clients_pos': client_pos_freq, 'clients_neg': client_neg_freq} for i in range(args.thread_number)]
    elif args.method=='fedalign':
        Server = fedalign.Server
        Client = fedalign.Client
        Model = resnet56_fedalign 
        width_range = [args.width, 1.0]
        resolutions = [32] if 'cifar' in args.data_dir else [224]
        server_dict = {'train_data':train_data_global, 'test_data': test_data_global, 'model_type': Model, 'num_classes': class_num, 'dir': args.data_dir, 'harmony': args.harmony,'imbalances': client_imbalances}
        client_dict = [{'train_data':train_data_local_dict, 'test_data': test_data_local_dict, 'device': i % torch.cuda.device_count(),
                            'client_map':mapping_dict[i], 'model_type': Model, 'num_classes': class_num, 
                            'width_range': width_range, 'resolutions': resolutions, 'dir': args.data_dir, 'harmony': args.harmony,
                            'clients_pos': client_pos_freq, 'clients_neg': client_neg_freq} for i in range(args.thread_number)]
    elif args.method=='fedbb':
        Server = fedbb.Server
        Client = fedbb.Client
        Model = resnet56
        width_range = [args.width, 1.0]
        resolutions = [32] if 'cifar' in args.data_dir else [224]
        server_dict = {'train_data':train_data_global, 'test_data': test_data_global, 'model_type': Model, 'num_classes': class_num, 'dir': args.data_dir, 'imbalances': client_imbalances}
        client_dict = [{'train_data':train_data_local_dict, 'test_data': test_data_local_dict, 'device': i % torch.cuda.device_count(),
                            'client_map':mapping_dict[i], 'model_type': Model, 'num_classes': class_num, 
                            'width_range': width_range, 'resolutions': resolutions, 'dir': args.data_dir,
                            'clients_pos': client_pos_freq, 'clients_neg': client_neg_freq} for i in range(args.thread_number)]
    else:
        raise ValueError('Invalid --method chosen! Please choose from availible methods.')
    
    #init nodes
    client_info = Queue()
    for i in range(args.thread_number):# thread의 갯수 만큼 client dict와 args를 복사해서 생성해서 client_info에 넣어준다
        client_info.put((client_dict[i], args))
    # the length of the client info is the number of threads

    ######################################################
    # Start server and get initial outputs
    pool = cm.MyPool(args.thread_number, init_process, (client_info, Client)) 
    # thread의 갯수 만큼 init_process 실행(일종의 멀티 프로세스 초기화 함수)
    # args.thread_number : 현재 시스템에서 사용할 프로세스의 갯수
    # thread 갯수 만큼의 client_dict와 client객체 하나를 인수로 넘겨줌 
    # -> thread 갯수 만큼의 client 생성
    # init server
    server_dict['save_path'] = '{}/logs/{}__{}__{}_e{}_c{}'.format(os.getcwd(), args.dataset, time.strftime("%Y%m%d_%H%M%S"), args.method, args.epochs, args.client_number)
    if not os.path.exists(server_dict['save_path']):
        os.makedirs(server_dict['save_path'])
    server = Server(server_dict, args) # Server initializaion
    # methods.fedavg.Server object
    server_outputs = server.start() ########### Server의 모델을 반환
    # weight of the server
    # Start Federated Training
    # the length is the number of treads
    time.sleep(150*(args.client_number/16)) #  Allow time for threads to start up
    for r in range(args.comm_round):
        logging.info('***** Round: {} ************************'.format(r))
        round_start = time.time()
        # server output length :        
        # map 함수는 자체적으로 iteration 기능이 포함되어있어서 thread에 갯수만큼 server output을 하나씩 run_client에 넣어주면서 thread의 갯수만큼 실행됨
        client_outputs = pool.map(run_clients, server_outputs) # 함수 하나와 그 함수가 프로세스의 갯수만큼 실행되는동안 하나씩 들어갈 인수 리스트
        client_outputs = [c for sublist in client_outputs for c in sublist]  ##########자세히 client output form 확인 요망
        # sublist : 'weights': OrderedDict
        # length : the number of clients
        # c is the weight of a client   
        server_outputs = server.run(client_outputs) # client_output에 imbalance를 집어 넣는 것도 좋을 듯
        round_end = time.time()
        total_sec = round_end-round_start
        total_min = (total_sec) // 60
        logging.info('Round {} Time: {:.0f}m {:.0f}s'.format(r, total_min, total_sec % 60))
    pool.close()
    pool.join()
