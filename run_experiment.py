import argparse, os, sys, importlib
from DL_attacks.utils import EarlyStopping, setup_data, setup_model, setup_data_without_attack
from DL_attacks.attacker_acc_logger import acc_logger
from tqdm import tqdm
import torch
import numpy as np
import random
import datetime

def get_parser(**parser_kwargs):
    def str2bool(v):
        if isinstance(v, bool):
            return v
        if v.lower() in ("yes", "true", "t", "y", "1"):
            return True
        elif v.lower() in ("no", "false", "f", "n", "0"):
            return False
        else:
            raise argparse.ArgumentTypeError("Boolean value expected.")

    parser = argparse.ArgumentParser(**parser_kwargs)
    
    parser.add_argument(
        "-s",
        "--setting",
        help="attack setting",
        choices=['s1', 's2', 's3', 's4', 's5','s6'],
        default='s6'
    )
    
    parser.add_argument(
        "-a",
        "--attack",
        help="attacker attack type",
        choices=['norm', 'unitnorm', 'angle'],
        default='angle'
    )

    parser.add_argument(
        "-d",
        "--defense",
        help="normal users defense type",
        choices=['trim', 'median'],
        default='median'
    )
    
    parser.add_argument(
        "-dt",
        "--data",
        help="dataset setup file",
        required=True,
        default='exp_setups.CIFAR10'
    )
    
    parser.add_argument(
        "-g",
        "--graph",
        help="graph setup file",
        required=True,
        default='exp_setups.complex40'
    )
    
    parser.add_argument(
        "-lr",
        help="customized init learning rate for training, used \
            different initial lr from the config file",
        default=None
    )
    
    parser.add_argument(
        "-r",
        "--victim_ratio",
        help="customized victim parameter weight when combine victim and cover params, \
            used different ratio from the config file",
        default=None,
        type=float
    )
    
    parser.add_argument(
        "-m",
        "--member",
        help="Number of member/ non-member",
        default=None,
        type=int
    )
    
    parser.add_argument(
        "--device",
        help="device index",
        default=0
    )
    
    parser.add_argument(
        "--dist",
        help="iid or non-iid setting",
        default='non-iid',
        type=str
    )
    
    parser.add_argument(
        "--cover_times",
        help="Number of times to try cover set",
        default=None
    )
    
    parser.add_argument(
        "--seed",
        help="seed for reproduction",
        default=42,
        type=int
    )
    
    parser.add_argument(
        "-o",
        "--output_dir",
        help="log output direction",
        default='./results-agrevader',
        type=str
    )

    args = parser.parse_args()
    return args
        
def seed_everything(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)  
    
if __name__ == '__main__':
    
    args = get_parser()
    seed_everything(args.seed)
    ds_setup_file = args.data
    graph_setup_file = args.graph
    
    data_distribution = args.dist
    setting = args.setting
    attack_type = args.attack # norm, unitnorm, angle, None
    defense_type = args.defense # None, trimX, median
    device = f'cuda:{args.device}'
     
    Cds = importlib.import_module(ds_setup_file)
    Ctop = importlib.import_module(graph_setup_file)

    init_lr = args.lr if args.lr is not None else Cds.init_lr
    victim_ratio = args.victim_ratio if args.victim_ratio is not None else Cds.victim_ratio
    cover_try_time = args.cover_times if args.cover_times is not None else Cds.cover_try_time
    num_member = args.member if args.member is not None else Cds.num_member
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M")
    num_attack_user = 1
    dataset = Cds.dsk
    
    # acc logger for train acc, test acc and attack acc
    acc_log_name = f'{dataset}_{Ctop.name}_{setting}_{data_distribution}_{attack_type}_{defense_type}_bs{Cds.attacker_batch_size}_r{victim_ratio}_NIter{Cds.normal_train_iter}_T{cover_try_time}_{timestamp}.txt'
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    acc_log_path = os.path.join(args.output_dir, acc_log_name)
    print(f"Acc log in file --> {acc_log_path}")
    
    
    print("Running setup ....")
    size_local_ds = Cds.compute_local_training_set_size(Ctop.nu)
    # loads and splits local training sets and test one (validation)
    if attack_type == 'None':
        # gets users' local training size
        train_sets, test_set, cover_set, x_shape, num_class = setup_data_without_attack(
            Cds.load_dataset,
            Ctop.nu,
            size_local_ds,
            Cds.user_batch_size,
            Cds.size_testset,
            0 if data_distribution == 'iid' else 1 # iid: 0, non_iid: 1
        )
    else:
        # NOTE: add cover set
        train_sets, test_set, cover_set, x_shape, num_class = setup_data(
            Cds.load_dataset,
            Ctop.nu,
            size_local_ds,
            Cds.user_batch_size,
            Cds.size_testset,
            0 if data_distribution == 'iid' else 1, # iid: 0, non_iid: 1
            num_member, num_member, size_local_ds,
            setting
        )
    print(f'user dataset size: {size_local_ds}')
    
    # setup model generator function
    make_model = setup_model(
        Cds.model_maker,
        [x_shape, num_class, init_lr, Cds.lrd, Ctop.pretrain, Ctop.checkpoint_path],
        Ctop.model_same_init #! model_same_init: True --> 所有用户的模型初始化相同
    ) #! setup_model --> DL_attacks.utils.setup_model --> 初始化一个模型生成函数
    
    # define comm. topology
    DL = Ctop.CDL(Ctop.graph_properties) #! CDL: DecentralizedLearning --> DL_attacks.DL.DecentralizedLearning --> 初始化一个DecentralizedLearning对象
    if Ctop.G is None:
        DL.setup(Ctop.nu, make_model, train_sets, test_set, cover_set, Ctop.USER, Ctop.ATTACKER, \
            device, Cds.normal_train_iter, attack_type, defense_type, cover_try_time, victim_ratio, Cds.attacker_batch_size) #! Attacker从哪来？: __init__.py中的ATTACKER
    else:
        DL.from_nx_graph(Ctop.G, make_model, train_sets, test_set, Ctop.USER, Ctop.ATTACKER, device)

    # it runs and logs metric during the training, including privacy risk
    acc_logger = acc_logger(Ctop, DL, acc_log_path)
    # log setting 
    acc_logger.logger.info('************* DL Setting *************')
    acc_logger.logger.info(f'   Setting = {setting}')
    acc_logger.logger.info(f'   Graph = {Ctop.name}')
    acc_logger.logger.info(f'   Number of Attacker = {num_attack_user}')
    acc_logger.logger.info(f'   Sample per user/ Num of Cover = {size_local_ds}')
    acc_logger.logger.info(f'   Num of member/non-member = {num_member}')
    acc_logger.logger.info(f'   Normal User Batch size = {Cds.user_batch_size}')
    acc_logger.logger.info(f'   Attacker Batch size = {Cds.attacker_batch_size}')
    acc_logger.logger.info(f'   Attack = {attack_type}')
    acc_logger.logger.info(f'   Defense = {defense_type}')
    acc_logger.logger.info(f'   data distribution = {data_distribution}')
    acc_logger.logger.info(f'   Init LR = {init_lr}')
    acc_logger.logger.info(f'   LR Decay = {Cds.lrd}')
    acc_logger.logger.info(f'   Max Iter = {Ctop.max_num_iter}')
    acc_logger.logger.info(f'   Normal Train Iter = {Cds.normal_train_iter}')
    acc_logger.logger.info(f'**************************************')
    
    # it implements early stopping
    # es = EarlyStopping(Cds.patience) #! EarlyStopping --> DL_attacks.utils.EarlyStopping --> 初始化一个EarlyStopping对象
    
    ## Main training loop
    print("Training ....")
    for i in tqdm(range(1, Cds.max_num_iter+1)):
        if i > 399 and i < 500:
            eval_interval = 10
        else:
            eval_interval = Cds.eval_interval
        # run a round of DL
        DL(i) #! DL() --> DecentralizedLearning.__call__() --> 一轮迭代过程
        
       # eval models  
        if i % eval_interval == 0 and i: #! eval_interval: 25 --> 每25个iteration进行一次evaluation
            # logs privacy risk (slow operation)
            if hasattr(DL.attacker, 'evaluate_attack_result'):
                DL.attacker.evaluate_attack_result()
            acc_logger(i)

    acc_logger.log_best_result(None, None)
