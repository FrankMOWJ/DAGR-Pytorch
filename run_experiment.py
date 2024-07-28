import os, sys, importlib
from DL_attacks.utils import EarlyStopping, setup_data, setup_model, setup_data_without_attack
from DL_attacks.attacker_acc_logger import acc_logger
from tqdm import tqdm

if __name__ == '__main__':
    
    try:
        ds_setup_file = sys.argv[1]
        top_setup_file = sys.argv[2]
        run_num = sys.argv[3]
    except:
        print(f"USAGE: dataset_setup_file topology_setup_file run_number")
        sys.exit(1)
        
    
    Cds = importlib.import_module(ds_setup_file)
    Ctop = importlib.import_module(top_setup_file)

    # acc logger for train acc, test acc and attack acc
    if not os.path.exists(Ctop.output_dir):
        os.makedirs(Ctop.output_dir)
    acc_log_name = f'{Ctop.name}' + Ctop.acc_log_name 
    acc_log_path = os.path.join(Ctop.output_dir, acc_log_name)
    print(f"Acc log in file --> {acc_log_path}")
    
    
    print("Running setup ....")
    # loads and splits local training sets and test one (validation)
    if Ctop.attack_type == 'None':
        # gets users' local training size
        size_local_ds = Cds.compute_local_training_set_size(Ctop.nu)
        train_sets, test_set, cover_set, x_shape, num_class = setup_data_without_attack(
            Cds.load_dataset,
            Ctop.nu,
            size_local_ds,
            Cds.batch_size,
            Cds.size_testset,
            Cds.type_partition
        )
    else:
        # do not consider attacker
        # size_local_ds = Cds.compute_local_training_set_size(Ctop.nu - 1)
        size_local_ds = 1250
        # NOTE: add cover set
        train_sets, test_set, cover_set, x_shape, num_class = setup_data(
            Cds.load_dataset,
            Ctop.nu,
            size_local_ds,
            Cds.batch_size,
            Cds.size_testset,
            Cds.type_partition,
            Ctop.num_member, Ctop.num_non_member, Ctop.num_cover,
            Ctop.setting
        )
    print(f'user dataset size: {size_local_ds}')
    
    # setup model generator function
    make_model = setup_model(
        Cds.model_maker,
        [x_shape, num_class, Ctop.init_lr, Ctop.lrd, Ctop.pretrain, Ctop.checkpoint_path],
        Cds.model_same_init #! model_same_init: True --> 所有用户的模型初始化相同
    ) #! setup_model --> DL_attacks.utils.setup_model --> 初始化一个模型生成函数
    
    # define comm. topology
    DL = Ctop.CDL(Ctop.graph_properties) #! CDL: DecentralizedLearning --> DL_attacks.DL.DecentralizedLearning --> 初始化一个DecentralizedLearning对象
    if Ctop.G is None:
        # NOTE: add cover set
        DL.setup(Ctop.nu, make_model, train_sets, test_set, cover_set, Ctop.USER, Ctop.ATTACKER, \
            Ctop.DEVICE, Ctop.normal_train_iter, Ctop.attack_type) #! Attacker从哪来？: __init__.py中的ATTACKER
    else:
        DL.from_nx_graph(Ctop.G, make_model, train_sets, test_set, Ctop.USER, Ctop.ATTACKER, Ctop.DEVICE)

    # it runs and logs metric during the training, including privacy risk
    # logr = Logger(Ctop, DL, output_file) #! Logger --> DL_attacks.logger.Logger --> 初始化一个Logger对象
    acc_logger = acc_logger(Ctop, DL, acc_log_path)
    
    # it implements early stopping
    # es = EarlyStopping(Cds.patience) #! EarlyStopping --> DL_attacks.utils.EarlyStopping --> 初始化一个EarlyStopping对象
    
    ## Main training loop
    print("Training ....")
    for i in tqdm(range(1, Cds.max_num_iter+1)):
        # run a round of DL
        DL(i) #! DL() --> DecentralizedLearning.__call__() --> 一轮迭代过程
        
       # eval models  
        if i % Cds.eval_interval == 0 and i: #! eval_interval: 25 --> 每25个iteration进行一次evaluation
            # logs privacy risk (slow operation)
            DL.attacker.evaluate_attack_result()
            acc_logger(i)

