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
    size_local_ds = Cds.compute_local_training_set_size(Ctop.nu)
    # loads and splits local training sets and test one (validation)
    if Ctop.attack_type == 'None':
        # gets users' local training size
        train_sets, test_set, cover_set, x_shape, num_class = setup_data_without_attack(
            Cds.load_dataset,
            Ctop.nu,
            size_local_ds,
            Cds.batch_size,
            Cds.size_testset,
            Cds.type_partition
        )
    else:
        # NOTE: add cover set
        train_sets, test_set, cover_set, x_shape, num_class = setup_data(
            Cds.load_dataset,
            Ctop.nu,
            size_local_ds,
            Cds.batch_size,
            Cds.size_testset,
            Cds.type_partition,
            Ctop.num_member, Ctop.num_non_member, size_local_ds,
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
        DL.setup(Ctop.nu, make_model, train_sets, test_set, cover_set, Ctop.USER, Ctop.ATTACKER, \
            Ctop.DEVICE, Ctop.normal_train_iter, Ctop.attack_type, Ctop.defense_type, Ctop.cover_try_time, Ctop.victim_ratio, Ctop.batch_size) #! Attacker从哪来？: __init__.py中的ATTACKER
    else:
        DL.from_nx_graph(Ctop.G, make_model, train_sets, test_set, Ctop.USER, Ctop.ATTACKER, Ctop.DEVICE)

    # it runs and logs metric during the training, including privacy risk
    acc_logger = acc_logger(Ctop, DL, acc_log_path)
    # log setting 
    acc_logger.logger.info('************* DL Setting *************')
    acc_logger.logger.info(f'   Setting = {Ctop.setting}')
    acc_logger.logger.info(f'   Graph = {Ctop.name}')
    acc_logger.logger.info(f'   Number of Attacker = {1}')
    acc_logger.logger.info(f'   Sample per user = {size_local_ds}')
    acc_logger.logger.info(f'   Normal User Batch size = {Cds.batch_size}')
    acc_logger.logger.info(f'   Attacker Batch size = {Ctop.batch_size}')
    acc_logger.logger.info(f'   Attack = {Ctop.attack_type}')
    acc_logger.logger.info(f'   Defense = {Ctop.defense_type}')
    acc_logger.logger.info(f'   Distribution = {Ctop.iid}')
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
