import os, sys, importlib

from DL_attacks.utils import EarlyStopping, setup_data, setup_model
from DL_attacks.logger import Logger
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

    name = f'{run_num}-{Cds.dsk}-{Ctop.name}' #TODO
    output_file = os.path.join(Cds.output_dir, name)
    print(f"Logging file in --> {output_file}")
    
    # gets users' local training size
    size_local_ds = Cds.compute_local_training_set_size(Ctop.nu)
    
    print("Running setup ....")
    print(f'user dataset size: {size_local_ds}')
    # loads and splits local training sets and test one (validation)
    # NOTE: add cover set
    train_sets, test_set, cover_set, x_shape, num_class = setup_data(
        Cds.load_dataset,
        Ctop.nu,
        size_local_ds,
        Cds.batch_size,
        Cds.size_testset,
        Cds.type_partition
    )

    # setup model generator function
    # NOTE: 这里可能要改
    make_model = setup_model(
        Cds.model_maker,
        [x_shape, num_class, Ctop.init_lr, Ctop.lrd],
        Cds.model_same_init #! model_same_init: True --> 所有用户的模型初始化相同
    ) #! setup_model --> DL_attacks.utils.setup_model --> 初始化一个模型生成函数
    
    # define comm. topology
    DL = Ctop.CDL(Ctop.graph_properties) #! CDL: DecentralizedLearning --> DL_attacks.DL.DecentralizedLearning --> 初始化一个DecentralizedLearning对象
    if Ctop.G is None:
        # NOTE: add cover set
        DL.setup(Ctop.nu, make_model, train_sets, test_set, cover_set, Ctop.USER, Ctop.ATTACKER, Ctop.DEVICE) #! Attacker从哪来？: __init__.py中的ATTACKER
    else:
        DL.from_nx_graph(Ctop.G, make_model, train_sets, test_set, Ctop.USER, Ctop.ATTACKER, Ctop.DEVICE)

    # it runs and logs metric during the training, including privacy risk
    logr = Logger(Ctop, DL, output_file) #! Logger --> DL_attacks.logger.Logger --> 初始化一个Logger对象
    # it implements early stopping
    es = EarlyStopping(Cds.patience) #! EarlyStopping --> DL_attacks.utils.EarlyStopping --> 初始化一个EarlyStopping对象
    
    ## Main training loop
    print("Training ....")
    for i in tqdm(range(1, Cds.max_num_iter+1)):
        # run a round of DL
        DL(i) #! DL() --> DecentralizedLearning.__call__() --> 一轮迭代过程
        
       # eval models  
        if i % Cds.eval_interval == 0 and i: #! eval_interval: 25 --> 每25个iteration进行一次evaluation
            # logs privacy risk (slow operation)
            # print(f'Epoch: {i}')
            score = logr(i) #! logr(i) --> Logger.__call__() --> 计算并记录utility, consensus和privacy
            
            # checks for early stopping
            # if es(i, score): #! es(i, score) --> EarlyStopping.__call__() --> 检查是否需要early stop
            #     print("\tEarly stop!")
            #     break
            
            # save current logs
            logr.dump() #! logr.dump() --> Logger.dump() --> 保存当前的logs
    
    # final evaluation
    logr(i, DL)
    
    # save final logs
    logr.dump()
