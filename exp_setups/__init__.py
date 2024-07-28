import numpy as np
from DL_attacks import model, user, attacker, DL, attacker_agrevader, utils 
import random
import torch
import datetime

# seed
seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

# how to create local training sets [0: uniform]
type_partition = 0

# where to save logs
melitious_rate = 2.5
num_attack_user = 1 # 2

setting = 's6'
'''
    s1	1 neighbour target 
    s2	1 non-neighbour target
    s3	2 neighbour targets
    s4	2 non-neighbour targets
    s5	1 neigh 1 non-neigh targets
    s6	FL
'''
attack_type = 'angle' # norm, unitnorm, angle, None
defense_type = 'None' # None, trimX, median
iid = 'iid' if type_partition == 0 else 'non-iid'

output_dir = './results-agrevader' 
# timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
timestamp = 'test'
acc_log_name = f'_xnumAttack={num_attack_user}_{setting}_{iid}_{attack_type}_{defense_type}_{timestamp}.txt'

# Graph topology
CDL = DL.DecentralizedLearning
USER = user.User
ATTACKER = attacker.Attacker if attack_type == 'None' else attacker_agrevader.Agrevader_v2
G = None

# pretrain
pretrain = False
checkpoint_path = r'./checkpoint/test_acc=88.23.pth'

# victim and cover set
num_member = 500
num_non_member = 500
num_cover = 1250

DEVICE = 'cuda:1' if torch.cuda.is_available() else 'cpu'
print("DEVICE:", DEVICE)

# learning-rate scheduler steps to reach consensus (it may vary based on the topology)  
lrd = [300, 400, 500]
    
# maximum number of training iterations 
max_num_iter = 1000
normal_train_iter = 300 if attack_type != 'None' else max_num_iter
# attacker node
ATTACKER_ID = 0
# additional conf for topology
graph_properties = {}
# is it an active attack? #! agrevader attack is active
active = False

# initial learning rate
init_lr = .1

# patience early stopping
patience = 10
# when to run MIAs
eval_interval = 25
# is it federated learning?
federated = False
        

## Obsolete #############################
# nodes starts with the same parameters 
model_same_init = True
