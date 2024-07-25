import numpy as np
from DL_attacks import model, user, attacker, DL, attacker_agrevader, utils 
import random
import torch

# seed
seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

# where to save logs
melitious_rate = 2.5
num_attack_user = 1 # 2

setting = 's7'
'''
    s1	1 neighbour target 
    s2	1 non-neighbour target
    s3	2 neighbour targets
    s4	2 non-neighbour targets
    s5	1 neigh 1 non-neigh targets
    s6	FL
    s7  random 
'''
attack_type = 'angle' # norm, unitnorm, angle
defense_type = 'None' # none, trimX, median

output_dir = './results-agrevader' 
log_name = f'_xnumAttack={num_attack_user}_{setting}_{attack_type}_{defense_type}.txt'

# Graph topology
CDL = DL.DecentralizedLearning
USER = user.User
# ATTACKER = attacker.Attacker #TODO: add attacker_agrevader
ATTACKER = attacker_agrevader.Agrevader_v2
G = None

num_member = 500
num_non_member = 500
num_cover = 1000

DEVICE = 'cuda:2' if torch.cuda.is_available() else 'cpu'
print("DEVICE: ", DEVICE)

# learning-rate scheduler steps to reach consensus (it may vary based on the topology)  
lrd = [400, 500, 600]
    
# maximum number of training iterations 
max_num_iter = 800
normal_train_iter = 50
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
# how to create local training sets [0: uniform]
type_partition = 0