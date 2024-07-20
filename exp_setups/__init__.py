import numpy as np
from DL_attacks import model, user, attacker, DL, attacker_agrevader, utils #TODO: add attacker_agrevader(FINISHED)
import random
import torch

# where to save logs
melitious_rate = 2.5
num_attack_user = 1 # 2
is_neigh = True

setting = 's1'
'''
    s1	1 neighbour target 
    s2	1 non-neighbour target
    s3	2 neighbour targets
    s4	2 non-neighbour targets
    s5	1 neigh 1 non-neigh targets
    s6	FL
'''
attack_type = 'unitnorm' # norm, unitnorm, angle
defense_type = 'trim1' # none, trimX, median

output_dir = './results-agrevader' 
# log_name = f'_MelitousRate{melitious_rate}_numAttack{num_attack_user}_isNeigh{is_neigh}.txt'
log_name = f'_xnumAttack={num_attack_user}_{setting}_{attack_type}_{defense_type}.txt'

# Graph topology
CDL = DL.DecentralizedLearning
USER = user.User
# ATTACKER = attacker.Attacker #TODO: add attacker_agrevader
ATTACKER = attacker_agrevader.Agrevader_v2
G = None

DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'
print("DEVICE: ", DEVICE)

# learning-rate scheduler steps to reach consensus (it may vary based on the topology)  
lrd = [200, 300, 400]
    
# maximum number of training iterations 
max_num_iter = 1000
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