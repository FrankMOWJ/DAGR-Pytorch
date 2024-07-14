import numpy as np
from DL_attacks import model, user, attacker, DL, attacker_agrevader, utils #TODO: add attacker_agrevader(FINISHED)
import random
import torch

# where to save logs
output_dir = './results-agrevader' #TODO: change to 'results-agrevader'

# Graph topology
CDL = DL.DecentralizedLearning
USER = user.User
# ATTACKER = attacker.Attacker #TODO: add attacker_agrevader
ATTACKER = attacker_agrevader.Agrevader_v2
G = None

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print("DEVICE: ", DEVICE)

# learning-rate scheduler steps to reach consensus (it may vary based on the topology)  
lrd = [300, 400, 500]
    
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
eval_interval = 50
# is it federated learning?
federated = False
        

## Obsolete #############################
# nodes starts with the same parameters 
model_same_init = True
# how to create local training sets [0: uniform]
type_partition = 0