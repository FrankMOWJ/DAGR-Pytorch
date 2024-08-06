from DL_attacks import model, user, attacker, DL, attacker_agrevader, utils 

# how to create local training sets [0: uniform]
type_partition = 0

# where to save logs
melitious_rate = 2.5
num_attack_user = 1 # 2

# setting = 's6'
# '''
#     s1	1 neighbour target 
#     s2	1 non-neighbour target
#     s3	2 neighbour targets
#     s4	2 non-neighbour targets
#     s5	1 neigh 1 non-neigh targets
#     s6	FL
# '''
# attack_type 改成None就是原本是的Attacker
# attack_type = 'angle' # norm, unitnorm, angle, None
# defense_type = 'trim' # None, trimX, median


# Graph topology
CDL = DL.DecentralizedLearning
USER = user.User
# ATTACKER = attacker.Attacker if attack_type == 'None' else attacker_agrevader.Agrevader_v2
ATTACKER = attacker_agrevader.Agrevader_v2
G = None

victim_ratio = 0.7
cover_try_time = 5
max_num_iter = 800
num_member = 500
num_non_member = 500

# pretrain
pretrain = False
checkpoint_path = r'./checkpoint/test_acc=89.64.pth'

# attacker node
ATTACKER_ID = 0
# additional conf for topology
graph_properties = {}
# is it an active attack? #! agrevader attack is active
active = False

# patience early stopping
patience = 10
# when to run MIAs
eval_interval = 25
# is it federated learning?
federated = False
        
# nodes starts with the same parameters 
model_same_init = True