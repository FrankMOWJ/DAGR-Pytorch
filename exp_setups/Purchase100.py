from . import *

dsk = 'purchase100'
ds_size = 124992 # total 149985 train 2505 test 2505

# model's arch
model_maker = model.purchase100

batch_size = 128
    
# size of the local training set of each user
def compute_local_training_set_size(nu):
    # all dataset uniform partition
    return ds_size // (nu)

# size of the global test set used to evaluate all the nodes (meta)
size_testset = 124992

load_dataset = utils.load_purchase100

# victim and cover set
cover_try_time = 5
batch_size = 256
victim_ratio = 0.7
num_member = 500
num_non_member = 500

# learning-rate scheduler steps to reach consensus (it may vary based on the topology)  
lrd = [300, 400, 500, 600]
    
# maximum number of training iterations 
max_num_iter = 800
normal_train_iter = lrd[0] if attack_type != 'None' else max_num_iter