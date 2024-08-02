from . import *

dsk = 'purchase100'
ds_size = 124992 # total 149985 train 2505 test 2505

# model's arch
model_maker = model.purchase100

user_batch_size = 128
    
# size of the local training set of each user
def compute_local_training_set_size(nu):
    # all dataset uniform partition
    return ds_size // (nu)

# size of the global test set used to evaluate all the nodes (meta)
size_testset = 124992

load_dataset = utils.load_purchase100

# victim and cover set
attacker_batch_size = 256

# learning-rate scheduler steps to reach consensus (it may vary based on the topology)  
lrd = [800]
    
# maximum number of training iterations 
normal_train_iter = 200