from . import *

dsk = 'location30'
ds_size = 2505 # total 5010 train 2505 test 2505

# model's arch
model_maker = model.location30

user_batch_size = 64
    
# size of the local training set of each user
def compute_local_training_set_size(nu):
    # all dataset uniform partition
    return ds_size // (nu)

# size of the global test set used to evaluate all the nodes (meta)
size_testset = 2505

load_dataset = utils.load_location30

# victim and cover set
attacker_batch_size = 256


# learning-rate scheduler steps to reach consensus (it may vary based on the topology)  
lrd = [800]
    
# maximum number of training iterations 
normal_train_iter = 200