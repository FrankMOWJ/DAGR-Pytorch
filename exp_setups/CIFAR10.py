from . import *

dsk = 'cifar10'
ds_size = 50000

# model's arch
model_maker = model.resnet20

user_batch_size = 64
    
# size of the local training set of each user
def compute_local_training_set_size(nu):
    # all dataset uniform partition
    return ds_size // (nu)

# size of the global test set used to evaluate all the nodes (meta)
size_testset = 10000

load_dataset = utils.load_cifar10

# victim and cover set
attacker_batch_size = 256
num_member = 500
num_non_member = 500

# learning-rate scheduler steps to reach consensus (it may vary based on the topology)  
lrd = [300, 400, 500, 600]
    
# maximum number of training iterations 
normal_train_iter = lrd[0]
    