from . import *

dsk = 'cifar10'
ds_size = 50000

# model's arch
model_maker = model.resnet20

# batch size for Distributed SGD
batch_size = 64 #TODO: 修改batchsize（64）--> ()
    
# size of the local training set of each user
def compute_local_training_set_size(nu):
    # all dataset uniform partition
    return ds_size // (nu)

# size of the global test set used to evaluate all the nodes (meta)
size_testset = 10000

load_dataset = utils.load_cifar10
    