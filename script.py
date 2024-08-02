import os


dataset = ['Purchase100', 'Location30']   # 'CIFAR10', 'Purchase100', 'Location30'

seed = [42, 3407]

lr = [0.1]

byz_type = ['trim', 'median']  # 'trim', 'median'
attack_type = ['norm', 'unitnorm', 'angle']

graph = ['regular30_15', 'er30', 'smallworld30']

num_member = [200, 300, 500]
r = [0.7, 0.8, 0.9]

gpu = [0]

distirbution = ['iid', 'non-iid']

output_dir = './new_output'


for each_seed in seed:
    for each_dataset in dataset:
        for each_graph in graph:
            for each_data_distribution in distirbution:
                for each_num_member in num_member:
                    for victim_ratio in r:
                        for each_lr in lr:
                            for each_byz_type in byz_type:
                                for each_attack_type in attack_type:
                                    suffix = "python run_experiment.py" \
                                        + " --data=exp_setups." + str(each_dataset)  \
                                        + " --graph=exp_setups." + str(each_graph) \
                                        + " --seed=" + str(each_seed) \
                                        + " --defense=" + str(each_byz_type) \
                                        + " --attack=" + str(each_attack_type) \
                                        + " --member=" + str(each_num_member) \
                                        + " --victim_ratio=" + str(victim_ratio) \
                                        + " --dist=" + str(each_data_distribution) \
                                        + " --device=" + str(gpu[0]) \
                                        + " --output_dir=" + str(output_dir)
                                    os.system(suffix)

