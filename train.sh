# complex 40 nodes
python run_experiment.py exp_setups.CIFAR10 exp_setups.complex40 0
# torus 36 nodes
python run_experiment.py exp_setups.CIFAR10 exp_setups.torus36 0
# complex 6 nodes
python run_experiment.py exp_setups.CIFAR10 exp_setups.complex6 0

python run_experiment.py exp_setups.Location30 exp_setups.complex6 0