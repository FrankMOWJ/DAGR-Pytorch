# Readme

## How to run

parameters need to change

- -a: attack type ['norm', 'unitnorm', 'angle']
- -d: defense type ['trim', median]
- --data: dataset setup file 
- -g: graph setup file
- --device: gpu idx
- --member: number of member = number of non-member
- --iid: if used means iid setting if not non-iid setting



example

```shell
python run_experiment.py --data exp_setups.CIFAR10 --g exp_setups.complex40 -a norm -d trim --device 0 --iid
```

