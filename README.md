# Readme

## How to run

parameters need to change

- -a: attack type ['norm', 'unitnorm', 'angle']
- -d: defense type ['trim', median]
- --data: dataset setup file 
- -g: graph setup file
- --device: gpu idx
- --member: number of member = number of non-member
- --dist: iid or non-iid
- --victim_ratio: victim parameters weight when combining victim and cover parameters



example

```shell
python run_experiment.py --data exp_setups.Location30 --g exp_setups.regular30_15 -a norm -d trim --device 0 --dist iid
python run_experiment.py --data exp_setups.Purchase100 --g exp_setups.regular30_15 -a norm -d trim --device 0 --dist iid --member 200
```
