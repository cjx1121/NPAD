
## Step 1. Train a backdoored model

- To train a BadNets ResNet-18, run,
```
python train_backdoor_cifar.py --poison-type badnets --poison-rate 0.05 --poison-target 0 --output-dir './save'
```


- To train a Blend ResNet-18, run,
```
python train_backdoor_cifar.py --poison-type blend --poison-rate 0.05 --trigger-alpha 0.2 --poison-target 0 --output-dir './save'
```


## Step 2. Optimize the masks of all neurons

- We optimize the masks on 1% of CIFAR-10 training data (0.01). We set eps=0.4 and alpha=0.2 in ANP by default. Then run,

```
python optimize_mask_cifar.py --val-frac 0.01 --anp-eps 0.4 --anp-alpha 0.2 --checkpoints './save/last_model.th' --trigger-info' './save/trigger_info.th' --output-dir './save'
```


## Step 3. Prune neurons

- Neurons are pruned based on their mask values. We stop pruning until reaching a predefined threshold, 

```
python prune_neuron_cifar.py --pruning-by threshold --pruning-step 0.05 --pruning-max 0.95 --mask-file './save/mask_values.txt' --checkpoints './save/last_model.th' --trigger-info' './save/trigger_info.th' --output-dir './save'
```