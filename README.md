# <center>Rethinking Reverse-engineering based Backdoor Removal in Self-supervised Learning</center>

## 1. overview

​		This is the repo for our paper *Rethinking Reverse-engineering based Backdoor Removal in Self-supervised Learning*.

​		The repo consists of 3 modules:

- DATASETS: the dataloader module,  STL10, CIFAR10, GTSRB, SVHN are supported
- INVERSION: the iterative trigger inversion module, our reproduction is based on  [IEEE S&P 2022 paper "Model Orthogonalization: Class Distance Hardening in Neural Networks for Better Security."](https://github.com/Gwinhen/MOTH).  Thanks for their amazing implementation.
- Trainer: the flow control module
- MODELS: the module to implement model architecture and early stop mechanism
- The proof details is in the [attached PDF](./proof_details.pdf).

## 2. step by step

### 2.1. iterative trigger inversion and unlearn 

​		The encoder elimination process is composed of 2 stages:

- Stage 1:  iterative trigger inversion and early stop:

  - ```
    python -u MAIN.py --seed 0 --phase moth --dataset cifar10 --model resnet18 --pretrained_encoder <your_encoder>  --batch_size 32 --epochs 20 --harden_lr 1e-3 --re_lr 1e-2  --data_ratio 0.01 --encoder_usage_info cifar10 --log_dir <your_log_directory> --results_dir <your_results_directory> --similarity_threshold  0.90 --gpu 0 --attack_size 100 --re_steps 300 --downstreamTask stl10 --reference_label 9 --init_cost 5e-2 --epsilon 1e-3 --patience 15
    ```

  - The inversion process will be printed on the screen and the re-triggers will be saved into the <your\_log\_directory>

  - ![image-20230523110652070](images%20in%20text/image-20230523110652070.png)

- Stage 2:  unlearn :

  - ```
    python -u Unlearn-trigger.py --seed 0 --results_dir <your_results_dir> --phase moth --dataset stl10 --model resnet18 --pretrained_encoder <your_encoder> --batch_size 128 --epochs 20 --lr 7e-1 --data_ratio 0.01 --encoder_usage_info cifar10 --log_dir <your_log_directory> --reference_label 9 --gpu 0 --downstreamTask stl10 --re_mask_pattern <log_dirtectory>/mask-pattern-best.npz  --portion 0.3 --beta 3.5
    ```

  - The unlearned encoders will be jumped into <your\_results\_dir>

  - ![image-20230523111230495](images%20in%20text/image-20230523111230495.png)



### 2.2. train downstream classifier

​		Based on the encoder unleaned in Stage 2, we can test the ACC and ASR by training a downstream classifier

```python
python -u training_downstream_classifier.py --dataset stl10 --trigger_file <injected_trigger> --encoder <dir>/Harden_Unlearn_model_20.pth --reference_label 9 --gpu 1 --nn_epochs 100 --batch_size 512  --encoder_usage_info cifar10 --results_dir <your_results_dir>
```

​		Then the classifier will be saved in \<your_results_dir\>.

​		ACC and ASR will print on the screen.



