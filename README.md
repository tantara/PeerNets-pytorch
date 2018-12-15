# PeerNets on PyTorch 1.0

A pytorch reimplementation of [PeerNets: Exploiting Peer Wisdom Against Adversarial Attacks](https://arxiv.org/abs/1806.00088)

**Keywords**

Adversarial Attack, Adversarial Example, Graph Convolutional Network(GCN)

##### Tutorails

* Adversarial Attack(Part 1): Adversarial Attack과 Defense [Link](https://medium.com/@tantara/adversarial-attack-part-1-a830ec92acde)
* Adversarial Attack(Part 2): PeerNets [Link](https://medium.com/@tantara/adversarial-attack-part-2-peernets-fd5ff62818a1)

## Environement

* Docker 18.06 on Ubuntu 16.04
* PyTorch 1.0



## Experiments

- [x] LeNet5 on MNIST
- [x] ResNet32 on CIFAR10
- [x] ResNet110 on CIFAR100
- [ ] PR-LeNet5 on MNIST
- [ ] PR-ResNet32 on CIFAR10
- [ ] PR-ResNet110 on CIFAR100

### Results

#### MNIST

| Method           | baseline | 0.2      | 0.4       | 0.6       | 0.8       | 1.0       |
| ---------------- | -------- | -------- | --------- | --------- | --------- | --------- |
| LeNet-5          | 98.6     | 92.7/7.1 | 33.9/66.0 | 14.1/85.9 | 7.9/92.2  | 8.2/91.7  |
| PR-LeNet-5       | 98.2     | 94.8/4.6 | 93.3/6.0  | 87.7/11.7 | 53.2/46.4 | 50.1/50.1 |
| LeNet-5(ours)    | 98.5     | 95.3/4.0 | 71.5/32.0 | 61.5/46.0 | 51.9/56.0 | 65.4/40.0 |
| PR-LeNet-5(ours) | -        | -        | -         | -         | -         | -         |


#### CIFAR10

| Method         | baseline | 0.04      |
| -------------- | -------- | --------- |
| ResNet-32       | 92.7     | 55.3/44.4 |
| PR-ResNet-32    | 89.3     | 87.3/7.1  |
| ResNet-32(ours) | 93.3     | -         |

#### CIFAR100

| Method          | baseline | 0.02      |
| --------------- | -------- | --------- |
| ResNet-110       | 71.6     | 45.5/49.8 |
| PR-ResNet-110    | 66.4     | 61.5/23.7 |
| ResNet-110(ours) | 68.8     | -         |


### Evaluation

```bash
# bash eval.sh GPU_ID EXP
bash eval.sh 0 lenet5 # lenet5, pr-lenet5
bash eval.sh 0 resnet32 # resnet32, pr-resnet32
bash eval.sh 0 resnet110 # resnet110, pr-resnet110
```



### Adversarial Attack

```bash
# bash attack.sh GPU_ID EXP NUM_SAMPLES RHO
bash attack.sh 0 lenet5 500 0.2 # lenet5, pr-lenet5
bash attack.sh 0 resnet32 500 0.04 # resnet32, pr-resnet32
bash attack.sh 0 resnet110 500 0.02 # resnet110, pr-resnet110
```



### Training

```bash
# bash attack.sh GPU_ID EXP
bash train.sh 0 lenet5 # lenet5, pr-lenet5
bash train.sh 0 resnet32 # resnet32, pr-resnet32
bash train.sh 0 resnet110 # resnet110, pr-resnet110
```



## References

1. **PeerNets: Exploiting Peer Wisdom Against Adversarial Attacks**<br>

   Jan Svoboda, Jonathan Masci, Federico Monti, Michael M. Bronstein, Leonidas Guibas. arXiv: 1806.00088.<br>

   [[link]](https://arxiv.org/abs/1806.00088). arXiv: 1806.00088, 2018.


## Authors

- [Taekmin Kim](https://www.linkedin.com/in/taekminkim/) [@tantara](https://www.linkedin.com/in/taekminkim/)



## Acknowledgement

This work was partially supported by [PyTorch KR](https://github.com/PyTorchKR).



## License

© [Taekmin Kim](https://www.linkedin.com/in/taekminkim/), 2018. Licensed under the [MIT](LICENSE) License.
