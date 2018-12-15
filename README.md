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

* LeNet5 on MNIST
* ResNet32 on CIFAR10
* ResNet110 on CIFAR100



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