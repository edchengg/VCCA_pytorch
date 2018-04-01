# VCCA: Variational Canonical Correlation Analysis

This is an Pytorch implementation of [Deep Variational Canonical Correlation Analysis (VCCA)](https://arxiv.org/abs/1610.03454) in Python.

## Variational CCA and Variational CCA Private [VCCA, VCCAP]

<img src="https://github.com/edchengg/VCCA-StudyNotes/blob/master/Notes/vcca.png" width="300">

<img src="https://github.com/edchengg/VCCA-StudyNotes/blob/master/Notes/vcca-p.png" width="300">


[Deep Variational Canonical Correlation Analysis](https://github.com/edchengg/VCCA-StudyNotes/blob/master/paper/DVCCA.pdf)

<img src="https://github.com/edchengg/VCCA-StudyNotes/blob/master/Notes/vccapg.png" width="300">

[Acoustic Feature Learning via Deep Variational Canonical Correlation Analysis](https://github.com/edchengg/VCCA-StudyNotes/blob/master/paper/DVCCA_app.pdf)

### Training
5 epochs:
<img src="https://github.com/edchengg/VCCA_pytorch/blob/master/results/sample1_5.png" width="300">
<img src="https://github.com/edchengg/VCCA_pytorch/blob/master/results/sample2_5.png" width="300">

50 epochs:
<img src="https://github.com/edchengg/VCCA_pytorch/blob/master/results/sample1_50.png" width="300">
<img src="https://github.com/edchengg/VCCA_pytorch/blob/master/results/sample2_50.png" width="300">

### Generation

<img src="https://github.com/edchengg/VCCA_pytorch/blob/master/results/final.png" width="300">

### Dataset
The model is evaluated on a noisy version of MNIST dataset. [Vahid Noroozi](https://github.com/VahidooX/DeepCCA) built the dataset exactly like the way it is introduced in the paper. The train/validation/test split is the original split of MNIST.

The dataset was large and could not get uploaded on GitHub. So it is uploaded on another server. You can download the data from:

[view1:](https://www2.cs.uic.edu/~vnoroozi/noisy-mnist/noisymnist_view1.gz)
[view2:](https://www2.cs.uic.edu/~vnoroozi/noisy-mnist/noisymnist_view2.gz)

save it in the same directory with python code.

### Differences with the original paper
The following are the differences between my implementation and the original paper (they are small):

 * I used simple bianry cross entropy loss for two decoder networks.

### Other Implementations

The following are the other implementations of DCCA in Tensorflow.

* [Tensorflow implementation](http://ttic.uchicago.edu/~wwang5/papers/vcca_tf0.9_code.tgz) from Wang, Weiran's website (http://ttic.uchicago.edu/~wwang5/)

