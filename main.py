import os
import torch
import torch.utils.data
from torch import optim
from torch.autograd import Variable
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
from vcca import VCCA
from utils import *

CUDA = False
SEED = 1
BATCH_SIZE = 128
LOG_INTERVAL = 10
EPOCHS = 50
ZDIMS = 20
PDIMS = 30
PRIVATE = True
# I do this so that the MNIST dataset is downloaded where I want it
os.chdir("/Users/edison/PycharmProjects/vcca_pytorch")

torch.manual_seed(SEED)

if CUDA:
    torch.cuda.manual_seed(SEED)

# DataLoader instances will load tensors directly into GPU memory
kwargs = {'num_workers': 1, 'pin_memory': True} if CUDA else {}

# Download or load downloaded MNIST dataset
# shuffle data at every epoch

data1 = load_data('noisymnist_view1.gz')
data2 = load_data('noisymnist_view2.gz')

train_set_x1, _ = data1[0]
train_set_x2, _ = data2[0]

train_loader = torch.utils.data.DataLoader(
             ConcatDataset(
                 train_set_x1,
                 train_set_x2
             ),
             batch_size=BATCH_SIZE, shuffle=True)

# Same for test data
#test_loader = torch.utils.data.DataLoader(
#    datasets.MNIST('data', train=False, transform=transforms.ToTensor()),
#    batch_size=BATCH_SIZE, shuffle=True, **kwargs)

model = VCCA(PRIVATE)

if CUDA:
    model.cuda()


def loss_function(recon_x1, recon_x2, x1, x2, mu, logvar) -> Variable:
    # how well do input x and output recon_x agree?
    BCE1 = F.binary_cross_entropy(recon_x1, x1.view(-1, 784))
    BCE2 = F.binary_cross_entropy(recon_x2, x2.view(-1, 784))

    # KLD is Kullback–Leibler divergence -- how much does one learned
    # distribution deviate from another, in this specific case the
    # learned distribution from the unit Gaussian

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # - D_{KL} = 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    # note the negative D_{KL} in appendix B of the paper
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    # Normalise by same number of elements as in reconstruction
    KLD /= BATCH_SIZE * 784

    # BCE tries to make our reconstruction as accurate as possible
    # KLD tries to push the distributions as close as possible to unit Gaussian
    return BCE1 + KLD + BCE2

def loss_function_private(recon_x1, recon_x2, x1, x2, mu, logvar, mu1, logvar1, mu2, logvar2) -> Variable:
    # how well do input x and output recon_x agree?
    BCE1 = F.binary_cross_entropy(recon_x1, x1.view(-1, 784))
    BCE2 = F.binary_cross_entropy(recon_x2, x2.view(-1, 784))

    # KLD is Kullback–Leibler divergence -- how much does one learned
    # distribution deviate from another, in this specific case the
    # learned distribution from the unit Gaussian

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # - D_{KL} = 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    # note the negative D_{KL} in appendix B of the paper
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    # Normalise by same number of elements as in reconstruction
    KLD /= BATCH_SIZE * 784

    KLD1 = -0.5 * torch.sum(1 + logvar1 - mu1.pow(2) - logvar1.exp())
    # Normalise by same number of elements as in reconstruction
    KLD1 /= BATCH_SIZE * 784

    KLD2 = -0.5 * torch.sum(1 + logvar2 - mu2.pow(2) - logvar2.exp())
    # Normalise by same number of elements as in reconstruction
    KLD2 /= BATCH_SIZE * 784

    # BCE tries to make our reconstruction as accurate as possible
    # KLD tries to push the distributions as close as possible to unit Gaussian
    return BCE1 + KLD + KLD1 + KLD2 + BCE2


# Dr Diederik Kingma: as if VAEs weren't enough, he also gave us Adam!
optimizer = optim.Adam(model.parameters(), lr=0.0001)


def train(epoch):
    # toggle model to train mode
    model.train()
    train_loss = 0
    # in the case of MNIST, len(train_loader.dataset) is 60000
    # each `data` is of BATCH_SIZE samples and has shape [128, 1, 28, 28]
    for batch_idx, (data1, data2) in enumerate(train_loader):
        data1 = Variable(data1).float()
        data2 = Variable(data2).float()
        if CUDA:
            data1 = data1.cuda()
            data2 = data2.cuda()
        optimizer.zero_grad()

        if not model.private:
            # push whole batch of data through VAE.forward() to get recon_loss
            recon_batch1, recon_batch2, mu, log_var = model(data1, data2)
            # calculate scalar loss
            loss = loss_function(recon_batch1, recon_batch2, data1, data2, mu, log_var)
        else:
            recon_batch1, recon_batch2, mu, log_var, mu1, log_var1, mu2, log_var2 = model(data1, data2)
            loss = loss_function_private(recon_batch1, recon_batch2, data1, data2, mu, log_var, mu1, log_var1, mu2, log_var2)
        # calculate the gradient of the loss w.r.t. the graph leaves
        # i.e. input variables -- by the power of pytorch!
        loss.backward()
        train_loss += loss.data[0]
        optimizer.step()
        if batch_idx % LOG_INTERVAL == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data1), len(train_loader.dataset),
                100. * batch_idx / len(train_loader),
                loss.data[0] / len(data1)))

    print('====> Epoch: {} Average loss: {:.4f}'.format(
          epoch, train_loss / len(train_loader.dataset)))



for epoch in range(1, EPOCHS + 1):
    train(epoch)
    #est(epoch)
    model.eval()
    # 64 sets of random ZDIMS-float vectors, i.e. 64 locations / MNIST
    # digits in latent space
    if model.private:
        sample = Variable(torch.randn(64, PDIMS+ZDIMS))
    else:
        sample = Variable(torch.randn(64, ZDIMS))

    if CUDA:
        sample = sample.cuda()
    sample1 = model.decode_1(sample).cpu()
    sample2 = model.decode_2(sample).cpu()
    # save out as an 8x8 matrix of MNIST digits
    # this will give you a visual idea of how well latent space can generate things
    # that look like digits
    if epoch % 5 == 0:
        save_image(sample1.data.view(64, 1, 28, 28),
                   'results/sample1_' + str(epoch) + '.png')
        save_image(sample2.data.view(64, 1, 28, 28),
                   'results/sample2_' + str(epoch) + '.png')

with open('model.pt','wb') as f:
    torch.save(model, f)