import torch.utils.data
from utils import *
from torchvision import datasets, transforms
from utils import *
from torch.autograd import Variable
CUDA = False
SEED = 1
BATCH_SIZE = 128
LOG_INTERVAL = 10
EPOCHS = 10
ZDIMS = 20

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

for batch_idx, (data1, data2) in enumerate(train_loader):
    data1 = Variable(data1)
    data2 = Variable(data2)

    print(data1.shape)