import torch.utils.data
from utils import *
from torchvision import datasets, transforms
from utils import *
from torchvision.utils import save_image
from torch.autograd import Variable
import matplotlib.pyplot as plt

with open('model.pt', 'rb') as f:
    model = torch.load(f)


model.eval()

data1 = load_data('noisymnist_view1.gz')
data2 = load_data('noisymnist_view2.gz')
train_set_x1, label1 = data1[0]
train_set_x2, label2 = data2[0]

train_loader = torch.utils.data.DataLoader(
             ConcatDataset(
                 train_set_x1,
                 train_set_x2
             ),
             batch_size=1, shuffle=False)


print(label1[:10])
print(label2[:10])

for batch_idx, (data1, data2) in enumerate(train_loader):
    data1 = Variable(data1).float()
    data2 = Variable(data2).float()

    model.eval()

    mu_z, _ = model.encode(data1)
    sample1 = data1
    for batch_idx2, (data11, data22) in enumerate(train_loader):
        data11 = Variable(data11).float()

        p_mu, log_var = model.private_encoder1(data11)
        std = log_var.mul(0.5).exp_()  # type: Variable

        eps = Variable(std.data.new(std.size()).normal_())

        input = eps.mul(std).add_(p_mu)


        sample_tmp = torch.cat((mu_z, input),1)
        sample_tmp = model.decode_1(sample_tmp).cpu()

        sample1 = torch.cat((sample1, sample_tmp),1)
        if batch_idx2 == 6:
            break
    if batch_idx == 0:
        res = sample1
    else:
        res = torch.cat((res, sample1),0)

    if batch_idx == 7:
        break
print(res.size())
save_image(res.data.view(64, 1, 28, 28),
           'results/final.png')