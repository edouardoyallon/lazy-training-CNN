'''Train CIFAR10 with PyTorch.'''
from __future__ import print_function

import torch
torch.manual_seed(58)
import numpy as np
np.random.seed(58)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

import copy
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import argparse
from models import *
from lazy_utils import progress_bar

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--model', default='vgg', type=str, help='model type')
parser.add_argument('--widening_factor', default=1, type=int, help='widening factor')
parser.add_argument('--bs', default=10, type=int, help='batch size')
parser.add_argument('--gain', default=2.0, type=float, help='gain at init')
parser.add_argument('--subset', default=500, type=int, help='subset of data')
parser.add_argument('--precision', default='float', type=str, help='precision...')
parser.add_argument('--data', default='cifar10', type=str, help='which dataset?...')


args = parser.parse_args()

if args.precision=='float':
    torch.set_default_dtype(torch.float32)
elif args.precision=='double':
    torch.set_default_dtype(torch.float64)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# Data
print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
if args.data == 'random':
    trainset.train_data=np.random.randint(256,size=(500000, 32,32,3),dtype=np.uint8) # be careful, there was a recent
    # modification of torch, you might have to switch 'train_data' to 'data'
    print('randomized')
trainset = torch.utils.data.Subset(trainset,range(args.subset))

trainloader = torch.utils.data.DataLoader(trainset, shuffle=False, batch_size=args.bs, num_workers=2)
trainloader2 = torch.utils.data.DataLoader(trainset,shuffle=False, batch_size=args.bs, num_workers=2)



k=args.widening_factor
# Model
print('==> Building model..')
net = None
if args.model=='vgg':
    net = VGG('VGG11',k)
elif args.model=='resnet':
    net = ResNet18(k)
net = net.to(device)
net = nn.DataParallel(net.to(device))


from torch.nn.init import xavier_normal_ as xavier
def weights_init(m):
    if isinstance(m, nn.Conv2d):
        xavier(m.weight.data,gain=args.gain)
        m.bias.data.zero_()

net.apply(weights_init)

net2=copy.deepcopy(net)

FC1 = nn.Linear(512*k, 10).cuda()
FC2 = nn.Linear(512*k, 10).cuda()

xavier(FC1.weight.data, gain=args.gain)


FC1.bias.data.zero_()
FC2.weight.data.copy_(FC1.weight.data)
FC2.bias.data.copy_(FC1.bias.data)



def linearized_outputs(inputs):
    net_parameters = list(net.parameters())+list(FC1.parameters())
    params = sum([torch.numel(p) for p in net_parameters])

    output_linearized=torch.zeros(inputs.size(0),10,params).cuda()
    output1 = net(inputs)
    output2 = net2(inputs)
    output = FC1(output1)-FC2(output2)
    for n in range(inputs.size(0)):
        for i in range(10):
            output[n, i].backward(retain_graph=True)
            p_idx=0
            for p in range(len(net_parameters)):
                output_linearized[n, i,p_idx:p_idx+net_parameters[p].numel()] = net_parameters[p].grad.data.view(-1)
                p_idx = p_idx + net_parameters[p].numel()
            for p in range(len(net_parameters)):
                net_parameters[p].grad.data.zero_()

    output_linearized = output_linearized.view(inputs.size(0)*10,params)
    return output_linearized



def extract_features(epoch):
    print('\nEpoch: %d' % epoch)
    K = torch.zeros([args.subset*10,args.subset*10],dtype=torch.float64)
    idx = 0
    idx2 = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        if args.precision == 'double':
            inputs = inputs.double()
        out = linearized_outputs(inputs)

        progress_bar(batch_idx, len(trainloader), 'bar 1')
        for batch_idx2, (inputs2, targets2) in enumerate(trainloader2):
            if(batch_idx2<batch_idx):
                idx2 = idx2 + inputs2.size(0)*10
                continue
            inputs2, targets2 = inputs2.to(device), targets2.to(device)
            if args.precision == 'double':
                inputs2 = inputs2.double()
            out2 = linearized_outputs(inputs2)

            progress_bar(batch_idx2, len(trainloader2), 'bar 2')
            K_sub = torch.mm(out.view(out.size(0),-1), out2.view(out.size(0),-1).t())

            K[idx:idx+10*inputs.size(0),idx2:idx2+10*inputs2.size(0)] =K_sub
            K[idx2:idx2 + 10*inputs2.size(0),idx:idx +10*inputs.size(0)] = K_sub.t()
            idx2 = idx2 + 10*inputs2.size(0)
        idx2 =0
        idx = idx+inputs.size(0)*10
    return K

K = extract_features(0)
if args.data == 'random':
    torch.save(K, 'kernel_data_.t7')
    print('saved random features')
else:
    torch.save(K,'kernel_cifar_.t7')
    print('saved cifar features')

