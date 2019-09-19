'''Train CIFAR10 with PyTorch.'''
from __future__ import print_function

import torch


# /!\ THOSE LINES MAKE THE WHOLE PROCESS DETERMINISTIC!
torch.manual_seed(58)
import numpy as np
np.random.seed(58)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms
import argparse
import copy

from models import *


from torch.utils.data.sampler import SubsetRandomSampler



parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.1, type=float, help='initial learning rate')
parser.add_argument('--model', default='vgg', type=str, help='model type, vgg or resnet')
parser.add_argument('--loss', default='ce', type=str, help='loss type, cross entropy (ce) or mean square error (mse)')
parser.add_argument('--scaling_factor', default=1.0, type=float, help='scaling factor')
parser.add_argument('--widening_factor', default=1, type=int, help='widening factor')
parser.add_argument('--length', default=300, type=int, help='number of epochs')
parser.add_argument('--bs', default=128, type=int, help='batch size at train')
parser.add_argument('--bs_test', default=100, type=int, help='batch size at test')  # (only for super wide models that consume a lot of memory)
parser.add_argument('--gain', default=2.0, type=float, help='multiplicative gain at initiallization')
parser.add_argument('--schedule', default='a', type=str, help='schedule type, a (wide), b (std)')
parser.add_argument('--subset', default=-1, type=int, help='subset of data, -1: full data') # not used in this paper
parser.add_argument('--precision', default='float', type=str, help='precision, float or double')
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
if args.schedule == 'b':
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.bs, shuffle=True, num_workers=2)

if args.subset>0:
    transform_train = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.bs, num_workers=2,sampler=SubsetRandomSampler(range(args.subset)))



testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=args.bs_test, shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')




k=args.widening_factor
# Model
print('==> Building model..')
net = None
if args.model=='vgg':
    net = VGG('VGG11',k)
elif args.model=='resnet':
    net = ResNet18(k)
net = net.cuda()

model_parameters = filter(lambda p: p.requires_grad, net.parameters())
params = sum([torch.numel(p) for p in model_parameters])

alpha = args.scaling_factor

import datetime
from random import randint

time_stamp = str(datetime.datetime.now().isoformat())
name_log_txt = time_stamp + str(randint(0, 1000)) + '_lr_'+str(args.lr)+'_alpha_'+str(args.scaling_factor)+'_wideningfactor_'+str(args.widening_factor)
name_log_txt+='_model'+str(args.model)+'_loss_'+str(args.loss)+'_gain_'+str(args.gain)
name_log_txt+='.log'


with open(name_log_txt, "a") as text_file:
    print(args)
    print(args,file=text_file)



criterion = None
criterion_train = None
if args.loss=='mse':
    criterion_train =  nn.MSELoss()
    criterion = nn.MSELoss()
elif args.loss=='ce':
    criterion = nn.CrossEntropyLoss()
    criterion_train = nn.CrossEntropyLoss()

from torch.nn.init import xavier_normal_ as xavier
def weights_init(m):
    if isinstance(m, nn.Conv2d):
        xavier(m.weight.data,gain=args.gain)
        m.bias.data.zero_()

net.apply(weights_init)
FC1 = nn.Linear(512*k, 10).cuda()
FC2 = nn.Linear(512*k, 10).cuda()
xavier(FC1.weight.data, gain=args.gain)
FC1.bias.data.zero_()


net2=copy.deepcopy(net)

# Symmetrize!
FC2.weight.data.copy_(FC1.weight.data)
FC2.bias.data.copy_(FC1.bias.data)



par = list(net.parameters())+list(FC1.parameters())
optimizer = optim.SGD(par, lr=args.lr, momentum=0.9, weight_decay=0)

net_clone = copy.deepcopy(net)
stack_hook=[]


pooling_layer = nn.MaxPool2d(kernel_size=2, stride=2,return_indices=True)
unpooling_layer = nn.MaxUnpool2d(kernel_size=2, stride=2)

def hook_extract_relu(module, input, out):
    global stack_hook
    p = out > 0
    if args.precision=='float':
        p = p.float()
    else:
        p=p.double()
    stack_hook.append(p)

def hook_extract_maxpool(module, inp, outp):
    global stack_hook
    inp = inp[0]

    _,idx=pooling_layer(inp)
    out = unpooling_layer(outp,idx)
    p = out > 0

    if args.precision == 'float':
        p = p.float()
    else:
        p = p.double()
    stack_hook.append(p)

def hook_extract_basicblock(module, inp, outp):
    global stack_hook
    inp = inp[0]
    a = F.relu(module.conv1(inp))
    p = a > 0
    if args.precision == 'float':
        p = p.float()
    else:
        p = p.double()

    q = outp>0
    if args.precision == 'float':
        q = q.float()
    else:
        q = q.double()
    stack_hook.append([p,q])

for i in range(len(net_clone.features)):
    if net_clone.features[i].__class__.__name__=='ReLU':
        net_clone.features[i].register_forward_hook(hook_extract_relu)
    elif net_clone.features[i].__class__.__name__ == 'MaxPool2d':
        net_clone.features[i].register_forward_hook(hook_extract_maxpool)
    elif net.features[i].__class__.__name__ == 'BasicBlock':
        net_clone.features[i].register_forward_hook(hook_extract_basicblock)

def lazy_net(x):
    global stack_hook
    z = x.clone()
    stack_hook = []
    net_clone(x)

    j = 0
    for i in range(len(net.features)):
        if net.features[i].__class__.__name__ == 'ReLU':
            p = stack_hook[j]
            z_ = torch.mul(z , p)
            z = z_
            j = j + 1
        elif net.features[i].__class__.__name__ == 'MaxPool2d':
            p = stack_hook[j]
            z = z * p
            z,_ =pooling_layer(z)
            j = j + 1
        elif net.features[i].__class__.__name__ == 'BasicBlock':
            p,q = stack_hook[j]

            z_ = net.features[i].conv1(z)
            z_ = z_*p
            z_ = net.features[i].conv2(z_)

            z = net.features[i].shortcut(z)+z_
            z = z*q

            j = j + 1
        else:
            z = net.features[i](z)
    z = z.view(z.size(0), -1)
    return z

stack_hook = []
x=torch.randn(1,3,32,32).cuda()
if args.precision=='double':
    x=x.double()
net_clone(x)
proportion_lazy = [0] * len(stack_hook)
del x

def net_activation(x):
    global stack_hook
    global proportion_lazy
    z = x.clone()
    stack_hook = []
    net_clone(x)

    j = 0
    for i in range(len(net.features)):
        if net.features[i].__class__.__name__ == 'ReLU':
            p = stack_hook[j]
            z = net.features[i](z)
            p_=z>0
            if args.precision == 'float':
                p_ = p_.float()
            else:
                p_ = p_.double()
            proportion_lazy[j]+= float(torch.sum(p_ == p)) / float(p.numel())
            j = j + 1
        elif net.features[i].__class__.__name__ == 'MaxPool2d':
            p = stack_hook[j]
            z_ = z * p
            z_, _ = pooling_layer(z_)
            z = net.features[i](z)
            proportion_lazy[j]+=float(torch.sum(z == z_)) / float(z.numel())
            j = j + 1
        elif net.features[i].__class__.__name__ == 'BasicBlock':
            p, q = stack_hook[j]
            z = net.features[i](z)
            q_ = z>0
            if args.precision == 'float':
                q_ = q_.float()
            else:
                q_ = q_.double()
            proportion_lazy[j] += float(torch.sum(q_ == q)) / float(q.numel())
            j = j + 1
        else:
            z = net.features[i](z)
    z = z.view(z.size(0), -1)
    return z



# Training
def train(epoch):
    print('\nEpoch: %d' % epoch)
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        if args.precision == 'double':
            inputs = inputs.double()
        optimizer.zero_grad()
        outputs_ = net(inputs)
        outputs2_ = net2(inputs)
        outputs = FC1(outputs_)-FC2(outputs2_)
        loss = None
        if args.loss == 'ce':
            loss = criterion_train(alpha*outputs, targets)/alpha**2
        elif args.loss== 'mse':
            targets_=targets.unsqueeze(1)
            targets_embed=torch.zeros(targets_.size(0),10).cuda()
            targets_embed.scatter_(1, targets_, 1)
            loss = criterion_train(outputs, targets_embed/alpha)

        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
    return train_loss/(1+len(trainloader)),100.*correct/total

def test_lazy():
    global best_acc
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            if args.precision == 'double':
                inputs = inputs.double()
            outputs_ = lazy_net(inputs)
            net_activation(inputs)
            outputs2_ = net2(inputs)
            outputs = FC1(outputs_) - FC2(outputs2_)
            loss = 0
            if args.loss == 'ce':
                loss = criterion_train(alpha * outputs, targets) / alpha ** 2
            elif args.loss == 'mse':
                targets_ = targets.unsqueeze(1)
                targets_embed = torch.zeros(targets_.size(0), 10).cuda()
                targets_embed.scatter_(1, targets_, 1)
                loss = criterion_train(outputs, targets_embed / alpha)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    acc = 100.*correct/total
    return test_loss / (1 + len(testloader)), acc


def test():
    global best_acc
    test_loss = 0
    test_loss_scaled = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device).float(), targets.to(device)
            if args.precision=='double':
                inputs=inputs.double()
            outputs_ = net(inputs)
            outputs2_ = net2(inputs)
            outputs = FC1(outputs_) - FC2(outputs2_)

            loss = 0
            loss_scaled = 0
            if args.loss == 'ce':
                loss = criterion(outputs, targets)
                loss_scaled = criterion(alpha * outputs, targets) / alpha ** 2
            elif args.loss == 'mse':
                targets_ = targets.unsqueeze(1)
                targets_embed = torch.zeros(targets_.size(0), 10).cuda()
                targets_embed.scatter_(1, targets_, 1)
                loss =criterion(outputs, targets_embed)
                loss_scaled = criterion(outputs, targets_embed / alpha)

            test_loss_scaled += loss_scaled.item()
            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    acc = 100.*correct/total
    return test_loss / (1 + len(testloader)), acc, test_loss_scaled/ (1 + len(testloader))

acc_train, acc_test, acc_test_lazy= 0, 0, 0

for epoch in range(args.length):
    lr = args.lr /(1.0+100.0*epoch/300.0)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    for i in range(len(stack_hook)):
        stack_hook[i]=None

    loss_train, acc_train = train(epoch)
    loss_test, acc_test, loss_scaled = test()

    proportion_lazy = [0] * len(stack_hook)

    loss_test_lazy, acc_test_lazy = test_lazy()
    print(proportion_lazy)
    print(
        "epoch {}, log train loss:{:.5f}, train acc:{}, log test loss:{:.5f}, log test loss scaled:{:.5f} , test acc:{}, log loss lazy: {}, test lazy acc:{};"
        .format(epoch, np.log(loss_train), acc_train, np.log(loss_test), np.log(loss_scaled), acc_test,
                np.log(loss_test_lazy), acc_test_lazy))
    with open(name_log_txt, "a") as text_file:
        print("epoch {}, log train loss:{:.5f}, train acc:{}, log test loss:{:.5f}, log test loss scaled:{:.5f} , test acc:{}, log loss lazy: {}, test lazy acc:{};"
              .format(epoch, np.log(loss_train), acc_train, np.log(loss_test), np.log(loss_scaled), acc_test, np.log(loss_test_lazy), acc_test_lazy), file=text_file)
        print(proportion_lazy, file=text_file)

with open("summary.log", "a") as text_file:
    print("alpha {} ; lr: {} ; train acc: {} ; test acc: {} ; test lazy : {} ; loss-type {}".format(args.scaling_factor, args.lr, acc_train, acc_test, acc_test_lazy, args.loss), file=text_file)

