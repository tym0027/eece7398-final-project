'''Train CIFAR10 with PyTorch.'''
'''Modified from  https://github.com/kuangliu/pytorch-cifar'''
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import time
from datetime import datetime
import numpy as np
import random
from random import uniform
import os
import argparse
import skimage.io

from models import *
from utils import progress_bar

os.environ["CUDA_VISIBLE_DEVICES"]='1,2,6,7'

# Random Seed
torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(0)
random.seed(0)


def printActivations(x, orig, subdir):
    print("activations:", x.shape)
    print(orig.shape)
    x = x.cpu().detach().numpy()
    orig = orig.cpu().detach().squeeze().numpy()
    orig = np.transpose(orig, (1,2,0))

    if not os.path.exists("./activations/" + args.save_name):
        os.mkdir("./activations/" + args.save_name)
    if not os.path.exists("./activations/" + subdir):
        os.mkdir("./activations/" + subdir)    

    skimage.io.imsave("./activations/" + subdir + "/original.png", orig)

    big_image = np.zeros((32 * 4, 32 * 8, 3))
    for row in range(0, 4):
        for col in range(0, 8):
            index = row * 8 + col
            img = np.transpose(x[:,index,:,:],(1,2,0))
            img = img.astype(np.float64)
            img = img / img.max()

            img2 = np.zeros_like(orig)
            img2[:,:,0] = img[:,:,0] * 255
            img2[:,:,1] = img[:,:,0] * 255
            img2[:,:,2] = img[:,:,0] * 255

            img2 = img2.astype(np.uint8)

            big_image[row * 32 : (row * 32) + 32, col * 32 : (col * 32) + 32, :] = img2# x_[4 * row + col,:,:, :] # RGB

    big_image = big_image.astype(np.uint8)
    skimage.io.imsave("./activations/" + subdir + "/activations.png" , big_image)
    return


def log(s, timestamp=True):
    if not os.path.exists('logfile/' + args.save_name):
        os.mkdir('logfile/' + args.save_name)
    if timestamp:
        dateTimeObj = datetime.now()
        timestamp = dateTimeObj.strftime("%d-%b-%Y (%H:%M:%S)")
        f = open('logfile/' + args.save_name + '/log.txt', 'a')
        f.write(timestamp + ": " + s.replace("\n","\n" + timestamp + ":") + '\n')
    else:
        f = open('logfile/' + args.save_name + '/log.txt', 'a')
        f.write(s + '\n')

    f.close()


## Function to save the checkpoint
def save_checkpoint(state, filename='checkpoint.pth'):
    torch.save(state, filename)

def load_checkpoint(path, model, optimizer):
    loadedcheckpoint = torch.load(path)

    try:
        model.load_state_dict(loadedcheckpoint['state_dict'])
    except:
        print("no state dict!!!")

    try:
        start_epoch = loadedcheckpoint['epoch']
    except:
        print("no epoch!!!")

    try:
        optimizer.load_state_dict(loadedcheckpoint['optimizer'])
    except:
        print("no optimizer!!!")
    '''
    try:
        scheduler.load_state_dict(loadedcheckpoint['scheduler'])
    except:
        print("No scheduler!!!")
    '''
    return model, optimizer, start_epoch

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--train', action='store_true', help='true for training')
parser.add_argument('--test', action='store_true', help='true for testing')
parser.add_argument('--lr', default=0.001, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument('--path', '-p', default="./checkpoint/ckpt.pth", type=str, help='path to your model checkpoint')
parser.add_argument('--save_name', '-s', default="my_experiment", type=str, help='the name of your experiment')
parser.add_argument('--net', '-n', default="LeNet", type=str, help='the name of your network')
parser.add_argument('--p0', default=0.0, type=float, help='the proportion of dropout ( p0 )')
parser.add_argument('--p1', default=0.0, type=float, help='the proportion of dropout ( p1 )')
parser.add_argument('--p2', default=0.0, type=float, help='the proportion of dropout ( p2 )')
parser.add_argument('--p3', default=0.0, type=float, help='the proportion of dropout ( p3 )')
parser.add_argument('--dataset', default="cifar10", type=str, help='select the dataset you wish to train on')
parser.add_argument('--ICL', action='store_true', help='Would you like to use models that use independent component layers?')
parser.add_argument('--dynamic1', action='store_true', help='change the the independent component layer\'s p values')
parser.add_argument('--dynamic2', action='store_true', help='change the the independent component layer\'s p values')
parser.add_argument('--dynamic3', action='store_true', help='change the the independent component layer\'s p values')
parser.add_argument('--stochastic',action='store_true', help='change the the independent component layer\'s p values')
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# Data
print('==> Preparing data..')
log('==> Preparing data..', True)
transform_train = transforms.Compose([
    transforms.Resize((32,32)),
    transforms.RandomCrop(28, padding=0),
    # transforms.RandomCrop(32, padding=4),
    transforms.Resize((32,32)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.Resize((32,32)),
    transforms.ToTensor(),
    # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])


if args.dataset == "mnist":
    log('==> using MNIST data..', True)
    trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=1)
    
    testset = torchvision.datasets.MNIST('./data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=True, num_workers=1)

    classes = ('1', '2', '3', '4', '5', '6', '7', '8', '9')

elif args.dataset == "fmnist":
    log('==> using Fashion MNIST data..', True)
    trainset = torchvision.datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=1)

    testset = torchvision.datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=True, num_workers=1)

    classes = ('T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot')

elif args.dataset == "cifar10":
    log('==> using CIFAR10 data..', True)

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=1)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=1)

    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


# Model
print('==> Building model..')
# net = VGG('VGG19')
# net = ResNet18()
# net = PreActResNet18()
# net = GoogLeNet()
# net = DenseNet121()
# net = ResNeXt29_2x64d()
# net = MobileNet()
# net = MobileNetV2()
# net = DPN92()
# net = ShuffleNetG2()
# net = SENet18()
# net = ShuffleNetV2(1)
# net = EfficientNetB0()
# net = RegNetX_200MF()

if args.net == "LeNet":
    if args.ICL: 
        log('LeNet selected\n\t(ICL enabled)', True)
        net = ICLeNet(args.p0,args.p1,args.p2, True)
    else:
        log('LeNet selected\n\t(ICL disabled)', True)
        net = ICLeNet(args.p0,args.p1,args.p2, False)
elif args.net == "TimNet":
    if args.ICL:
        log('TimNet selected\n\t(ICL enabled)', True)
        net = ICTimNet(args.p0,args.p1,args.p2,args.p3, True)
    else:
        log('TimNet selected\n\t(ICL disabled)', True)
        net = ICTimNet(args.p0,args.p1,args.p2,args.p3,False)

net = net.to(device)

if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True


criterion = nn.CrossEntropyLoss()
# optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
optimizer = optim.Adam(net.parameters(), lr=args.lr) #, weight_decay=.000001)

if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint... ' + str(args.path))
    log('==> Resuming from checkpoint... ' + str(args.path))
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    net, optimizer, start_epoch = load_checkpoint(args.path, net, optimizer)
    # checkpoint = torch.load('./checkpoint/ckpt.pth')
    # net.load_state_dict(checkpoint['net'])
    # best_acc = checkpoint['acc']
    # start_epoch = checkpoint['epoch']

p0, p1, p2, p3 = args.p0, args.p1, args.p2, args.p3

# Training
def train(epoch):
    global p0, p1, p2, p3
    print('\nEpoch: %d' % epoch)
    net.train()
    # optimizer.zero_grad()
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, len(trainloader))

    ret_loss = 0
    ret_acc = 0
    train_loss = 0
    correct = 0
    total = 0

    for batch_idx, (inputs, targets) in enumerate(trainloader):
        # print("here:", inputs.shape)
        # inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        
        if args.dataset == "mnist" or args.dataset == "fmnist":
            inputs = inputs.repeat(1, 3, 1, 1)


        inputs, targets = inputs.to(device), targets.to(device)

        # Dynamic implmentations of IC Layers
        if args.net == "LeNet":
            outputs, original, activations = net(inputs, p0, p1, p2)
        elif args.net == "TimNet":
            outputs, original, activations = net(inputs, p0, p1, p2, p3)

        loss = criterion(outputs, targets)
        train_loss += loss.item()

        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))

        loss.backward()
        optimizer.step()
        scheduler.step()

        # Dynamic IC Layer Implmenatation
        if args.stochastic:
            if args.net == "LeNet":
                # print("changing net p values from ", p0, p1, p2)
                p0 = uniform(0, .75) 
                p1 = uniform(0, .75)
                p2 = uniform(0, .75)
                # print("to p values ----> ", p0, p1, p2)

            if args.net == "TimNet":
                # print("changing net p values from ", p0, p1, p2, p3)
                p0 = uniform(0, .75)
                p1 = uniform(0, .75)
                p2 = uniform(0, .75)
                p3 = uniform(0, .75)
                # print("to p values ----> ", p0, p1, p2, p3)

    ret_loss = train_loss/(batch_idx+1)
    ret_acc = 100.*correct/total

    return ret_loss, ret_acc

def test(epoch):
    global best_acc
    global p0, p1, p2, p3
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        index = 0
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)

            if args.dataset == "mnist" or args.dataset == "fmnist":
                inputs = inputs.repeat(1, 3, 1, 1)

            if args.net == "LeNet":
                outputs, original, activations = net(inputs, p0, p1, p2)
            elif args.net == "TimNet":
                outputs, original, activations = net(inputs, p0, p1, p2, p3)
            
            '''
            if args.test:
                for inp in range(0, inputs.size(0)):
                    printActivations(activations[inp,:,:,:], original[inp,:,:,:], str(args.save_name) + "/" + str(index).zfill(5))
                    index += 1    
            '''

            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))


    # Save checkpoint.
    acc = 100.*correct/total
    if acc > best_acc:
        print('Saving..')
        '''
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        '''
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        # torch.save(state, './checkpoint/ckpt.pth')
        best_acc = acc

        if not os.path.isdir('checkpoint/' + args.save_name + "/"):
            os.mkdir('checkpoint/' + args.save_name + "/")

        save_checkpoint({'epoch': epoch + 1,
                        'state_dict': net.state_dict(),
                        'optimizer' : optimizer.state_dict(),
                        # 'scheduler' : scheduler.state_dict(),
                        }, "./checkpoint/" + args.save_name + "/" + args.net + "-" + str(int(best_acc)) + "-" + str(epoch)+'.pth')

    return test_loss/(batch_idx+1), 100.*correct/total


if args.train:
    for epoch in range(start_epoch, start_epoch+200):
        train_loss, train_acc = train(epoch)
        test_loss, test_acc = test(epoch)
        log(str(epoch) +  ", " + str(train_loss) + ", " + str(train_acc) + ", " + str(test_loss) + ", " + str(test_acc), False)

        # Dynamic IC Layer Implmenatation
        if args.dynamic1 and (epoch % 50 == 49):
            if args.net == "LeNet":
                print("changing net p values from ", p0, p1, p2)
                p0 -= 0.05
                p1 -= 0.05
                p2 -= 0.05
                print("to p values ----> ", p0, p1, p2)

            if args.net == "TimNet" and (epoch % 50 == 49):
                print("changing net p values from ", p0, p1, p2, p3)
                p0 -= 0.05
                p1 -= 0.05
                p2 -= 0.05
                p3 -= 0.05
                print("to p values ----> ", p0, p1, p2, p3)
        elif args.dynamic2 and (epoch % 50 == 49):
            if args.net == "LeNet":
                print("changing net p values from ", p0, p1, p2)
                p0 = p0 * .5
                p1 = p1 * .5
                p2 = p2 * .5
                print("to p values ----> ", p0, p1, p2)
            if args.net == "TimNet" and (epoch % 50 == 49):
                print("changing net p values from ", p0, p1, p2, p3)
                p0 = p0 * .5
                p1 = p1 * .5
                p2 = p2 * .5
                p3 = p3 * .5
                print("to p values ----> ", p0, p1, p2, p3)

        elif args.dynamic3 and (epoch % 50 == 49):
            if args.net == "LeNet":
                print("changing net p values from ", p0, p1, p2)
                p0 = p0 * 2
                p1 = p1 * 2
                p2 = p2 * 2
                print("to p values ----> ", p0, p1, p2)
            if args.net == "TimNet" and (epoch % 50 == 49):
                print("changing net p values from ", p0, p1, p2, p3)
                p0 = p0 * 2
                p1 = p1 * 2
                p2 = p2 * 2
                p3 = p3 * 2
                print("to p values ----> ", p0, p1, p2, p3)



elif args.test:
    net, optimizer, _ = load_checkpoint(args.path, net, optimizer)
    test_loss, test_acc = test(0)
    log("test:  " + str(test_loss) + ", " + str(test_acc), True)
