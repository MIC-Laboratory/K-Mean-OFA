import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import os
import argparse
from torchvision import transforms
from tqdm import tqdm
from torchvision import models
from ofa.imagenet_classification.networks import MobileNetV3Large
from ofa.imagenet_classification.elastic_nn.networks import OFAMobileNetV3

from torch.utils.tensorboard import SummaryWriter
from cutout import Cutout

# This is training file. Training Resnet18, MobileNetV2 and VGG16 in Cifar10
# Here is some basic training hyper-parameters setup
parser = argparse.ArgumentParser(description=
"""
This is training file. Training Resnet18, MobileNetV2 and VGG16 in Cifar10
argument: 
[models:resnet18,mobilenetv2,vgg16],
[weight_path:The location that you want to save weight file],
[dataset: Cifar10,Cifar100,Imagenet]
[dataset_path:The location of the dataset(For cifar10/cifar100, it can be the save location of the dataset)]
ORDER MATTERS
"""
)
parser.add_argument('models', type=str,help='models:resnet18,mobilenetv2,vgg16')
parser.add_argument('weight_path', type=str)
parser.add_argument('dataset', type=str,help="dataset: Cifar10,Cifar100,Imagenet")
parser.add_argument('dataset_path', type=str)
args = parser.parse_args()
print("==> Setting up hyper-parameters...")
batch_size = 64

training_epoch = 1000
num_workers = 4
lr_rate = 3e-4
momentum = 0.9
weight_decay = 6e-5
best_acc = 0
n_holes = 1
length = 16
if args.dataset == "Cifar10":
    dataset_mean = [0.4914, 0.4822, 0.4465]
    dataset_std = [0.2470, 0.2435, 0.2616]
    input_size = 32
elif args.dataset == "Cifar100":
    dataset_mean = [0.5071, 0.4867, 0.4408]
    dataset_std = [0.2675, 0.2565, 0.2761]
    input_size = 224
elif args.dataset == "Imagenet":
    dataset_mean = [0.485, 0.456, 0.406]
    dataset_std = [0.229, 0.224, 0.225]
    input_size = 224
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
dataset_path = args.dataset_path
weight_path = args.weight_path+"/"+args.models
filename = args.models
log_path = "Experiment_data/Training_Record/"+args.dataset+"/"+args.models
writer = SummaryWriter(log_dir=log_path)
train_transform = transforms.Compose(
    [
    transforms.RandomSizedCrop(input_size),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=dataset_mean,std=dataset_std),
    Cutout(n_holes=n_holes, length=length)
    ])

test_transform = transforms.Compose([
    transforms.RandomSizedCrop(input_size),
    transforms.ToTensor(),
    transforms.Normalize(mean=dataset_mean,std=dataset_std)
])


# Data Preparation
print("==> Preparing data")
if args.dataset == "Cifar10":
    train_set = torchvision.datasets.CIFAR10(dataset_path,train=True,transform=train_transform,download=True)
    test_set = torchvision.datasets.CIFAR10(dataset_path,train=False,transform=test_transform,download=True)
elif args.dataset == "Cifar100":
    train_set = torchvision.datasets.CIFAR100(dataset_path,train=True,transform=train_transform,download=True)
    test_set = torchvision.datasets.CIFAR100(dataset_path,train=False,transform=test_transform,download=True)
elif args.dataset == "Imagenet":
    train_set = torchvision.datasets.ImageFolder(os.path.join(dataset_path,"train"),train=True,transform=train_transform)
    test_set = torchvision.datasets.ImageFolder(os.path.join(dataset_path,"val"),train=False,transform=test_transform)
train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size,
                                          shuffle=True, num_workers=num_workers)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size,
                                         shuffle=False, num_workers=num_workers)

classes = len(train_set.classes)


# Netword preparation
print("==> Preparing models")
print(f"==> Using {device} mode")
if args.models == "OFA":
    net = OFAMobileNetV3(
        n_classes=100,
        bn_param=(0.1, 1e-5),
        dropout_rate=0.1,
        width_mult=1.0,
        ks_list=[3,5,7],
        expand_ratio_list=[3,4,6],
        depth_list=[2,3,4],
    )
net.to(device)
net.load_state_dict(torch.load("OFAWeights/OFA/V4/acc69.09%_OFA")["state_dict"])
lr_rate = torch.load("OFAWeights/OFA/checkpoint")["optimizer"]["param_groups"][0]['lr']
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
optimizer = optim.SGD(net.parameters(), lr=lr_rate,weight_decay=weight_decay)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)


# training
def validation(network,dataloader,file_name,save=True):
    
    # loop over the dataset multiple times
    global best_acc
    accuracy = 0
    running_loss = 0.0
    total = 0
    correct = 0
    network.eval()
    with tqdm(total=len(dataloader)) as pbar:
        with torch.no_grad():
            for i, data in enumerate(dataloader, 0):
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)


                # forward + backward + optimize
                outputs = network(inputs)
                loss = criterion(outputs, labels)
                _, predicted = torch.max(outputs, 1)
                running_loss += loss.item()
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                accuracy = 100 * correct / total
                pbar.update()
                pbar.set_description_str("Acc: {:.3f} {}/{} | Loss: {:.3f}".format(accuracy,correct,total,running_loss/(i+1)))
            if accuracy > best_acc:
                best_acc = accuracy
                if save:
                    if not os.path.isdir(weight_path):
                        os.makedirs(weight_path)
                    PATH = os.path.join(weight_path,"acc"+str(accuracy)+"%_"+file_name)
                    torch.save({"state_dict":network.state_dict()}, PATH)
                    print("Save: Acc "+str(best_acc))
                else:
                    print("Best: Acc "+str(best_acc))
            PATH = os.path.join(weight_path,"checkpoint")
            torch.save({"state_dict":network.state_dict(),"optimizer":optimizer.state_dict()}, PATH)
    return running_loss/len(dataloader),accuracy

def train(epoch,network,optimizer,dataloader):
    # loop over the dataset multiple times
    running_loss = 0.0
    total = 0
    correct = 0
    network.train()
    with tqdm(total=len(dataloader)) as pbar:
        for i, data in enumerate(dataloader, 0):
            
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = network(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            running_loss += loss.item()            
            
            
            accuracy = 100 * correct / total
            pbar.update()
            pbar.set_description_str("Epoch: {} | Acc: {:.3f} {}/{} | Loss: {:.3f}".format(epoch,accuracy,correct,total,running_loss/(i+1)))

    
print("==> Start training/testing")
for epoch in range(training_epoch):
    train(epoch, network=net, optimizer=optimizer,dataloader=train_loader)
    loss,accuracy = validation(network=net,file_name=filename,dataloader=test_loader)
    scheduler.step()
    writer.add_scalar('Test/Loss', loss, epoch)
    writer.add_scalar('Test/ACC', accuracy, epoch)
writer.close()
print("==> Finish")