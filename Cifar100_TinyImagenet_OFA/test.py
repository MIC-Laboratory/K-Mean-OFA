from ofa.imagenet_classification.elastic_nn.networks import OFAMobileNetV3
import torchvision
from torchvision import transforms
import torch
import os
from tqdm import tqdm
net = OFAMobileNetV3(
        n_classes=100,
        bn_param=(0.1, 1e-5),
        dropout_rate=0.1,
        width_mult=1.0,
        ks_list=[3,5,7],
        expand_ratio_list=[3,4,6],
        depth_list=[2,3,4],
    )
dataset_mean = [0.5071, 0.4867, 0.4408]
dataset_std = [0.2675, 0.2565, 0.2761]
test_transform = transforms.Compose([
    transforms.RandomSizedCrop(223),
    transforms.ToTensor(),
    transforms.Normalize(mean=dataset_mean,std=dataset_std)
])
test_set = torchvision.datasets.CIFAR100("/home/zhenyulin/Training_data",train=False,transform=test_transform,download=True)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=32,
                                         shuffle=False, num_workers=2)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
net.load_state_dict(torch.load("exp/kernel_depth2kernel_depth_width_L1/phase2/checkpoint/model_best.pth.tar")["state_dict"])
# net.set_active_subnet(**{'d': 4, 'e': 5, 'ks': 7, 'w': 0})
def validation(network,dataloader,save=True):
    
    # loop over the dataset multiple times
    global best_acc
    accuracy = 0
    running_loss = 0.0
    total = 0
    correct = 0
    network.eval()
    network.to(device)
    with tqdm(total=len(dataloader)) as pbar:
        with torch.no_grad():
            for i, data in enumerate(dataloader, 0):
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)


                # forward + backward + optimize
                outputs = network(inputs)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                accuracy = 100 * correct / total
                pbar.update()
                pbar.set_description_str("Acc: {:.3f} {}/{} | Loss: {:.3f}".format(accuracy,correct,total,running_loss/(i+1)))
            
    return running_loss/len(dataloader),accuracy
validation(net,test_loader)