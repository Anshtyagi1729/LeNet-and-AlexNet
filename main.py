import torch 
from torchvision import datasets,transforms
from torch.utils.data import DataLoader
from models import LeNet , AlexNet 
from utils import init_kaiming, init_xavier
from train import train
device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_name="AlexNet"
resize=224 if model_name =="AlexNet" else 28
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                         std=[0.2023, 0.1994, 0.2010])
])
train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
test_dataset  = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader  = DataLoader(test_dataset, batch_size=64, shuffle=False)


model=LeNet() if model_name=="LeNet" else AlexNet()
if model_name=="LeNet":
    model.apply(init_xavier)
else:
    model.apply(init_kaiming)

train(model,train_loader,test_loader,device)

