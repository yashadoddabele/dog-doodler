import torch
from torchvision import transforms, datasets
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import *
from VGG16 import VGG16

#Get the data 
data = datasets.ImageFolder(root='cropped_data_10')
num_classes = 10
torch.manual_seed(555)

#Set hyperparams
learning_rate = 0.0001
epochs = 30
batch_size = 10
loss_fn = nn.CrossEntropyLoss()
criterion = nn.CrossEntropyLoss()

#VGG16 model
vgg = VGG16(num_classes)
vgg_optimizer = torch.optim.SGD(vgg.parameters(), lr=learning_rate, weight_decay = 0.005, momentum = 0.9)

#Transformations
transforms = {
    'train':
    transforms.Compose([
        #For Vgg, 224x224
        transforms.RandomResizedCrop(size=236, scale=(0.95, 1.0)),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(),
        transforms.RandomHorizontalFlip(),
        transforms.CenterCrop(size=224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]) 
    ]),
    'test':
    transforms.Compose([
        transforms.CenterCrop(size=224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], 
                             [0.229, 0.224, 0.225])
    ])
}

train_data_len = int(len(data)*0.8)
test_data_len = int(len(data) - train_data_len)
train_data, test_data = random_split(data, [train_data_len, test_data_len])

train_data.dataset.transform = transforms['train']
test_data.dataset.transform = transforms['test']

train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True)

#Determining tensor size
train_features, train_labels = next(iter(train_loader))
print(train_features.size())

#Training loop
def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    avg_batch_loss = 0.0
    for batch, (X, y) in enumerate(dataloader):

        pred = model(X)
        loss = loss_fn(pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        avg_batch_loss += loss.item()

        if ((batch > 0) and (batch % 100 == 0)):
            loss, current = loss.item(), batch * len(X)
            print(f"Average Loss: {avg_batch_loss/100:>7f}  [{current:>5d}/{size:>5d}]")
            avg_batch_loss = 0.0

#Test loop, prints accuracy 
def test_loop(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= size
    correct /= size

    return correct, test_loss

#for determining best model
max_Correct = 0
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train_loop(train_loader, vgg.train(), loss_fn, vgg_optimizer)
    correct, test_loss = test_loop(test_loader, vgg.eval(), loss_fn)

    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

    if (correct > max_Correct):
        max_Correct = correct
        torch.save({'epoch': epochs,
                'model_state_dict': vgg.state_dict(),
                'optimizer_state_dict': vgg_optimizer.state_dict(),
                'loss': criterion,
                }, 'trained_10.pth')

print("Done")