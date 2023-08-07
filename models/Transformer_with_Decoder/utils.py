import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch.nn.functional as F
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np

def get_dataloader(image_size, batch_size, split='train'):
    if split == 'train':
        train = True
        shuffle = True
        transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.RandomCrop(image_size, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    elif split == 'test':
        train = False
        shuffle = False
        transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    
    # Load data
    dataset = datasets.CIFAR10(root='./data', train=train, download=True, transform=transform)

    # Package it up in batches
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

    return dataloader

def train_and_eval(model, optimizer, device, train_loader, test_loader, num_epochs, DEBUG=False):
    # Train model
    train_loss_list = []
    test_loss_list = []
    for epoch in tqdm(range(num_epochs)):
        train_loss = 0
        test_loss = 0
        for i, (images, labels) in tqdm(enumerate(train_loader)):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = F.cross_entropy(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * images.size(0)

            if DEBUG:
                if (i + 1) % 100 == 0:
                    print(f"Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}")

        train_loss /= len(train_loader.dataset)
        train_loss_list.append(train_loss)

        # Evaluate model on test set
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = F.cross_entropy(outputs, labels)
                test_loss += loss.item() * images.size(0)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        test_loss /= len(test_loader.dataset)
        test_loss_list.append(test_loss)
        accuracy = (100 * correct / total)
        print(f"Accuracy on test set: {accuracy:.2f}%")

        # Save checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            torch.save(model.state_dict(), 'ckpt_ViT_{}epochs_8patch.pth'.format(epoch))

        # Set model back to training mode
        model.train()
    
    return train_loss_list, test_loss_list

def plot_losses(train_loss_list, test_loss_list):
    num_epochs = len(train_loss_list)
    plt.plot(range(num_epochs), train_loss_list, label='Training loss')
    plt.plot(range(num_epochs), test_loss_list, label='Evaluation loss')
    plt.xlabel('Number of epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

class UnitGaussianNormalizer(object):  
    def __init__(self, x, eps=1e-5):  
        super(UnitGaussianNormalizer, self).__init__()  
  
        self.mean = np.mean(x, 0)  
        self.std = np.std(x, 0) 
        self.eps = eps  
  
    def encode(self, x):  
        return (x - self.mean) / (self.std + self.eps)  
  
    def decode(self, x):  
        return (x * (self.std + self.eps)) + self.mean  
    
class InactiveNormalizer(object):  
    def __init__(self, x, eps=1e-5):  
        super(InactiveNormalizer, self).__init__()  
    
    def encode(self, x):  
        return x 
  
    def decode(self, x):  
        return x


class MaxMinNormalizer(object):  
    def __init__(self, x, eps=1e-5):  
        super(MaxMinNormalizer, self).__init__()  
  
        self.max = np.max(x, 0)  
        self.min = np.min(x, 0) 
        self.range = self.max - self.min
        self.eps = eps  
  
    def encode(self, x):  
        return (x - self.min) / (self.range + self.eps)  
  
    def decode(self, x):  
        return self.min + x * (self.range + self.eps)