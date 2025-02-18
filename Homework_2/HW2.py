import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from torchvision.models import resnet18

#Part 1
# Define the modified AlexNet
class AlexNetCIFAR(nn.Module):
    def __init__(self, num_classes=10, use_dropout=False):
        super(AlexNetCIFAR, self).__init__()
        self.use_dropout = use_dropout
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5) if use_dropout else nn.Identity(),
            nn.Linear(1024, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

# Data preparation
def get_data_loaders(batch_size=128, dataset='CIFAR100'):
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    if dataset == 'CIFAR10':
        dataset_class = torchvision.datasets.CIFAR10
        num_classes = 10
    else:
        dataset_class = torchvision.datasets.CIFAR100
        num_classes = 100
    
    train_set = dataset_class(root='./data', train=True, download=True, transform=transform)
    test_set = dataset_class(root='./data', train=False, download=True, transform=transform)
    
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader, num_classes

# Training function
def train_model(model, train_loader, test_loader, num_epochs=20, lr=0.001):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct, total = 0, 0
        
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
        
        train_acc = 100 * correct / total
        val_loss, val_acc = evaluate_model(model, test_loader, criterion, device)
        
        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {running_loss/len(train_loader):.4f}, "
              f"Train Acc: {train_acc:.2f}%, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
    
# Evaluation function
def evaluate_model(model, loader, criterion, device):
    model.eval()
    loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss += criterion(outputs, labels).item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    return loss / len(loader), 100 * correct / total

# Parameter count
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

if __name__ == "__main__":
    dataset = 'CIFAR100'  # Change to 'CIFAR100' for CIFAR-100
    train_loader, test_loader, num_classes = get_data_loaders(dataset=dataset)
    
    print("Training AlexNet without Dropout")
    model_no_dropout = AlexNetCIFAR(num_classes=num_classes, use_dropout=False)
    print(f"Total parameters: {count_parameters(model_no_dropout):,}")
    train_model(model_no_dropout, train_loader, test_loader)
    
    print("\nTraining AlexNet with Dropout")
    model_with_dropout = AlexNetCIFAR(num_classes=num_classes, use_dropout=True)
    print(f"Total parameters: {count_parameters(model_with_dropout):,}")
    train_model(model_with_dropout, train_loader, test_loader)

#Part 2
# Define the VGG architecture for CIFAR with Dropout
class VGG_CIFAR(nn.Module):
    def __init__(self, cfg, num_classes=10, use_dropout=False):
        super(VGG_CIFAR, self).__init__()
        self.use_dropout = use_dropout
        self.features = self._make_layers(cfg)
        self.classifier = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5) if use_dropout else nn.Identity(),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
    
    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for v in cfg:
            if v == 'M':
                layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            else:
                layers.append(nn.Conv2d(in_channels, v, kernel_size=3, padding=1))
                layers.append(nn.ReLU(inplace=True))
                if self.use_dropout:
                    layers.append(nn.Dropout(0.3))
                in_channels = v
        return nn.Sequential(*layers)

# VGG configurations (VGG-11, VGG-13, VGG-16, VGG-19)
vgg_configs = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M']
}

selected_vgg = 'VGG11'  # Change to 'VGG13' if needed

def get_data_loaders(batch_size=128, dataset='CIFAR100'):
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    dataset_class = torchvision.datasets.CIFAR10 if dataset == 'CIFAR10' else torchvision.datasets.CIFAR100
    num_classes = 10 if dataset == 'CIFAR10' else 100
    train_set = dataset_class(root='./data', train=True, download=True, transform=transform)
    test_set = dataset_class(root='./data', train=False, download=True, transform=transform)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader, num_classes

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def train_model(model, train_loader, test_loader, num_epochs=20, lr=0.001):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    for epoch in range(num_epochs):
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
        train_acc = 100 * correct / total
        print(f"Epoch {epoch+1}, Train Loss: {running_loss/len(train_loader):.4f}, Train Acc: {train_acc:.2f}%")

if __name__ == "__main__":
    dataset = 'CIFAR10'  # Change to 'CIFAR100' for CIFAR-100
    train_loader, test_loader, num_classes = get_data_loaders(dataset=dataset)
    
    print("Training VGG without Dropout")
    model_no_dropout = VGG_CIFAR(cfg=vgg_configs[selected_vgg], num_classes=num_classes, use_dropout=False)
    print(f"Total parameters in {selected_vgg} (no dropout): {count_parameters(model_no_dropout):,}")
    train_model(model_no_dropout, train_loader, test_loader)
    
    print("\nTraining VGG with Dropout")
    model_with_dropout = VGG_CIFAR(cfg=vgg_configs[selected_vgg], num_classes=num_classes, use_dropout=True)
    print(f"Total parameters in {selected_vgg} (with dropout): {count_parameters(model_with_dropout):,}")
    train_model(model_with_dropout, train_loader, test_loader)

#Part 3
# Define a modified ResNet-18 with optional dropout
class ModifiedResNet18(nn.Module):
    def __init__(self, num_classes=10, dropout_rate=0.0):
        super(ModifiedResNet18, self).__init__()
        self.model = resnet18(weights=None)
        self.model.fc = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(self.model.fc.in_features, num_classes)
        )
    
    def forward(self, x):
        return self.model(x)

# Data preprocessing
def get_dataloaders(batch_size=128, dataset="CIFAR10"):
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    if dataset == "CIFAR10":
        trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
        testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    else:
        trainset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform)
        testset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform)
    
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    return trainloader, testloader

# Training function
def train_model(model, trainloader, testloader, num_epochs=10, lr=0.001):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    train_losses, val_losses, val_accuracies = [], [], []
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in trainloader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        
        train_losses.append(running_loss / len(trainloader))
        
        # Validation
        model.eval()
        correct = 0
        total = 0
        val_loss = 0.0
        with torch.no_grad():
            for inputs, labels in testloader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        
        val_losses.append(val_loss / len(testloader))
        val_accuracy = 100 * correct / total
        val_accuracies.append(val_accuracy)
        
        print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_losses[-1]:.4f}, Val Loss: {val_losses[-1]:.4f}, Val Acc: {val_accuracy:.2f}%')
    
    return train_losses, val_losses, val_accuracies

# Plot results
def plot_results(train_losses, val_losses, val_accuracies, dataset, dropout_rate):
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title(f'{dataset} Loss (Dropout={dropout_rate})')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(val_accuracies, label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.title(f'{dataset} Accuracy (Dropout={dropout_rate})')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    for dataset in ["CIFAR10", "CIFAR100"]:
        for dropout_rate in [0.0, 0.3]:
            print(f'Training on {dataset} with Dropout={dropout_rate}')
            trainloader, testloader = get_dataloaders(dataset=dataset)
            model = ModifiedResNet18(num_classes=10 if dataset == "CIFAR10" else 100, dropout_rate=dropout_rate)
            train_losses, val_losses, val_accuracies = train_model(model, trainloader, testloader, num_epochs=10)
            plot_results(train_losses, val_losses, val_accuracies, dataset, dropout_rate)

