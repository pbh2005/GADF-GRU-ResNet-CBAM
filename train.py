import torch
import torchvision
from joblib import dump, load
import torch.nn as nn
import time
import matplotlib.pyplot as plt
import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
from matplotlib import MatplotlibDeprecationWarning
warnings.filterwarnings("ignore", category=MatplotlibDeprecationWarning)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from model import ResNetCBAMGRUModel
class SignalImageDataset(Dataset):
    def __init__(self, signal_data, image_dir, transform=None):
        self.signal_data = signal_data
        self.image_dir = image_dir
        if transform is None:
            self.transform = torchvision.transforms.Compose(
                [
                    torchvision.transforms.Resize(size = (224,224)),
                    torchvision.transforms.ToTensor(),
                    torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                ])
        else:
            self.transform = transform

        self.num_samples = len(signal_data)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        signal = self.signal_data[index, :-1].astype(np.float32)
        signal = torch.tensor(signal, dtype=torch.float32)

        label = torch.tensor(self.signal_data[index, -1], dtype=torch.long)

        image_path = os.path.join(self.image_dir, f'signal_{index}.png')
        image = Image.open(image_path).convert('RGB')
        if self.transform:
            image = self.transform(image)

        return signal, image, label

def moedel_train(train_loader, val_loader, model, parameter):
    model = model.to(device)
    batch_size = parameter['batch_size']
    epochs = parameter['epochs']
    loss_function = nn.CrossEntropyLoss(reduction='sum')
    optimizer = torch.optim.Adam(model.parameters(), lr=parameter['learn_rate'])
    train_size = len(train_loader) * batch_size
    val_size = len(val_loader) * batch_size
    best_accuracy = 0.0
    best_model = model
    last_model = model

    train_loss = []
    train_acc = []
    validate_acc = []
    validate_loss = []
    print('*'*20, '开始训练', '*'*20)
    start_time = time.time()
    for epoch in range(epochs):
        model.train()
        print(f"epoch--:{epoch}")
        loss_epoch = 0.
        correct_epoch = 0
        for signals, images, labels in train_loader:
            signals, images, labels = signals.to(device), images.to(device), labels.to(device)
            y_pred = model(signals, images)
            correct_epoch += torch.sum(y_pred.argmax(dim=1).view(-1) == labels.view(-1)).item()
            loss = loss_function(y_pred, labels)
            loss_epoch += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        train_Accuracy = correct_epoch / train_size
        train_loss.append(loss_epoch / train_size)
        train_acc.append(train_Accuracy)
        print(f'Epoch: {epoch + 1:2} train_Loss: {loss_epoch / train_size:10.8f} train_Accuracy:{train_Accuracy:4.4f}')
        with torch.no_grad():
            model.eval()
            loss_validate = 0.
            correct_validate = 0
            for signals, images, labels in val_loader:
                signals, images, labels = signals.to(device), images.to(device), labels.to(device)
                pre = model(signals, images)
                correct_validate += torch.sum(pre.argmax(dim = 1).view(-1) == labels.view(-1)).item()
                loss = loss_function(pre, labels)
                loss_validate += loss.item()

            val_accuracy = correct_validate / val_size
            print(f'Epoch: {epoch + 1:2} val_Loss:{loss_validate / val_size:10.8f},  validate_Acc:{val_accuracy:4.4f}')
            validate_loss.append(loss_validate / val_size)
            validate_acc.append(val_accuracy)
            if val_accuracy > best_accuracy:
                best_accuracy = val_accuracy
                best_model = model

    last_model = model
    print('*' * 20, '训练结束', '*' * 20)
    print(f'\nDuration: {time.time() - start_time:.0f} seconds')
    print(f'best_accuracy: {best_accuracy}')
    plt.figure(figsize=(14, 7), dpi=300)
    plt.plot(range(epochs), train_loss, color='blue', marker='o', label='Train-loss')
    plt.plot(range(epochs), train_acc, color='green', marker='*', label='Train-accuracy')
    plt.plot(range(epochs), validate_loss, color='red', marker='+', label='Validate_loss')
    plt.plot(range(epochs), validate_acc, color='orange', marker='x', label='Validate_accuracy')

    plt.xlabel('Epochs', fontsize=12)
    plt.ylabel('Loss-Accuracy', fontsize=12)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.legend(fontsize=12)
    plt.title('Multimodal-ResNetCBAMGRU-Model training visualization', fontsize=16)
    dump(train_loss, './0ph_A_结果/train_loss')
    dump(train_acc, './0ph_A_结果/train_acc')
    dump(validate_loss, './0ph_A_结果/validate_loss')
    dump(validate_acc, './0ph_A_结果/validate_acc')

    plt.savefig('Train_visualization.png', dpi=300)
    return  last_model, best_model

if __name__ == '__main__':
    train_data = load('./dataresult_A/train_data')
    val_data = load('./dataresult_A/val_data')
    test_data = load('./dataresult_A/test_data')

    train_path = './GADFImages_A/train/'
    val_path = './GADFImages_A/val/'
    test_path = './GADFImages_A/test/'

    train_dataset = SignalImageDataset(train_data, train_path)
    val_dataset = SignalImageDataset(val_data, val_path)
    test_dataset = SignalImageDataset(test_data, test_path)

    batch_size = 32
    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                                               shuffle=True, pin_memory=True, num_workers=2, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size,
                                             shuffle=True, pin_memory=True, num_workers=2, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size,
                                              shuffle=True, pin_memory=True, num_workers=2, drop_last=True)
    dump(test_loader, '0ph_A_结果/test_loader')

    num_classes = 10
    hidden_layer_sizes = [32, 64]
    gru_input_dim = 32

    init_channels = 3
    resnet_type = 18

    epochs = 50
    learn_rate = 0.0003
    parameter = {
        'batch_size': batch_size,
        'epochs': epochs,
        'learn_rate': learn_rate
    }

    model = ResNetCBAMGRUModel(gru_input_dim, hidden_layer_sizes, num_classes, resnet_type, init_channels)

    last_model, best_model = moedel_train(train_loader, val_loader, model, parameter)
    torch.save(best_model, 'GADF_A.pt')
