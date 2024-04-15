import dataset
from model import LeNet5, CustomMLP

import argparse
import time
import random
import sys
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

parser = argparse.ArgumentParser(description='MNIST-Classification')
parser.add_argument('--batch_size', default=256, type=int, help='train batch size')
parser.add_argument('--num_epochs', default=50, type=int)

parser.add_argument('--gpuid', default=0, type=int)
parser.add_argument('--seed', default=42, type=int)
args = parser.parse_args()

def set_env(args):
    torch.cuda.set_device(args.gpuid)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

set_env(args)

def train(model, epoch, trn_loader, device, criterion, optimizer):
    """ Train function

    Args:
        model: network
        epoch : epoch
        trn_loader: torch.utils.data.DataLoader instance for training
        device: device for computing, cpu or gpu
        criterion: cost function
        optimizer: optimization method, refer to torch.optim

    Returns:
        trn_loss: average loss value
        acc: accuracy
    """

    model.train()
    total_loss = 0
    correct = 0
    total = 0
    start_time = time.time()
    for batch_idx, (inputs, labels) in enumerate(trn_loader):
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs= model(inputs)
        loss = criterion(outputs,labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * inputs.size(0)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        elapsed_time = time.time() - start_time  # 현재까지의 경과 시간 계산
        hours, rem = divmod(elapsed_time, 3600)  # 시간 및 나머지 계산
        minutes, seconds = divmod(rem, 60)  # 분 및 초 계산
        time_str = "{:0>2}:{:0>2}:{:02.0f}".format(int(hours), int(minutes), seconds)
        # Monitoring
        sys.stdout.write('\rEpoch [%3d/%3d] | Iter[%3d/%3d] | Elapsed Time %s | loss: %.4f' 
                         % (epoch, args.num_epochs, batch_idx + 1, len(trn_loader), time_str, loss.item()))
        sys.stdout.flush()

    trn_loss = total_loss / total
    acc = 100. * correct / total

    return trn_loss, acc

def test(model, tst_loader, device, criterion):
    """ Test function

    Args:
        model: network
        tst_loader: torch.utils.data.DataLoader instance for testing
        device: device for computing, cpu or gpu
        criterion: cost function

    Returns:
        tst_loss: average loss value
        acc: accuracy
    """
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for data, targets in tst_loader:
            data, targets = data.to(device), targets.to(device)
            outputs = model(data)
            loss = criterion(outputs, targets)
            total_loss += loss.item() * data.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()

    tst_loss = total_loss / total
    acc = 100. * correct / total

    return tst_loss, acc

# Function to save loss & acc
def loss_acc_plot(filename, train_acc_history, train_loss_history, test_acc_history, test_loss_history):
    plt.figure(figsize=(12, 6))  # 전체 그래프 크기 조정

    # Accuracy Plot (Training and Test)
    plt.subplot(1, 2, 1)  # 1행 2열의 첫 번째 위치
    plt.plot(train_acc_history, 'b-', label='Train Accuracy', linewidth=2)  # 파란색 실선
    plt.plot(test_acc_history, 'r-', label='Test Accuracy', linewidth=2)  # 빨간색 실선
    plt.title('Train & Test Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True)  # 그리드 추가

    # Loss Plot (Training and Test)
    plt.subplot(1, 2, 2)  # 1행 2열의 두 번째 위치
    plt.plot(train_loss_history, 'b-', label='Train Loss', linewidth=2)  # 파란색 실선
    plt.plot(test_loss_history, 'r-', label='Test Loss', linewidth=2)  # 빨간색 실선
    plt.title('Train & Test Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)  # 그리드 추가

    plt.tight_layout()  # 레이아웃 조정
    plt.savefig(filename)  # 파일로 저장
    plt.close()  # 그림 닫기

def plot_test_comparison(filename, test_acc_history, test_loss_history, custom_test_acc_history, custom_test_loss_history):
    plt.figure(figsize=(12, 6))

    # Testing Accuracy Comparison
    plt.subplot(1, 2, 1)
    plt.plot(test_acc_history, 'r-', label='LeNet5 Test Accuracy', linewidth=2, markersize=6)  # cyan, diamond markers
    plt.plot(custom_test_acc_history, 'b-', label='CustomMLP Test Accuracy', linewidth=2, markersize=6)  # black, x markers
    plt.title('Test Accuracy Comparison')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True)

    # Testing Loss Comparison
    plt.subplot(1, 2, 2)
    plt.plot(test_loss_history, 'r-', label='LeNet5 Test Loss', linewidth=2, markersize=6)  # yellow, vline markers
    plt.plot(custom_test_loss_history, 'b-', label='CustomMLP Test Loss', linewidth=2, markersize=6)  # orange, hexagon markers
    plt.title('Test Loss Comparison')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(filename)
    plt.show()

    
def main():
    """ Main function

        Here, you should instantiate
        1) Dataset objects for training and test datasets
        2) DataLoaders for training and testing
        3) model
        4) optimizer: SGD with initial learning rate 0.01 and momentum 0.9
        5) cost function: use torch.nn.CrossEntropyLoss

    """
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    
    BATCH_SIZE = args.batch_size
    EPOCH = args.num_epochs

    # Define Dataloader
    train_dataset = dataset.MNIST(data_dir='./data/train.tar', augmentation=False)
    custom_train_dataset = dataset.MNIST(data_dir='./data/train.tar', augmentation=True)
    test_dataset = dataset.MNIST(data_dir='./data/test.tar', augmentation=False)

    train_loader = DataLoader(dataset=train_dataset,
                              batch_size = BATCH_SIZE,
                              shuffle=True)
    custom_train_loader = DataLoader(dataset=custom_train_dataset,
                                     batch_size=BATCH_SIZE,
                                     shuffle=True)
    test_loader = DataLoader(dataset=test_dataset,
                             batch_size=BATCH_SIZE*5,
                             shuffle=False)
    
    # Initialize LeNet5 Model
    LeNet = LeNet5(n_classes=10).cuda()
    Custom_model = CustomMLP(n_classes=10).cuda()
    print(f'The number of LeNet5 parameters: {count_parameters(LeNet)}')
    print(f'The number of Custom_model parameters: {count_parameters(Custom_model)}')

    # Cost function & Optimizer
    criterion = torch.nn.CrossEntropyLoss()
    LeNet_optimizer = torch.optim.SGD(LeNet.parameters(), lr=0.01, momentum=0.9)
    Custom_model_optimizer = torch.optim.SGD(Custom_model.parameters(), lr=0.01, momentum=0.9)

    # # LeNet5 train & test
    print('============LeNet5 train & test Start============')
    LeNet5_train_loss_history, LeNet5_train_acc_history = [], []
    LeNet5_test_loss_history, LeNet5_test_acc_history = [], []
    for epoch in range(EPOCH):
        trn_loss, trn_acc = train(LeNet, epoch+1, train_loader, device, criterion, LeNet_optimizer)
        test_loss, test_acc = test(LeNet, test_loader, device, criterion)
        print("\n(Epoch %d) Train acc : %.2f%%, Train loss : %.3f | Test acc : %.2f%%, Test loss : %.3f \n" % (epoch+1, trn_acc, trn_loss, test_acc, test_loss))
        
        LeNet5_train_loss_history.append(trn_loss)
        LeNet5_train_acc_history.append(trn_acc)
        LeNet5_test_loss_history.append(test_loss)
        LeNet5_test_acc_history.append(test_acc)
    loss_acc_plot('./plot/LeNet5_train_test_plot_tanh', LeNet5_train_acc_history, LeNet5_train_loss_history, LeNet5_test_acc_history, LeNet5_test_loss_history)
    
    # Custom model train & test
    print('============CustomMLP train & test Start============')
    Custom_model_train_loss_history, Custom_model_train_acc_history = [], []
    Custom_model_test_loss_history, Custom_model_test_acc_history = [], []
    for epoch in range(EPOCH):
        trn_loss, trn_acc = train(Custom_model, epoch+1, custom_train_loader, device, criterion, Custom_model_optimizer)
        test_loss, test_acc = test(Custom_model, test_loader, device, criterion)
        print("\n(Epoch %d) Train acc : %.2f%%, Train loss : %.3f | Test acc : %.2f%%, Test loss : %.3f \n" % (epoch+1, trn_acc, trn_loss, test_acc, test_loss))
        
        Custom_model_train_loss_history.append(trn_loss)
        Custom_model_train_acc_history.append(trn_acc)
        Custom_model_test_loss_history.append(test_loss)
        Custom_model_test_acc_history.append(test_acc)
    loss_acc_plot('./plot/Custom_model_train_test_plot_tanh', Custom_model_train_acc_history, Custom_model_train_loss_history, Custom_model_test_acc_history, Custom_model_test_loss_history)

    plot_test_comparison( './plot/test_performance_comparison_tanh.png', LeNet5_test_acc_history, LeNet5_test_loss_history, Custom_model_test_acc_history, Custom_model_test_loss_history)


    print('='*40)

    print(f'LeNet5 - Last epoch test acc: {LeNet5_test_acc_history[-1]}')
    print(f'Custom model - Last epoch test acc: {Custom_model_test_acc_history[-1]}')


if __name__ == '__main__':
    main()