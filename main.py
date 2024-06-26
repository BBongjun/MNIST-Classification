import dataset
from model import LeNet5, LeNet5_Reg, CustomMLP

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
    np.random.seed(args.seed)
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
    plt.subplot(1, 2, 1) 
    plt.plot(train_acc_history, 'b-', label='Train Accuracy', linewidth=2)  
    plt.plot(test_acc_history, 'r-', label='Test Accuracy', linewidth=2)  
    plt.title('Train & Test Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True)

    # Loss Plot (Training and Test)
    plt.subplot(1, 2, 2) 
    plt.plot(train_loss_history, 'b-', label='Train Loss', linewidth=2) 
    plt.plot(test_loss_history, 'r-', label='Test Loss', linewidth=2) 
    plt.title('Train & Test Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)  

    plt.tight_layout()  
    plt.savefig(filename)  
    plt.close()  

def LeNet_Reg_plot_test_comparison(filename, test_acc_history, test_loss_history, custom_test_acc_history, custom_test_loss_history):
    plt.figure(figsize=(12, 6))

    # Testing Accuracy Comparison
    plt.subplot(1, 2, 1)
    plt.plot(test_acc_history, 'r-', label='LeNet5 Test Accuracy', linewidth=2, markersize=6)  # cyan, diamond markers
    plt.plot(custom_test_acc_history, 'b-', label='LeNet5_Reg Test Accuracy', linewidth=2, markersize=6)  # black, x markers
    plt.title('Test Accuracy Comparison')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True)

    # Testing Loss Comparison
    plt.subplot(1, 2, 2)
    plt.plot(test_loss_history, 'r-', label='LeNet5 Test Loss', linewidth=2, markersize=6)  # yellow, vline markers
    plt.plot(custom_test_loss_history, 'b-', label='LeNet5_Reg Test Loss', linewidth=2, markersize=6)  # orange, hexagon markers
    plt.title('Test Loss Comparison')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(filename)
    plt.show()

def CustomMLP_plot_test_comparison(filename, test_acc_history, test_loss_history, custom_test_acc_history, custom_test_loss_history):
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
    
    # Initialize LeNet5 Model & CustomMLP & LeNet5_Reg
    LeNet = LeNet5(n_classes=10).cuda()
    MLP = CustomMLP(n_classes=10).cuda()
    LeNet_Reg = LeNet5_Reg(n_classes=10).cuda()

    print(f'The number of LeNet5 parameters: {count_parameters(LeNet)}')
    print(f'The number of MLP parameters: {count_parameters(MLP)}')
    print(f'The number of LeNet_Reg parameters: {count_parameters(LeNet_Reg)}')

    # Cost function & Optimizer
    criterion = torch.nn.CrossEntropyLoss()
    LeNet_optimizer = torch.optim.SGD(LeNet.parameters(), lr=0.01, momentum=0.9)
    MLP_optimizer = torch.optim.SGD(MLP.parameters(), lr=0.01, momentum=0.9)
    LeNet_Reg_optimizer = torch.optim.SGD(LeNet_Reg.parameters(), lr=0.01, momentum=0.9)

    # LeNet5 train & test
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
    loss_acc_plot('./plot/LeNet5_train_test_plot', LeNet5_train_acc_history, LeNet5_train_loss_history, LeNet5_test_acc_history, LeNet5_test_loss_history)
    
    # CustomMLP train & test
    print('============MLP train & test Start============')
    MLP_train_loss_history, MLP_train_acc_history = [], []
    MLP_test_loss_history, MLP_test_acc_history = [], []
    for epoch in range(EPOCH):
        trn_loss, trn_acc = train(MLP, epoch+1, train_loader, device, criterion, MLP_optimizer)
        test_loss, test_acc = test(MLP, test_loader, device, criterion)
        print("\n(Epoch %d) Train acc : %.2f%%, Train loss : %.3f | Test acc : %.2f%%, Test loss : %.3f \n" % (epoch+1, trn_acc, trn_loss, test_acc, test_loss))
        
        MLP_train_loss_history.append(trn_loss)
        MLP_train_acc_history.append(trn_acc)
        MLP_test_loss_history.append(test_loss)
        MLP_test_acc_history.append(test_acc)
    loss_acc_plot('./plot/MLP_train_test_plot', MLP_train_acc_history, MLP_train_loss_history, MLP_test_acc_history, MLP_test_loss_history)
    CustomMLP_plot_test_comparison('./plot/MLP_test_performance_comparison.png', LeNet5_test_acc_history, LeNet5_test_loss_history, MLP_test_acc_history, MLP_test_loss_history)

    # LeNet5_Reg train & test
    print('============LeNet5_Reg train & test Start============')
    LeNet_Reg_train_loss_history, LeNet_Reg_train_acc_history = [], []
    LeNet_Reg_test_loss_history, LeNet_Reg_test_acc_history = [], []
    for epoch in range(EPOCH):
        trn_loss, trn_acc = train(LeNet_Reg, epoch+1, custom_train_loader, device, criterion, LeNet_Reg_optimizer)
        test_loss, test_acc = test(LeNet_Reg, test_loader, device, criterion)
        print("\n(Epoch %d) Train acc : %.2f%%, Train loss : %.3f | Test acc : %.2f%%, Test loss : %.3f \n" % (epoch+1, trn_acc, trn_loss, test_acc, test_loss))
        
        LeNet_Reg_train_loss_history.append(trn_loss)
        LeNet_Reg_train_acc_history.append(trn_acc)
        LeNet_Reg_test_loss_history.append(test_loss)
        LeNet_Reg_test_acc_history.append(test_acc)
    loss_acc_plot('./plot/LeNet5_Reg_train_test_plot', LeNet_Reg_train_acc_history, LeNet_Reg_train_loss_history, LeNet_Reg_test_acc_history, LeNet_Reg_test_loss_history)
    LeNet_Reg_plot_test_comparison('./plot/LeNet5_Reg_test_performance_comparison.png', LeNet5_test_acc_history, LeNet5_test_loss_history, LeNet_Reg_test_acc_history, LeNet_Reg_test_loss_history)


    print('='*40)

    print(f'LeNet5 - Last epoch test acc: {LeNet5_test_acc_history[-1]}')
    print(f'Custom model - Last epoch test acc: {MLP_test_acc_history[-1]}')
    print(f'LeNet5_Reg model - Last epoch test acc: {LeNet_Reg_test_acc_history[-1]}')


if __name__ == '__main__':
    main()