# MNIST-Classification
인공신경망과 딥러닝 강의 Toy Project입니다.

## Setups
- numpy : 1.26.0
- Python : 3.9.18
- pytorch : 2.1.1

## Experiment Setting
- epoch : 30
- batch_size : 64

## Run

```
python main.py --batch_size 64 --num_epochs 30
```

## Difference between Custom MLP and LeNet-5
LeNet-5보다 더 나은 일반화성능을 보이기 위해, Custom MLP 모델과 학습 과정에서 여러가지 Regularization 기법을 적용
1. Augmentation
    - transforms.RandomRotation(degrees=0.3)
    - transforms.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1))
    - transforms.ColorJitter(brightness=0.2, contrast=0.2)
2. Dropout
    - dropout rate : 0.3
3. Batch Nomarlization

## Accuracy & Loss plot
- LeNet5 train & test acc/loss
![LeNet5_train_test_plot](https://github.com/BBongjun/MNIST-Classification/blob/main/plot/LeNet5_train_test_plot.png)

- Custom model train & test acc/loss
![custom_train_test_plot](https://github.com/BBongjun/MNIST-Classification/blob/main/plot/Custom_model_train_test_plot.png) 

## Performance

| Model | Accuracy  |
| ---------------: | -----: |
| LeNet-5    | 98.89% |
| CustomMLP W/O Augmentation  | 91.68% |
| CustomMLP W/O Dropout  | 97.71% |
| CustomMLP W/O BatchNorm  | 97.71% |