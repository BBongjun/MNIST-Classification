# MNIST-Classification
인공신경망과 딥러닝 강의 Toy Project입니다.

## Setups
- numpy : 1.26.0
- Python : 3.9.18
- pytorch : 2.1.1

## Experiment Setting
- epoch : 50
- batch_size : 128
- activation function : ReLU
- optimizer : SGD with momentum 0.9
- lr : 0.01 

## Run

```
python main.py --batch_size 128 --num_epochs 50
```

## Difference between Custom MLP and LeNet-5
LeNet-5보다 더 나은 일반화성능을 보이기 위해, Custom MLP 모델과 학습 과정에서 여러가지 Regularization 기법을 적용
1. Augmentation
    - transforms.RandomRotation(degrees=0.2)
    - transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0))
2. Dropout
    - dropout rate : 0.3
3. Batch Nomarlization
    - careful weight initialization
    - 평균과 분산이 지속적으로 변하므로, weight 업데이트에 영향을 주어 

### Model parameter
| Model | Accuracy  |
| ---------------: | -----: |
| LeNet5    | 61706 |
| CustomMLP  | 62158 |
- CustomMLP에서 Batch Nomarlization가 추가되어, 학습해야하는 파라미터 수가 조금 더 많음

## Performance
| Model | Accuracy  |
| ---------------: | -----: |
| LeNet5    | 99.03% |
| **CustomMLP**  | **99.23%** |

- 직접 구현한 LeNet5은 알려져있는 예측 성능 약 99%와 유사함을 확인 가능

## Accuracy & Loss plot
- LeNet5 train & test acc/loss
![LeNet5_train_test_plot](https://github.com/BBongjun/MNIST-Classification/blob/main/plot/LeNet5_train_test_plot.png)

- CustomMLP train & test acc/loss
![custom_train_test_plot](https://github.com/BBongjun/MNIST-Classification/blob/main/plot/Custom_model_train_test_plot.png) 

- LeNet5 vs CustomMLP 
![custom_train_test_plot](https://github.com/BBongjun/MNIST-Classification/blob/main/plot/test_performance_comparison.png) 

## Performance 및 Acc/Loss plot 결과 해석
#### [LeNet5 train & test acc/loss] 
- Train loss는 학습이 진행될수록, 0에 수렴하게 됨
- 하지만, Test loss는 초기 에폭 이후 증가하는 모습을 보임
- LeNet5는 과적합되는 모습을 accuracy와 loss plot을 통해 확인 가능
#### [CustomMLP train & test acc/loss] 
- Train loss가 천천히 작아짐
- Test loss도 계속해서 작아지는 모습을 보임
- CustomMLP는 에폭이 지날수록, 더 안정적이고 더 나은 방향으로 학습되고 있음을 확인 가능
#### [LeNet5 vs CustomMLP] 
- CustomMLP가 안정적으로 빠르게 더 나은 일반화 성능을 보임을 확인 가능
#### [Performance]
- Augmentation, Dropout, Batch Normalization을 적용한 후, 더 나은 일반화 성능을 보임(CustomMLP > LeNet5)
### [결론]
- **Regularization 효과가 적절하게 적용되었음 확인 가능!**
