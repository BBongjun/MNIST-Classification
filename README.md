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

## Difference between LeNet5 and LeNet5_Reg
LeNet5_Reg 모델은 여러가지 Regularization 기법을 적용하여 학습

1. **Augmentation**
    - transforms.RandomRotation(degrees=0.2)
    - transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0))
2. **Dropout**
    - dropout rate : 0.3
3. **Batch Nomarlization**

### Model parameter
| Model | Accuracy  |
| ---------------: | -----: |
| LeNet5    | 61706 |
| CustomMLP  | 62930 |
| LeNet5_Reg  | 62158 |


- LeNet5 parameter 계산
    - C1 : 6 x (5x5x1 + 1) = 156
    - S2 : 0
    - C3 : 16 x (5x5x6 + 1) = 2416
    - S4 : 0
    - C5 : 120 x (5x5x16 + 1) = 48120
    - F6 : 84 x (120 + 1) = 10164
    - F7 : 10 x (84 + 1) = 850
        - 총 파라미터 수 : 61706
- CustomMLP parameter 계산
    - F1 : (1024 + 1) * 60 = 61500
    - F2 : (60 + 1) * 20 = 1220
    - F3 : (20 + 1) * 10 = 210
        - 총 파라미터 수 : 62930
- LeNet5_Reg에서 Batch Nomarlization가 추가되어, 파라미터 수가 452(226 x 2)개 더 많음

## Performance
| Model | Accuracy  |
| ---------------: | -----: |
| LeNet5    | 99.03% |
| CustomMLP  | 92.19% |
| **LeNet5_Reg**    | **99.40%** |

- 직접 구현한 LeNet5은 알려져있는 예측 성능 약 99%와 유사함을 확인 가능
- Accuracy : CustomMLP < LeNet5 < LeNet5_Reg

## Accuracy & Loss plot
- LeNet5 train & test acc/loss
![LeNet5_train_test_plot](https://github.com/BBongjun/MNIST-Classification/blob/main/plot/LeNet5_train_test_plot.png)

-------------------------------------------------
- CustomMLP train & test acc/loss
![custom_train_test_plot](https://github.com/BBongjun/MNIST-Classification/blob/main/plot/MLP_train_test_plot.png) 

- LeNet5 vs CustomMLP 
![custom_train_test_plot](https://github.com/BBongjun/MNIST-Classification/blob/main/plot/MLP_test_performance_comparison.png)

--------------------------------------------------
- LeNet5_Reg train & test acc/loss
![LeNet5_Reg_train_test_plot](https://github.com/BBongjun/MNIST-Classification/blob/main/plot/LeNet5_Reg_train_test_plot.png)

- LeNet5 vs LeNet5_Reg 
![custom_train_test_plot](https://github.com/BBongjun/MNIST-Classification/blob/main/plot/LeNet5_Reg_test_performance_comparison.png)

## LeNet5 / CustomMLP / LeNet5_Reg 결과 분석
#### [LeNet5 train & test acc/loss] 
- Train loss는 학습이 진행될수록, 0에 수렴하게 됨
- 하지만, Test loss는 초기 에폭 이후 증가하는 모습을 보임
- LeNet5는 과적합되는 모습을 accuracy와 loss plot을 통해 확인 가능

----------------------------------------

#### [CustomMLP train & test acc/loss]
- Train set에 대하여, LeNet5 보다 학습이 더딘 양상을 확인 가능

#### [LeNet5 vs CustomMLP] 
- Test set에 대하여, LeNet5이 CustomMLP보다 더 높은 성능을 보임

-----------------------------------------
#### [LeNet5_Reg train & test acc/loss] 
- Train loss가 천천히 작아짐
- Test loss도 계속해서 작아지는 모습을 보임
- LeNet5_Reg는 에폭이 지날수록, 더 안정적이고 더 나은 방향으로 학습되고 있음을 확인 가능

#### [LeNet5 vs LeNet5_Reg] 
- LeNet5_Reg가 안정적으로 빠르게 더 나은 일반화 성능을 보임을 확인 가능

#### [Performance]
- Augmentation, Dropout, Batch Normalization을 적용한 후, 더 나은 일반화 성능을 보임(LeNet5_Reg > LeNet5)

### [결론]
- **Regularization 효과가 적절하게 적용되었음 확인 가능!**
