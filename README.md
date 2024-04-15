# MNIST-Classification
인공신경망과 딥러닝 강의 Toy Project입니다.

## Setups
- numpy : 1.26.0
- Python : 3.9.18
- pytorch : 2.1.1

## Experiment Setting
- epoch : 50
- batch_size : 64

## Run

```
python main.py --batch_size 64 --num_epochs 50
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

## Performance
| Model | Accuracy  |
| ---------------: | -----: |
| LeNet-5    | 99.03% |
| CustomMLP  | 99.23% |

- 같은 에폭, 같은 배치사이즈에서 CustomMLP가 더 나은 일반화 성능을 보임

## Accuracy & Loss plot
- LeNet5 train & test acc/loss
![LeNet5_train_test_plot](https://github.com/BBongjun/MNIST-Classification/blob/main/plot/LeNet5_train_test_plot.png)

- Custom model train & test acc/loss
![custom_train_test_plot](https://github.com/BBongjun/MNIST-Classification/blob/main/plot/Custom_model_train_test_plot.png) 

- LeNet5 vs Custom model 
![custom_train_test_plot](https://github.com/BBongjun/MNIST-Classification/blob/main/plot/test_performance_comparison.png) 


> Regularization 효과가 적절하게 적용되었음 확인 가능
> 1. 학습 에폭이 지나도, 더 나은 일반화 성능을 보임
> 2. LeNet5는 이른 에폭 이후, 