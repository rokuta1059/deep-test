import numpy as np
 
def sigmoid(x):
    """3.2.1 시그모이드 함수 
    
    신경망에서 자주 이용하는 활성화 함수인 시그모이드 함수(sigmoid function)를 나타냅니다.
    exp(-x)는 자연상수 e^(-x)를 나타냅니다.
    """
    return 1 / (1 + np.exp(-x))

def step_function(x):
    """3.2.2 계단 함수

    배열 x의 원소 각각이 0보다 크면 True로, 0 이하면 False로 변환한 새로운 배열이 생성됩니다.
    dtype 인수를 np.int로 주어 bool을 int로 형변환하여 줍니다.
    """
    return np.array(x > 0, dtype=np.int)

def relu(x):
    """3.2.7 ReLU(Rectified Linear Unit) 함수
    
    입력이 0을 넘으면 그 입력을 그대로 출력하고, 0 이하면 0을 출력하는 함수입니다."""
    return np.maximum(0, x)

def identity_function(x):
    """3.5.1 항등 함수 구현하기
    
    입력을 그대로 출력합니다."""
    return x

def softmax(a):
    """3.5.1 소프트맥스 함수 구현하기
    
    소프트맥스 함수의 분자는 입력 신호의 지수 함수,
    분모는 모든 입력 신호의 지수 함수의 합으로 구성
    그냥 사용시 오버플로 문제가 발생하므로 각각의 분자 분모에 임의의 정수 C를 곱해줍니다.
    즉 지수함수 안쪽에 logC = C'를 빼 준 값을 이용합니다.
    c는 배열의 최대값을 사용합니다.
    결과값의 총합은 1입니다.
    """
    c = np.max(a)
    exp_a = np.exp(a - c)  # 오버플로 방지
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a

    return y