import numpy as np

# 손실 함수
# 신경망 학습에서 사용하는 지표, 신경망 성능의 나쁨을 나타냄
# 임의의 함수를 사용할 수도 있지만 일반적으로는 평균 제곱의 오차와
# 교차 엔트로피 오차를 사용한다

def mean_squared_error(y, t):
    """평균 제곱 오차(Mean Squared Error, MSE)를 구하는 함수
    y = 신경망의 출력(신경망이 추정한 값)
    t = 정답 레이블
    신경망의 추정값과 정답 레이블의 차의 제곱을
    데이터 차원 전부 더해준 다음
    0.5를 곱해준다
    """
    return 0.5 * np.sum((y-t) ** 2)

def cross_entropy_error(y, t):
    """교차 엔트로피 오차(Cross Entropy Error, CEE)를 구하는 함수
    y = 신경망의 출력(신경망이 추정한 값)
    t = 정답 레이블
    정답일때의 추정의 자연 로그를 계산한다.
    """
    delta = 1e-7 # y에 아주 작은 값을 더하여 log함수에서 0을 입력하지 않도록 한다.
    return -np.sum(t * np.log(y + delta))

# 정답은 2
t = [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]

# 예시1 - 정답을 2라고 유추함
y1 = [0.1, 0.05, 0.6, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0]
print("Case 1. MSE: {}".format(mean_squared_error(np.array(y1), np.array(t))))
print("Case 1. CEE: {}".format(cross_entropy_error(np.array(y1), np.array(t))))

# 예시2 - 정답을 7라고 유추함
y2 = [0.1, 0.05, 0.1, 0.0, 0.05, 0.1, 0.0, 0.6, 0.0, 0.0]
print("Case 2. MSE: {}".format(mean_squared_error(np.array(y2), np.array(t))))
print("Case 2. CEE: {}".format(cross_entropy_error(np.array(y2), np.array(t))))