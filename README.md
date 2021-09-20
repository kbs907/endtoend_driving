## Composition

* end2end_drive </br></br>학습된 모델을 가지고 xycar를 이용하여 주행하는 패키지

* end2end_learning </br></br>학습 데이터를 가지고 모델을 학습시키는 패키지

* make_learning_data </br></br>자이카를 이용하여 영상과 각 프레임별 조향값을 생성하는 패키지

## Procedure

1. make_learning_data 패키지를 이용하여 학습에 사용할 영상과 각 프레임별 조향값 데이터를 생성
2. end2end_learning 패키지에 영상과 조향값 데이터 파일을 넣고 vnorm.py를 실행하여 학습에 직접 이용할 수 있는 형태로 데이터를 생성
3. end2end_learning 패키지의 train.py를 실행하여 모델 학습 시작
4. end2end_drive 패키지의 save폴더에 이전 3번 과정의 save폴더에 생성된 학습된 모델을 가져와서 적용

* 학습에 충분한 양질의 데이터를 확보할 수 없기도 하고, 학습의 효율성을 위해 이미지 전처리를 함. (최대한 차선과 관련된 이미지 데이터만 쓰기 위해 ROI 설정과 Canny edge 처리)

## Limitations

학습에 사용할 영상을 제작할 때 주행한 속도보다 빠른 속도로 주행하지 못함

## What I've learned

* 이미지 전처리를 하니 학습 효과가 상당히 좋았음
* 양질의 데이터와 수가 매우 중요하며 수집하는데 어려움
* 차선을 따라 이동하는 것뿐만 아니라 장애물 회피도 가능
