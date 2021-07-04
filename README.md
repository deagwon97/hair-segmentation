# 1. directory 구성

### 1) 학습 및 추론 데이터

 학습 및 추론 데이터는 테스트 서버의 해당 위치에 저장되어 있다고 가정하고 하드코딩 되어있습니다.
```
$/DATA/Final_DATA/
|
|-- task02_test
|    |-- images
|    |-- sample_submission.json
|-- task02_train
     |-- images
     |-- labels.json
```
### 2) 프로젝트 폴더 구성
```
${PROJECT}
|--README.md
|-- data/
|        |-- mask/ # train.py를 통해 생성
|
|-- model/
|    |-- first_model.pth
|    |-- second_model.pth
|
|-- modules/
|    |-- dataset.py
|    |-- gen_mask.py
|    |-- train_fold.py
|    |-- utils.py
|
|-- train.py
|-- predict.py
|-- output/
|     |-- submission.json
|
|-- Dockerfile
|-- build_image.sh
|-- access.sh
```
## 2. 실행

### 1) nvidia-docker setting

테스트 환경에 nvidia docker가 setting되어 있어야 합니다.

```bash
# setting up docker
{PROJECT}$ curl https://get.docker.com | sh \
  && sudo systemctl --now enable docker

# setting up nvidia container toolkit
{PROJECT}$ distribution=$(. /etc/os-release;echo $ID$VERSION_ID) \
   && curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add - \
   && curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list

{PROJECT}$ apt-get update
{PROJECT}$ apt-get install -y nvidia-docker2
systemctl restart docker
```

### 2) 학습 및 추론

build_image를 통해서 docker image를 build하고 access.sh를 통해 container를 실행 및 접근합니다.

실행된 contanier의 /workspace에서 train.py와 predict.py를 실행합니다.

```bash
{PROJECT}$ sh build_image.sh

{PROJECT}$ sh access.sh

/workspace# python train.py

/workspace# python predict.py
```

## 3. 상세

### 1) /data

 대회의 label데이터는 json format으로 되어있습니다. modules/gen_mask.py를 통해서 lables.json을 .jpg 포맷으로 변환하여 /data/mask 위치에 저장합니다.

### 2) /model

 저희 jayeonsoft 팀에서는 2개의 모델을 통해서 추론을 진행합니다. 우선 [train.py](http://train.py) 및 modules/train_fold.py를 통해서 정의된 방법을 통해 first_model.pth를 생성합니다. 

 이렇게 생성된 모델을 통해서 전체 학습 데이터의 iou score를 계산합니다. 이렇게 계산된 결과를 통해 iou score가 낮은 약 10%의 데이터를 제거합니다.

제거된 데이터를 통해서  second_model.pth를 생성합니다. 

/model이라는 directory에는 이렇게 생성된 두개의 first_model.pth와 second_mode.pth가 저장되어 있습니다.

### 3) /output

predict.py로 생성되는 최종 제출파일 submission.jason 파일이 저장되는 directory 입니다.

### 4) /modules

1. dataset.py

    학습에 사용되는 HairDataset, first_model.pth로 학습데이터 전체의 점수를 계산하는데 사용되는 HairDatasetName가 정의되어있습니다.

2. gen_mask.py

    labels.json을 /data/mask 폴더 속 .jpg에 저장합니다.

3. train_fold.py

    segmentation_models_pytorch, pytorch, albumentation을 통해서 하나의 폴드(1/5)를 학습하는 일련의 프로세스가 run()함수에 정의되어있습니다. 학습할 이미지들의 이름을 list형태로 담고 있는 trainlist: list, 학습에 관련된 정보를 담고있는 CONFIG: dict, 모델의 이름을 정의할 model_name: str를 인자로 받습니다.

    학습데이터가 존재하는 /DATA/Final_DATA/task02_train/경로가 하드코딩되어있습니다.

4. utils.py

    학습 및 추론에서 사용되는 코드가 정의되어있습니다.

    gen_clendf()함수는 전체 학습 데이터의 iou score를 계산합니다. 이렇게 계산된 결과를 통해 iou score가 낮은 약 10%의 데이터가 제거된 clean_df를 반환합니다.