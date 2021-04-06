## VCD Baseline

- 비디오 복사 검출을 위한 baseline 코드
- 데이터셋에 대한 수정된 annotation을 제공


#### Usage
```bash
git clone
# edit docker-compose.yml - volumes, port .. 
vim docker-compose.yml
docker-compose up
```

#### docker-compose.yml
- `{container name}` : 도커컨테이너의 이름
- 포트 `{host port}:{local port}` : tensorboard(6006) 및 ssh(22) 포트 open을 추천함
- 볼륨 `{host path}:{container path}` : host의 특정경로를 mount, 데이터셋을 컨테이너로 복사할수 없기 때문에 hdd나 nas를 마운트할것을 추천함

```
version: '2.3'

services:
  main:
    container_name: "{container name}"
    build:
      context: ./
      dockerfile: Dockerfile
    runtime: nvidia
    restart: always
    environment:
      - PYTHONPATH=/workspace
    volumes:
      - "{host path}:{container path}"
    ports:
      - "{host port}:{container port}"
    ipc: host
    stdin_open: true
    tty: true
```
#### Dataset settings
- `mv /workspace/VCD/dataset ${DATASET ROOT}`
- `${DATASET ROOT}`를 다음과 같이 셋팅, 데이터셋의 비디오 및 프레임등이 저장되기 때문에 충분히 여유있는 storage를 확보해야한다.
<p align="left"><img src="https://i.imgur.com/Sww1J7d.jpg"; height="200px"></p>


#### Script

##### A-DecodeVideo.py
- 비디오 디코딩 및 메타데이터 추출 script
```bash
usage: A-DecodeVideo.py [-h] --root ROOT
                        [--video_dir VIDEO_DIR]
                        [--frame_dir FRAME_DIR] [--meta_dir META_DIR]
                        [--fps FPS]
```
| option | required | 설명 | default|
|--------|--------|--------|--------|
|   --root   | True |  데이터셋의 root 디렉토리 | |
|   --video_dir   | False |  root 디렉토리 내의 비디오가 저장된 폴더  | `videos` |
|   --frame_dir   | False |   root 디렉토리 내의 프레임를 저장할 폴더 |  `frames` |
|   --meta_dir   | False |   root 디렉토리 내의 메타데이터를 저장할 폴더  | `meta` |
|   --fps   | False |   디코딩 레이트  | 1 |
|   --log   | False|  로그파일 경로  |  `extract_frame.log`|
- 프레임 - `{root}/{frame_dir}`
- 비디오 별 프레임 수 - `{root}/{meta_dir}/frames.json`
- 메타데이터 - `{root}/{meta_dir}/metadata.json`
- 실행 후 쉘 출력이 이상해 지는 경우 `reset ` 명령어 실행
- example
	- `python A-DecodeVideo.py --root ${DATASET_DIR}/CC_WEB`
	- 실행시 CC_WEB 데이터셋을 1fps 로 디코딩, 아래와 같은 디렉토리 생성
```bash
CC_WEB
├── GT
├── videos
│	├── xxx.flv 
│    └── yyy.flv
├── frames
│   ├── xxx.flv
│   │   ├── 000001.jpg
│   │   ├── ...
│   │   └── yyyyyy.jpg    
│   └── yyy.flv
│       ├── 000001.jpg
│       ├── ...
│       └── zzzzzz.jpg
├── meta
│   ├── metadata.json
│   └── frames.json
	```

#### B-ExtractFrameFeature.py
- 프레임에 대한 cnn Feature 추출 script
- 비디오 단위로 frame feature를 저장하며 데이터셋의 `videos`와 동일한 디렉토리 구조를 구성
```bash
usage: B-ExtractFrameFeuture.py [-h] --model {MobileNet_AVG,Resnet50_RMAC} \
                                [--ckpt CKPT] \
								--dataset {VCDB,CC_WEB,FIVR,NSFW} \
								--dataset_root DATASET_ROOT \
								--feature_path FEATURE_PATH \
                                [--batch BATCH] \
								[--worker WORKER]
```
| option | required | 설명 | default|
|--------|--------|--------|--------|
|   --model   | True |  CNN model |  |
|   --ckpt   | False |  model checkpoint  |  |
|   --dataset   | True |   데이터셋 | |
|   --dataset_root   | True |   데이터셋의 root 디렉토리  |  |
|   --feature_path   | True |   frame feature를 저장할 디렉토리  |  |
|   --batch   | False|  batch size  |  256|
|   --worker   | False|  num of worker | 4 |
- example
	- `python B-ExtractFrameFeuture.py --model mobilenet-avg --dataset cc_web --dataset_root {path-to-dataset}/CC_WEB --feature_dir {path-to-features}/features --batch 256 --worker 4` 
	- `mobilenet-avg` 모델을 이용해 `CC_WEB` 데이터셋의 frame feature를 추출
```bash
CC_WEB
├── features
│   ├── xxx.flv.pth
│   └── yyy.flv.pth
 ...
```


#### C-ExtractSegmentFeatureWithPooling.py.py
- 세그먼트에 대한 딥러닝 Featrue 추출 script
- 프레임 Feature를 입력으로 하는 딥러닝 모델
```bash
usage: C-ExtractSegmentFeatureWithPooling.py [-h] --model {Segment_MaxPool,Segment_AvgPool}
                                  --frame_feature_path FRAME_FEATURE_PATH
                                  --segment_feature_path SEGMENT_FEATURE_PATH
                                  --count COUNT
```
| option | required | 설명 | default|
|--------|--------|--------|--------|
|   --model   | True |  segment feature 추출 모델 | |
|   --frame_feature_path   | True |  frame feature가 저장된 디렉토리  |  |
|   --segment_feature_path   | True |   segment feature를 저장할 디렉토리 | |
|   --count   | False | 융합할 프레임 feature 개수 | 5 |
- example
	- `python C-ExtractSegmentFeatureWithPooling.py --model max-pool --frame_feature_dir {path-to-frame features} --segment_feature_dir {path-to-segment features} --count 5 ` 
	- 프레임 feature 를 5개를 max-pooling 모델로 융합하여 세그먼트 feature를 생성

#### D-ImageRetrieval_VCDB.py
- VCDB 프레임에 대한 이미지 검색 script
- 정답 프레임의 평균 rank/distance, 정답 세그먼트의 평균 rank/distance 출력
```bash
usage: D-ImageRetrieval_VCDB.py extract [-h] --vcdb_root VCDB_ROOT
                                        [--chunk CHUNK] [--margin MARGIN]
                                        [--model {MobileNet_AVG,Resnet50_RMAC}]
                                        [--ckpt CKPT] [--batch BATCH]
                                        [--worker WORKER]
```
```bash
usage: D-ImageRetrieval_VCDB.py load [-h] --vcdb_root VCDB_ROOT
										--feature_path FEATURE_PATH
										[--chunk CHUNK] [--margin MARGIN]
```
| action |  설명 |
|--------|--------|
|extract| Extract features and Evaluate VCDB frame ranking|
|load| Load features and Evaluate VCDB frame ranking|

 | option | required | 설명 | default |
|--------|--------|--------|--------|
|   --vcdb_root   | True |  vcdb의 root 디렉토리  |  |
|   --chunk   | False |    | 1000 |
|   --margin   | False |  | 1 |
|   --model   | True |  CNN model |  |
|   --ckpt   | False |  model checkpoint  |  |
|   --batch   | False|  batch size  |  256|
|   --worker   | False|  num of worker | 4 |
|   --feature_path   | True | frame feature가 저장된 디렉토리 | |
- example
	- `python E-ImageRetrieval_VCDB.py load --feature_path {path-to-feature} --vcdb_root {vcdb-root-directory}`
	- {path-to-feature}에 저장된 VCDB의 frame feature를 이용하여 정답 프레임의 rank 및 distance 출력
	- `python D-ImageRetrieval_VCDB.py extract --model MobileNet_AVG --vcdb_root {vcdb-root-directory}`
	- `MobileNet_AVG`모델의 VCDB frame feature를 추출 및 정답 프레임의 rank 및 distance 출력

#### E-PartialCopyDetection_VCDB.py
- VCDB 에 대한 partial copy 검출 script
```bash
usage: E-PartialCopyDetection_VCDB.py [-h] --vcdb_root VCDB_ROOT
                                      --feature_path FEATURE_PATH
                                      [--topk TOPK]
                                      [--feature_intv FEATURE_INTV]
                                      [--window WINDOW] [--path_thr PATH_THR]
                                      [--score_thr SCORE_THR]
```
| option | required | 설명 | default|
|--------|--------|--------|--------|
|   --vcdb_root   | True |  VCDB 데이터셋의 root 디렉토리 | |
|   --feature_path   | True |  feature가 저장된 디렉토리  |  |
|   --feature_intv   | False |   단일 feature가 나타내는 세그먼트의 길이  | 1 |
|   --topk   | False |   Temporal network의 topk | 50 |
|   --window   | False |   Temporal network의 window 크기 |5 |
|   --path_thr   | False |   Temporal network의 path threshold |3 |
|   --score_thr   | False |   Temporal network의 score threshold | -1 |
- example
	- `python E-PartialCopyDetection_VCDB.py --vcdb_root /hdd/ms/MLVD/VCDB-core --feature_path {path-to-feature}/mobilenet-avg/pretrained/frame-features --topk 50 --feature_intv 1 --window 5 --path_thr 5 --score_thr -1`
	- Mobilenet-avg 모델의 frame feature를 이용하여 부분 복사 검출 수행

	- `python E-PartialCopyDetection_VCDB.py --vcdb_root /hdd/ms/MLVD/VCDB-core --feature_path {path-to-feature}/mobilenet-avg/pretrained/max-pool-5 --topk 100 --feature_intv 5 --window 10 --path_thr 5 --score_thr -1`
	- Mobilenet-avg 모델의 segment feature를 이용하여 부분 복사 검출 수행

#### F-GenerateFrameTriplets.py
- FIVR 데이터셋에 대한 프레임 Triplet 생성 script
- 사전에 구한 7000개의 positive 프레임 쌍`fivr_1fps_positve.csv`을 이용하여 Triplet을 생성
```bash
usage: D-GenerateFrameTriplets.py [-h] --fivr_root FIVR_ROOT
                                  --feature_path FEATURE_PATH
								  [--triplet_csv TRIPLETS_CSV]
                                  [--negative NEGATIVE]
								  [--margin MARGIN] [--topk TOPK] 
```
| option | required | 설명 | default|
|--------|--------|--------|--------|
|   --fivr_root   | True |  fivr의 root 디렉토리  |  |
|   --feature_path   | True |  feature가 저장된 경로  | |
|   --triplet_csv   | False |  | fivr_triplet.csv |
|   --negative   | False | 샘플링에 사용할 negative 비디오의 개수| 10000 |
|   --margin   | False | negative distance 마진 | 0.3 |
|   --topk   | False | 각 pair가 갖는 negative 프레임의 최대 개수  | 10 |
- example
    - `python F-GenerateFrameTriplets.py --fivr_root {path-to-fivr dataset} --feature_path {path-to-feature} --triplet_csv`
	- {path-to-feature}에 저장된 FIVR feature를 이용, FIVR 에 대한 프레임 Triplet을 생성 

#### G-TrainFrameModel.py
- FIVR으로 생성한 프레임 Triplet을 이용, CNN 모델의 학습 script 
```bash
usage: G-TrainFrameModel.py [-h] --model {MobileNet_AVG,Resnet50_RMAC}
                            [--ckpt CKPT]
							--triplet_csv TRIPLET_CSV
                            --fivr_root FIVR_ROOT
							--vcdb_root VCDB_ROOT
                            [-lr LEARNING_RATE] [-wd WEIGHT_DECAY] [-m MARGIN]
                            [-e EPOCH] [-b BATCH] [-o OPTIM] [--worker WORKER]
                            [-l LOG_DIR] [-c COMMENT]
```
| option | required | 설명 | default|
|--------|--------|--------|--------|
|   --model   | True |    |  |
|   --ckpt   | False |    | |
|   --triplet_csv   | True |  | fivr_triplet.csv |
|   --fivr_root   | True | fivr의 root 디렉토리|  |
|   --vcdb_root   | True | vcdb의 root 디렉토리|  |
|   -lr, --learning_rate   | False | learning_rate | 1e-4 |
|   -wd, --weight_decay   | False | weight_decay  | 0 |
|   -m, --margin  | False | triplet margin | 0.3 |
|   -e, --epoch   | False | epoch | 100 |
|   -b, --batch   | False |  | 64 |
|   -w, --worker  | False |  | 4 |
|   -o, --optim   | False |  | radam |
|   -l, --log_dir   | False |  | ./log |
|   -c, --comment   | False |  | 10 |
- example
	- `python G-TrainFrameModel.py --model MobileNet_AVG --triplet_csv {path-to-triplet} --fivr_root {path-to-fivr dataset} --vcdb_root {path-to-vcdb}`
