# Open-Domain Question Answering
![image](https://user-images.githubusercontent.com/86893209/211719468-d335a84f-ce77-4859-86d9-7b36d90c1713.png)
# 1. 프로젝트 개요

 MRC(Machine Reading Comprehension)이란 주어진 지문을 이해하고, 주어진 질의의 답변을 추론하는 태스크이다. 본 ODQA 대회에서 모델은 질문에 관련된 문서를 찾아주는 “Retriever”와 찾아온 문서를 읽고 적절한 답변을 찾아주는 “Reader”로 구성된다.

 데이터는 다음과 같이 구성되어있다.

![Untitled (68)](https://user-images.githubusercontent.com/86893209/211719960-d4f95ae9-fd02-40d3-b186-726e3b65cb58.png)

 Test dataset 중에 리더보드에서 계산이 되는 Public data는 240개, 최종 리더보드의 최종 등수를 위해 계산되는 Private data는 360개로 구성되며 평가지표는 **Exact Match (EM), F1 Score**를 사용한다.

**Exact Match (EM)**: 모델의 예측과, 실제 답이 정확하게 일치할 때만 점수가 주어진다. 즉 모든 질문은 0점 아니면 1점으로 처리된다. 

![Untitled (64)](https://user-images.githubusercontent.com/86893209/211719771-cdb9d274-0dfe-4ba5-b657-d414b503f6d3.png)

**F1 Score**: EM과 다르게 부분 점수를 제공한다. 예를 들어, 정답은 "Barack Obama"지만 예측이 "Obama"일 때, EM의 경우 0점을 받겠지만 F1 Score는 겹치는 단어도 있는 것을 고려해 부분 점수를 받을 수 있다.

![Untitled (65)](https://user-images.githubusercontent.com/86893209/211719997-caf4af64-fdb9-4d7b-8d4f-2a65bfa114d6.png)

# 2. 팀 구성 및 협업툴

### 팀 구성 및 역할

| 김건우_T4017 | github action, hardvoting |
| --- | --- |
| 백단익_T4098 | basecode sweep, BM25 |
| 손용찬_T4108 | Pytorch lightning 재구현, korquad TAPT, Curriculum learning, BM25, 발표 |
| 이재덕_T4163 | PL 이식, 일정 관리, 정보 취합 및 아이디어 제공 |
| 정석희_T4194 | EDA, 외부 데이터셋 추가, Elasticsearch Retrival |

### Notion

 프로젝트 일정 관리를 위해 노션의 칸반보드를 활용하여 각 팀별, 팀원별 태스크 진행 상황을 공유했다. 

- Rule
    - 진행상황 업데이트 결과 칸반보드 및 카카오톡으로 공유
    - DeadLine 설정
    - Progress를 통해 진행사항 공유
    
    ![Untitled (66)](https://user-images.githubusercontent.com/86893209/211720012-3d92bbc6-d2a0-4117-844f-d4be36327433.png)
    

### Git

GitFlow 전략을 활용하여 branch를 관리하였다. 

- Rule
    - develop에 기능 업데이트시, UPDATE.md에 기능 설명 작성하기
    - Pull Reqest : develop에 push or 수정시, 코드리뷰 후 결정
    - Pre-commit
        - CI/CD - black, isort, autoflake → flake8
    - Git hub action
        - Pre-commit : flake8
        - Git Commit Convention 사용

![Untitled (69)](https://user-images.githubusercontent.com/86893209/211720113-75f03069-5d6c-4437-acc7-395c69e0501a.png)

![Untitled (70)](https://user-images.githubusercontent.com/86893209/211720133-52bdaf7b-7ee2-4c42-a6bd-0df92dfda1e9.png)

# **3. 프로젝트 진행 과정**

## TimeLine

![Untitled (67)](https://user-images.githubusercontent.com/86893209/211720039-d1b91e95-0fb4-4854-816c-fbb7c3a449d8.png)

## **1) EDA**

### Reader 데이터 - question, answer, context

### 데이터 개수

Train : 3952 Valid : 240 Test : 600

### 데이터 형태

### Train & Validation dataset

![https://blog.kakaocdn.net/dn/nt3h8/btrVzZFVTCQ/vbieIBR5CsoeUaa7io10A0/img.png](https://blog.kakaocdn.net/dn/nt3h8/btrVzZFVTCQ/vbieIBR5CsoeUaa7io10A0/img.png)

![https://blog.kakaocdn.net/dn/bsnpr1/btrVBdKqFWL/9Mk3mGQnJ3l1fnxTUn8VcK/img.png](https://blog.kakaocdn.net/dn/bsnpr1/btrVBdKqFWL/9Mk3mGQnJ3l1fnxTUn8VcK/img.png)

(상) Train, (하) Validation

### Test dataset

![https://blog.kakaocdn.net/dn/p7uzP/btrVG0QQern/PP5LR5w3Nf3URbTyP9aT00/img.png](https://blog.kakaocdn.net/dn/p7uzP/btrVG0QQern/PP5LR5w3Nf3URbTyP9aT00/img.png)

## 데이터 길이

### Train & Valid

![https://blog.kakaocdn.net/dn/bZEnBw/btrVI6b9YdI/vL4QTTuSNzIrK60ZlFsOP0/img.png](https://blog.kakaocdn.net/dn/bZEnBw/btrVI6b9YdI/vL4QTTuSNzIrK60ZlFsOP0/img.png)

![https://blog.kakaocdn.net/dn/bd6c27/btrVz9BHOJP/3xWcdED239qKWalNqGPjzk/img.png](https://blog.kakaocdn.net/dn/bd6c27/btrVz9BHOJP/3xWcdED239qKWalNqGPjzk/img.png)

![https://blog.kakaocdn.net/dn/cgI0fT/btrVA1pKr79/5jKwUgH37lkcOdTUQDI6cK/img.png](https://blog.kakaocdn.net/dn/cgI0fT/btrVA1pKr79/5jKwUgH37lkcOdTUQDI6cK/img.png)

(좌) Train 통계치, (우) Validation 통계치

- context 평균 900~, 최소 512, 최대 2060
- question 평균 30, 최소 8, 최대 78
- answer 평균 6, 최소 1, 최대 83

### Test

![https://blog.kakaocdn.net/dn/ABplI/btrVBRmMOu3/p1754pir63fUqT7xRoRZT1/img.png](https://blog.kakaocdn.net/dn/ABplI/btrVBRmMOu3/p1754pir63fUqT7xRoRZT1/img.png)

![https://blog.kakaocdn.net/dn/KWvfC/btrVCeB7snO/k6dfCivGUOD92cd93OXK80/img.png](https://blog.kakaocdn.net/dn/KWvfC/btrVCeB7snO/k6dfCivGUOD92cd93OXK80/img.png)

- question 평균 30, 최소 8, 최대 62

### Retrieval 데이터 - Document corpus

### 데이터 개수

- 60613개

### 데이터 형태

![https://blog.kakaocdn.net/dn/ZqTsm/btrVzE93XH2/aN2kkK1kv6Npybf8V5dQl1/img.png](https://blog.kakaocdn.net/dn/ZqTsm/btrVzE93XH2/aN2kkK1kv6Npybf8V5dQl1/img.png)

![https://blog.kakaocdn.net/dn/tVSso/btrVEppQO97/asKcPsBqER2WavqKUaObEk/img.png](https://blog.kakaocdn.net/dn/tVSso/btrVEppQO97/asKcPsBqER2WavqKUaObEk/img.png)

- 평균 584, 최소 46, 최대 46099

## 2) Pytorch Lightning Refactoring

### Base Line code

![https://blog.kakaocdn.net/dn/pFjVw/btrVzJpOph4/UQDjjhXKETNCKl1iJHsvrk/img.png](https://blog.kakaocdn.net/dn/pFjVw/btrVzJpOph4/UQDjjhXKETNCKl1iJHsvrk/img.png)

```bash
base
├─ README.md
├─ arguments.py
├─ assets
│  ├─ dataset.png
│  ├─ mrc.png
│  └─ odqa.png
├─ inference.py
├─ retrieval.py
├─ train.py
├─ trainer_qa.py
└─ utils_qa.py
```

- Nested function과 huggingface의 dataloader, dataset 등 ~~제한된 라이브러리 사용~~
- 실험 편의 증대 및 유지보수를 위해 파이토치 라이트닝으로 리팩토링

## Pytorch Lightning

![https://blog.kakaocdn.net/dn/I0ZCT/btrVF8uEGBJ/TGisoMITq3b749q41IjKWK/img.png](https://blog.kakaocdn.net/dn/I0ZCT/btrVF8uEGBJ/TGisoMITq3b749q41IjKWK/img.png)

```bash
pl
├─ UPDATE.md
├─ config
│  └─ base_config.yaml
├─ datamodule
│  └─ base_data.py
├─ inference.py
├─ main.py
├─ models
│  └─ base_model.py
├─ output
├─ retrievals
│  ├─ BM25.py
│  ├─ base_retrieval.py
│  ├─ elastic_retrieval.py
│  ├─ elastic_setting.py
│  └─ setting.json
├─ sweep.py
├─ train.sh
├─ tune
│  ├─ batch_find.ipynb
│  └─ lr_find.ipynb
├─ utils
│  ├─ data_utils.py
│  └─ util.py
└─ wandb
```

- sweep을 통해 HyperParameter Tuning
- PL Tuner를 사용하여 LR, BatchSize 초기값 세팅

## 3) Base score

![Untitled (71)](https://user-images.githubusercontent.com/86893209/211720528-40112f34-2bff-4ec5-af7f-a515448d716d.png)

- 다양한 모델 실험을 통해, 가장 성능이 좋았던 RoBERTa - Large 모델을 backbone model로 설정

## 4) Retrieval Experiment

- TF-IDF: 문서 내에 단어가 자주 등장하지만 다른 문서에는 단어가 등장하지 않을 때, 단어를 중요하다고 판단 점수를 더 주는 방식
- BM25: TF-IDF의 개념을 바탕으로 문서의 길이까지 고려하여 점수를 매김
    - TF 값에 한계를 지정해두어 일정한 범위를 유지하도록 함
    - 평균적인 문서 길이보다 더 작은 문서에서 단어가 매칭된 경우 그 문서에 대해 가중치 부여
- Elasticsearch: 분산 검색 엔진
    - 텍스트를 여러 단어로 변형하여 검색할 수 있으며 스코어링 알고리즘을 통한 연관도에 따른 정렬 기능 제공.
    - 대량의 데이터에서 빠르고 정확한 검색이 가능하게 만들어 줌
- Dense Retrieval (미사용)
    - 데이터셋 분석 결과 대부분의 정답이 Context에 있음을 확인 및 EM Score를 활용하기에 Term overlap을 정확하게 찾을 수 있는 Sparse Retrieval를 선택

![https://blog.kakaocdn.net/dn/btNzo5/btrVAMfij7Q/6u4qSHrQ1d86nWhBAyArY0/img.png](https://blog.kakaocdn.net/dn/btNzo5/btrVAMfij7Q/6u4qSHrQ1d86nWhBAyArY0/img.png)

RoBERTa-Lage batch16, learning rate 1e-5, epoch 3

- TF-IDF, BM25, Elasticsearch 비교하여 Retrival 실험 진행
- Elasticsearch의 은전한닢 형태소 분석기 기반 한국어형태소 분석기 nori tokenizer, 단어 단위로 한 묶음을 구성하는 single token filter, 복합명사를 분리하고 기존 형태를 보존할 수 있도록 decompound mode는 mixed, 유사도 점수로는 BM25를 사용
- BM25의 성능이 가장 좋은 것으로 확인
    - Elastic search 또한 BM25로 scoring을 하였지만, 세부적인 tokenizer 등에서 성능이 저하 되었을 것이라 추측

## 5-1) Reader Experiment - ****Curriculum Learning¹⁾****

**Result EDA**

- 모델의 Predict²⁾를 분석한 결과, context의 길이가 길수록 question의 길이가 짧을 수록 answer의 길이가 길수록 학습에 더 어려운 경향이 있음을 확인, 논문의 Sample Length 단락 채용
    
    ![Untitled (72)](https://user-images.githubusercontent.com/86893209/211720611-41e6d04b-4917-4db6-bf04-db6e48695646.png)
    
- context가 긴 문장의 경우 모델에 들어가는 정보량이 많아지므로 학습이 어려운 것으로 추론
- **쉬운 것에서 어려운 것 순**으로 Curriculum learning하기 위해, 데이터셋을  **context 길이가 짧은 것에서 긴 것** **순**으로 정렬 후 학습

![Untitled (74)](https://user-images.githubusercontent.com/86893209/211720767-9e4821b0-f51b-43b4-b58d-28572830b0b8.png)

→ 원본 데이터 대비 EM 1.25, F1 2.47 상승

## 5-2) Reader Experiment - 외부 데이터셋 추가, TAPT³⁾

- KorQuAD 1.0: 1,550개의 위키피디아 문서에 대해서 10,649 건의 하위 문서들과 크라우드소싱을 통해 제작한 63,952 개의 질의응답 쌍 데이터 사용
- AI Hub 일반상식: 한국어 위키백과내 주요 문서 15만개에 포함된 지식을 추출하여 객체(entity), 속성(attribute), 값(value)을 갖는 트리플 형식의 데이터 75만개를 구축. WIKI 본문내용과 관련한 질문과 질문에 대응되는 WIKI 본문 내의 정답 쌍 데이터 사용

1. 원본 데이터 + 외부 데이터셋 추가 finetuning

→ 성능 하락

2. 외부 데이터셋 pre-training early stopping 까지, 원본 데이터 finetuning

→ 성능 하락. 외부 데이터셋 대략 12만개 원본 데이터 대략 4천 개, 약 30배 차이. 너무 큰 데이터 규모 차이가 성능 하락을 만든 것으로 추론

3. 외부 데이터셋 pre-training 1 epoch, 원본 데이터 finetuning

![https://blog.kakaocdn.net/dn/9JLBP/btrVF82BVWL/YY2gRVjosFYJSqODcOf40K/img.png](https://blog.kakaocdn.net/dn/9JLBP/btrVF82BVWL/YY2gRVjosFYJSqODcOf40K/img.png)

→ 원본 데이터 대비 EM 2.83, F1 5.26 성능향상

## 5-3) Reader Experiment - 질문 앞에 명사 형용사 관형사 붙히기

![Untitled (75)](https://user-images.githubusercontent.com/86893209/211720902-43440d87-1bc2-4ed0-b084-5fa087f4e44e.png)

- 형태소 분석을 이용하여 질문에 나오는 고유명사, 일반 명사, 형용사, 관형사를 질문(question) 앞에 추가하여 실험
- 질문의 핵심 키워드인 명사 및 형용사 등을 강조하여, 모델이 질문의 의미를 더 잘 파악하도록 함

→ 유의미한 성능 개선을 얻어내지 함

## 6) Ensemble Experiment

![Untitled (76)](https://user-images.githubusercontent.com/86893209/211720922-c88bbaf0-7e84-4e65-af0a-5f5cbcb1666a.png)

- Hard Voting
    - 다수의 Predictions.json 값으로부터 최빈값을 찾아 정답값으로 반환
    - 상위 5개 inference 결과 값을 활용한 hardvoting 결과 EM 기준 4점 정도 향상됨

# 4. 프로젝트 수행 결과

![Untitled (77)](https://user-images.githubusercontent.com/86893209/211720938-fadf4a33-8a8b-48e6-b939-36d79207107f.png)

- Private 2위 (2/14)
- EM : 66.39, F1 : 77.36

# 5. 결론

- **Sparse Retrieval인 BM25를 통한 관련성 높은 Supporting Documents 검색**
- **Curriculum Learning을 통한 Reader 모델 최적화**
- **외부데이터셋 추가 시 Task Adaptive Pre-training 수행**

# 6. 개선사항

### 외부 데이터 아웃라이어 제거

### 스케줄러 적용

- Curriculum learning에 Learning rate scheduler 활용
- Curriculum learning 초반에 "쉬운 것" 학습, 후반에 "어려운 것" 학습 특성 기반
- Learning rate를 초반에 "높게", 후반에 낮게 설정

### SHAP Values 활용

- Context과 Question에서 어떤 text가 output logit에 영향을 주어주었는지 확인
- Retrival 변경, top k 선택 등 다양한 실험 결과 분석

![Untitled (78)](https://user-images.githubusercontent.com/86893209/211720953-a6f24ae7-9769-4e3c-9cc1-ee87a830caad.png)

### Soft voting

- Retrieval ensemble
    - TF-IDF, BM25, Elastic search로부터 얻은 top 40개 context의 합집합 사용
- Reader ensemble
    - Top 5개 model의 logit 값과 score의 가중 평균을 통한 ensemble

# 7. Appendix

1. Campos, D. (2021). Curriculum learning for language modeling. *arXiv preprint arXiv:2108.02170*
2. Klue/RoBERTa-Large, batch_size: 16, LR: 1e-5, Epoch: 3
3. Gururangan, S., Marasović, A., Swayamdipta, S., Lo, K., Beltagy, I., Downey, D., & Smith, N. A. (2020). Don't stop pretraining: adapt language models to domains and tasks. *arXiv preprint arXiv:2004.10964*
