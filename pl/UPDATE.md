# UPDATE
## version 1.5.0
* BM-25, ElasticSearch
  * Retrievals 추가, pl_inference 추가
* Shap 추가 
* Elastic 환경 설정 방법

1. 터미널 창에 아래 실행
~~~
# sudo 설치
apt-get update && apt-get -y install sudo

# 패키지 색인을 업데이트
sudo apt update
# HTTPS를 통해 리포지토리에 액세스하는 데 필요한 apt-transport-https 패키지를 설치
sudo apt install apt-transport-https

# OpenJDK 8 설치
sudo apt install openjdk-8-jdk
java -version  # openjdk version "1.8.0_191"

# OpenPGP 암호화 툴 설치
apt-get install -y gnupg2

# Elasticsearch 저장소의 GPG key를 사용해 설치 (GPG key를 암호화/복호화 프로그램이라고 이해하고 넘김)
# 리눅스 배포판에는 기본적으로 GPG가 설치되어 있음
# OK가 반환되어야 함 (맨 뒤 add까지 명령어로 입력해야 함)
wget -qO - https://artifacts.elastic.co/GPG-KEY-elasticsearch | sudo apt-key add -  

# Elasticsearch 저장소를 시스템에 추가
echo "deb https://artifacts.elastic.co/packages/7.x/apt stable main" | sudo tee -a /etc/apt/sources.list.d/elastic-7.x.list

# 이제 진짜 elasticsearch 설치

sudo apt update
sudo apt install elasticsearch  # elasticsearch 7.10.0 설치됨(7.x 버전이면 작동할듯)

# elasticsearch 시작(설치 완료 후 자동으로 시작되지 않음)
service elasticsearch start

# 경로 이동해서 nori 형태소분석기 설치
cd /usr/share/elasticsearch
bin/elasticsearch-plugin install analysis-nori

# elasticsearch 재시작 (형태소분석기 설치 후 재시작이 필수적!)
service elasticsearch restart

# curl 명령어 설치
sudo install curl

# Elasticsearch가 실행 중인지 확인
curl "localhost:9200"
~~~
2. `install_requierments.sh` 실행하여 elasticsearch 설치
3. retrievals 폴더에서 `python elastic_setting.py` 실행
4. config 파일에서 `retrieval: elastic` 설정 확인 후 `inference.py` 실행

## version 1.4.0
* Tune 
  * lr_find, batch_find 추가
  * batch_find는 오류가 있음
  * lr_find는 정상동작
## version 1.3.1
* base_model fix
  * test_step의 해당 코드 변경
  ~~~python
  prediction = (start_logits, end_logits)
  return {"prediction": prediction, "id": id}
  ~~~
 * 해당 코드 validation과 맞춤

## version 1.3.0
* EDA 추가
  * 데이터 개수, 형태, 길이 확인
  * 외부데이터셋 위와 동일하게 EDA 진행
* question 앞에 명사, 형용사, 관형사 추가

## version 1.2.1 (12.23)
* sweep_config.yaml에 sweep parameter 추가

## version 1.2.0 (12.22)
* PL Sweep 추가
  * default가 sweep_config.yaml이기에 해당 파일만 수정하면 가능
  * python sweep.py로 실행

## version 1.1.0 (12.22)
* PL 이식
  * predict를 제외하고 이식 완료(추가 예정)
  * 기존과 동일하게 train.sh을 통해서 멀티 실행 가능
  * 사용법은 sts, re와 같음

## BASIC README
* 기존 코드 및 README.md -> base 폴더
* install 폴더의 install_requirements.sh을 통해 requirements 설치 가능
* PL 폴더에 모듈화된 코드로 기존 코드를 이식
* VSCode markdown all in one 확장 프로그램을 통해 마크다운 보기 가능
* Live Share을 통해 코딩 환경 공유 가능
* Commit Convention 잘 지키기
  * 노션 Project Rule에 Convention 참고 링크 참조
* Git Flow branch strategy 사용
  * 노션 참조
## 소개

> Update된 기능을 소개

### 기능들
양식

* 기능 이름 (만든사람)
  * 기능 설명
  * 기능 코드 위치
  * 예상 충돌 및 해결 방안
