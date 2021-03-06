## <목차> [1. 제출 모델] [2. 실험 기록] [3. 시사점 및 개선 방향]

## 1. 제출 모델 (코드명: [final_code_0523.ipynb](https://github.com/bunchacha/dacon-competition-credit_score_prediction/blob/main/notebook/%EC%9D%B4%EB%AC%B8%ED%98%95/final_code_0523.ipynb))
```
 1. 데이터 불러오기(Colab)
 2. 라이브러리 설명
 3. 데이터 탐색
    Data feature description
 4. 데이터 전처리
    중복 데이터 처리
    탐색적 분석(EDA) 및 전처리 연습 (중복 데이터 처리 후)
 5. 데이터 전처리 함수 (preprocess)
 6. 모델링 (제출모델 : Customized LGBM(Optuna 튜닝 후))
    TabNet
    Optuna
    Customized (LightGBM)
    Stratified K-fold ensemble
 7. 제출 파일 생성
 ```
 
## 2. 실험 기록
### 주요 이슈사항
(1) 중복 데이터 처리 문제
```
- 모든 피쳐가 동일한 경우 : 중복 데이터 제거
- 신용도(Label)을 제외한, 모든 피쳐가 동일한 경우 : 그대로 분석에 사용함
- 실험을 진행했던 방법 : 1) 전부 삭제, 2) 최빈값으로 처리, 3) 예측값으로 처리, 4) label에 비율값 부여 or label smoothing(딥러닝)
```
(2) 같은 고객 처리 문제
```
- begin_month와 신용도(Label)을 제외한, 모든 피쳐가 동일한 경우를 같은 고객으로 처리함
- 추가로, income_total, family_size, occyp_type 등에서도 변화한 경우 몇 개의 컬럼을 확인함
- 같은 고객 처리 문제를 해결하고, 고객의 과거 이력을 구분하기 위한 파생변수들을 추가함
```
(3) 불균형 데이터 처리 문제
```
- 신용도 별로 데이터 불균형이 존재함 : Stratified K-Fold를 적용함
- 실험을 진행했던 방법 : Imblearn (언더/오버 샘플링)
```
(4) train/test 및 신용도(Label)에 대한 분포 비교
```
- begin_month 등 에서의 분포 차이를 발견함
- 최초 발급 및 발급 초기에는 신용도를 좋게 평가받는 경향이 존재함
- DAYS_BIRTH, DAYS_EMPLOYED, income_total은 신용도와 양의 상관관계를 보임
- adversarial validation을 수행한 결과, AUC 값이 0.5에 가깝게 나옴
```
### 피쳐 엔지니어링
(1) 파생변수 추가
```
# 같은 고객 처리 문제를 해결하고, 고객의 과거 이력을 구분하기 위한 파생변수들을 추가함
- idx : 행 index (분석 시 제거)
- id : 고객 id
- interval_begin_month : 재발급 경과일(월)
- total_begin_month : 최초 발급부터 재발급 경과일(월)
- reissue : 재발급 여부 (0: 신규 발급, 1: 재발급)
- cnt_card : 누적 카드 발급 수
- before_credit : 직전 신용도
- tb_divide_cnt : total_begin_month / cnt_card

# 기타 고려 가능한 파생변수
- income_total 등 연속형 변수의 최소값/최대값/평균값/증감비 등의 집계 데이터
```

(2) 피쳐별 전처리
```
- 변수 제거 : FLAG_MOBIL, phone, email, cnt_card 등 분석에 영향이 거의 없는 변수를 제거함
- 결측치 처리 : occyp_type 결측치에 None 대입
- 이상치 처리 (범주 치환 및 제거) # child_num, family_size는 수치형 변수지만, 비선형적인 특징을 고려하여 범주형 변수처럼 치환과 제거를 진행함
: child_num>5 인 행 제거 후, child_num>=2 인 행을 child_num==2 로 치환함
  family_size<7 인 행 제거 후, family_size>=4 인 행을 family_size==4 로 치환함
  DAYS_EMPLOYED>0 인 행(noise)을 DAY_SEMPLOYED==0 으로 치환함
- 이상치 처리 (수치형 변수)
: Z-Score를 이용해 이상치를 제거함
- 범주형 변수 결합(축소)
: EMPLOYED==0이고, occyp_type=='None'인 경우, occyp_type=='unemployment'로 치환함
  house_type, family_type, begin_month 을 도메인 지식을 바탕으로 범주를 축소함
- 일 단위를 연 단위로 변환 : DAYS_BIRTH, DAYS_EMPLOYED
- 비닝 (수치형 변수 구간화)
: DAYS_BIRTH -> AGE_GROUP (6그룹) 연령대
  DAYS_EMPLOYED -> EMP_GROUP (4그룹) 경력기간
  begin_month -> month_group (2그룹) 카드 발급 경과 기간
 - 범주형 변수 인코딩 : 원핫 인코딩, 라벨 인코딩, catboost 인코딩(data leakage 아닐 시),
  Ordinal 인코딩 : car, reality, edu_type, house_type (도메인 지식을 바탕으로 높은 점수를 부여함)
 - 스케일링 : 딥러닝 모델에서 z-score 정규화(StandardScaler())를 진행함
```

## 3. 시사점 및 개선 방향
### 시사점
```
- 정형 데이터 셋(Tabuler) 위주의 머신러닝 문제는 특성 공학이 핵심임
- 본 경진대회의 경우 도메인 지식이 특히 중요했음
- 불완전한 데이터 처리 방법에 대해 학습을 진행함 (불균형, 중복, 결측치, 이상치 데이터)
- Data leakage를 조심해야 함 (데이콘 경진대회의 경우, 조건이 까다로움)
```
### 개선 방향
```
- Stacking 앙상블 알고리즘의 적용
- 더 많은 파생 변수와 catboost 알고리즘의 적용
- TabNet 모델 같은 경우, 튜닝이 더 필요함
- 우수 사례 : https://www.dacon.io/competitions/official/235713/codeshare/2746?page=1&dtype=recent
- 후처리(post-processing) 작업의 적용
```
