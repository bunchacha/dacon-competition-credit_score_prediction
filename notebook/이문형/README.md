## 1. 제출 모델 (코드명: [final_code_0523.ipynb](https://github.com/bunchacha/dacon-competition-credit_score_prediction/blob/main/notebook/%EC%9D%B4%EB%AC%B8%ED%98%95/final_code_0523.ipynb))
```
 1. 데이터 불러오기(`**`Colab`**`)
 2. 라이브러리 설명
 3. 데이터 탐색
    Data feature description
 4. 데이터 전처리
    중복 데이터 처리
    탐색적 분석(EDA) 및 전처리 연습 (중복 데이터 처리 후)
 5. 데이터 전처리 함수 (preprocess)
 6. 모델링 (제출모델 : Customized LGBM(Optuna 튜닝 후))
    TabNet
    **Optuna
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
```
(3) 불균형 데이터 처리 문제
```
- 신용도 별로 데이터 불균형이 존재함 : Stratified K-Fold를 적용함
- 실험을 진행했던 방법 : Imblearn (언더/오버 샘플링)
```
(4) train/test 및 신용도(Label)에 대한 분포 비교
```
- begin_month 등 에서의 분포 차이를 발견함
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

# 기타 고려 가능한 파생변수
- 수입 등 연속형 변수의 최소값/최대값/평균값/증감비 등의 집계 데이터
```

(2) 피쳐별 전처리
