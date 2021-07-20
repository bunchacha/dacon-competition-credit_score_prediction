## 1. 제출 모델 (코드명: final_code_0523.ipynb.ipynb)
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
 
 
## 2. 실험 기록
### 주요 이슈사항
(1) 중복 데이터 처리 문제
- 모든 피쳐가 동일한 경우
- 신용도(Label)을 제외한, 모든 피쳐가 동일한 경우

(2) 같은 고객 처리 문제
- begin_month와 신용도(Label)을 제외한, 모든 피쳐가 동일한 경우

(3) 불균형 데이터 처리 문제

(4) train/test 분포 비교

### 피쳐 엔지니어링



