# 데이콘 경진대회
## 개요 : 신용카드 사용자 연체 예측 AI 경진대회 | 월간 데이콘 14 | 금융 | 정형 | Logloss
- 팀명 : TEAM3 (남탕, )
- 팀원 : 이문형😎, 이종섭😁, 안준용😍, 안동현😒, 김태용😘 
- 기간 : 2021.04.05 ~ 2021.05.24 17:59
- 순위 : 
  Public score 기준 - Logloss score 0.72151 갱신, 343위/1428팀 (+ 상위 24% 달성)
  Private score 기준 - Logloss score 0.68963 갱신, 250위/1428팀 (+ 상위 17% 달성)
- 링크 : [https://www.dacon.io/competitions/official/235713/overview/description](https://www.dacon.io/competitions/official/235713/overview/description)


## 1. 제출 모델
**notebook 폴더에 개별적으로 정리했습니다.**
- [안동현](https://github.com/bunchacha/dacon-competition-credit_score_prediction/tree/main/notebook/%EC%95%88%EB%8F%99%ED%98%84) (코드명 : [EDA _ 1st_Feature_Engineering.ipynb](https://github.com/bunchacha/dacon-competition-credit_score_prediction/blob/main/notebook/%EC%95%88%EB%8F%99%ED%98%84/EDA%20_%201st_Feature_Engineering.ipynb), [Final_Feature_Engineering_Modeling](https://github.com/bunchacha/dacon-competition-credit_score_prediction/blob/main/notebook/%EC%95%88%EB%8F%99%ED%98%84/Final_Feature_Engineering_Modeling.ipynb))
- [이문형](https://github.com/bunchacha/dacon-competition-credit_score_prediction/tree/main/notebook/%EC%9D%B4%EB%AC%B8%ED%98%95) (코드명 : [final_code_0523.ipynb](https://github.com/bunchacha/dacon-competition-credit_score_prediction/blob/main/notebook/%EC%9D%B4%EB%AC%B8%ED%98%95/final_code_0523.ipynb))
- [이종섭](https://github.com/bunchacha/dacon-competition-credit_score_prediction/tree/main/notebook/%EC%9D%B4%EC%A2%85%EC%84%AD) (코드명 : )
- 
- 


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
#### 탐색적 분석, 피쳐 엔지니어링, 모델링 부분에 대해서는, notebook 폴더에 개별적으로 정리했습니다.


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
- 고객을 구별해주는 고객 ID 변수를 만들어 놓고, 활용하지 못했음 -> ID 변수 활용
- Stacking 앙상블 알고리즘의 적용
- 더 많은 파생 변수와 catboost 알고리즘의 적용 
  (범주형 변수가 많았기 때문에 Catboost와 같은 더 좋은 모델을 찾아서 활용하면 좋은 결과를 가져갈 수 있었을 것으로 사료됨)
- 모델링에 따른 적절한 하이퍼 파라미터 선정이 필요함
  (특히, TabNet 모델 같은 경우에는 튜닝이 더 필요함)
- 우수 사례 : https://www.dacon.io/competitions/official/235713/codeshare/2746?page=1&dtype=recent
- 후처리(post-processing) 작업의 적용
```


## 4. (참고)캐글 경진대회 spooky author identification
[https://www.kaggle.com/c/spooky-author-identification](https://www.kaggle.com/c/spooky-author-identification)

참고한 코드 : 

## (참고) 신용카드 사용자 연체 예측 AI 경진대회

참고한 코드 : 

## (참고)논문 spooky author identification
[https://www.kaggle.com/c/spooky-author-identification](https://www.kaggle.com/c/spooky-author-identification)

참고한 코드 : 
