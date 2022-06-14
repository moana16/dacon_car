# -*- coding: utf-8 -*-


import pandas as pd
pd.options.display.max_rows=100
pd.options.display.max_columns=100

# 1. 데이터 적재
train = pd.read_csv('data/train.csv')
test = pd.read_csv('data/test.csv')


# 2. EDA 수행
# - 데이터의 형식 파악
print(train.head())

# - 데이터의 전체 개수 : 1015
# - 데이터의 타입이 다양함 : object, int64
print(train.info())
print(train.shape)

# - 결측 데이터는 존재하지 않음
print(train.isnull().sum())
train.target.head()

# 3. 데이터 전처리

# 오탈자 처리
train[train['year'] < 1900]
train = train[train['year'] > 1900]
train = train.drop('id', axis = 1).reset_index().drop('index', axis = 1).reset_index().rename({'index':'id'}, axis = 'columns')
train.shape


train['title'].value_counts()[:20]

# Brand 상위 션수 생성 후 브랜드 별 카테고리 변수 추가
train['brand'] = train['title'].apply(lambda x : x.split(" ")[0])
train.head()


# 201개의 카테고리에서 41개의 카테고리로 줄어든 것을 알 수 있음
print('title의 unique 카테고리 개수 : ', len(train['title'].value_counts()))
print('brand의 unique 카테고리 개수 : ', len(train['brand'].value_counts()))

# 띄어쓰기, 대소문자 획일화
train['paint'].value_counts()[:20]

import re 

def clean_text(texts): 
    corpus = [] 
    for i in range(0, len(texts)): 
        review = re.sub(r'[@%\\*=()/~#&\+á?\xc3\xa1\-\|\.\:\;\!\-\,\_\~\$\'\"\n\]\[\>\<]', '',texts[i]) #@%*=()/+ 와 같은 문장부호 제거
        review = re.sub(r'\d+','',review)#숫자 제거
        review = review.lower() #소문자 변환
        review = re.sub(r'\s+', ' ', review) #extra space 제거
        review = re.sub(r'<[^>]+>','',review) #Html tags 제거
        review = re.sub(r'\s+', ' ', review) #spaces 제거
        review = re.sub(r"^\s+", '', review) #space from start 제거
        review = re.sub(r'\s+$', '', review) #space from the end 제거
        review = re.sub(r'_', ' ', review) #space from the end 제거
        #review = re.sub(r'l', '', review)
        corpus.append(review) 
        
    return corpus

print('정제 전 brand의 unique 카테고리 개수 : ', len(train['paint'].unique()))

temp = clean_text(train['paint']) #메소드 적용
train['paint'] = temp

print('정제 후 brand의 unique 카테고리 개수 : ', len(train['paint'].unique()))

train['paint'].value_counts()[:20]

# 색상변수 획일화
train['paint'] = train['paint'] = train['paint'].apply(lambda x : 'blue' if x.find('blue') >= 0 else x)
train['paint'] = train['paint'] = train['paint'].apply(lambda x : 'red' if x.find('red') >= 0 else x)
train['paint'] = train['paint'] = train['paint'].apply(lambda x : 'green' if x.find('green') >= 0 else x)
train['paint'] = train['paint'] = train['paint'].apply(lambda x : 'white' if x.find('white') >= 0 else x)
train['paint'] = train['paint'] = train['paint'].apply(lambda x : 'grey' if x.find('grey') >= 0 else x)
train['paint'] = train['paint'] = train['paint'].apply(lambda x : 'grey' if x.find('gery') >= 0 else x)
train['paint'] = train['paint'] = train['paint'].apply(lambda x : 'grey' if x.find('gray') >= 0 else x)
train['paint'] = train['paint'] = train['paint'].apply(lambda x : 'ash' if x.find('ash') >= 0 else x)
train['paint'] = train['paint'] = train['paint'].apply(lambda x : 'brown' if x.find('brown') >= 0 else x)
train['paint'] = train['paint'] = train['paint'].apply(lambda x : 'silver' if x.find('silver') >= 0 else x)
train['paint'] = train['paint'] = train['paint'].apply(lambda x : 'silver' if x.find('sliver') >= 0 else x)
train['paint'] = train['paint'] = train['paint'].apply(lambda x : 'black' if x.find('black') >= 0 else x)
train['paint'] = train['paint'] = train['paint'].apply(lambda x : 'gold' if x.find('gold') >= 0 else x)
train['paint'] = train['paint'] = train['paint'].apply(lambda x : 'wine' if x.find('whine') >= 0 else x)


train['paint'].value_counts()


print('paint의 unique 카테고리 개수 : ', len(train['paint'].value_counts()))


# 라벨인코딩
obj_columns = [cname for cname in train.columns if train[cname].dtype=='object']
num_columns = [cname for cname in train.columns if train[cname].dtype=='int64']
print(obj_columns)

from sklearn.preprocessing import LabelEncoder

# - 각 문자열 컬럼에 대한 인코더를 저장하기 위해서 딕셔너리 생성
dict_encoder = {}

# - 문자열 컬럼의 개수만큼 반복을 수행하며 인코더를 생성 및 학습
for cname in obj_columns : 
    encoder = LabelEncoder()
    encoder.fit(train[cname])
    dict_encoder[cname] = encoder
    
    # 원본 데이터를 라벨 인코딩 결과로 대체
    train[cname] = encoder.transform(train[cname])

print(train[obj_columns].head())
# print(dict_encoder)

# - 라벨 인코딩 적용 결과 확인
print(train.info())




X = train.drop(['id', 'target'], axis = 1) #training 데이터에서 피쳐 추출
y = train.target #training 데이터에서 중고차 가격 추출
X.describe()
print(X.head())
X.info()
print(y.head())


# 4. 데이터 분할
from sklearn.model_selection import train_test_split

data = train.drop('id', axis = 1).copy() #필요없는 id열 삭제
train_data, val_data = train_test_split(data, test_size=0.25, random_state=1) #25프로로 설정
train_data.reset_index(inplace=True) #전처리 과정에서 데이터가 뒤섞이지 않도록 인덱스를 초기화
val_data.reset_index(inplace=True)

train_data_X = train_data.drop(['target', 'index'], axis = 1) #training 데이터에서 피쳐 추출
train_data_y = train_data.target #training 데이터에서 target 추출

val_data_X = val_data.drop(['target', 'index'], axis = 1) #training 데이터에서 피쳐 추출
val_data_y = val_data.target #validation 데이터에서 target 추출








# 5. 각 머신러닝 클래스 별 모델 객체를 생성하고 학습을 진행

from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor

model_rf = RandomForestRegressor(n_estimators=300,
                                     max_depth=3,
                                     subsample=1.0,
                                     criterion='mae',
                                     random_state=11).fit(train_data_X,train_data_y)

model_gb = GradientBoostingRegressor(n_estimators=300,
                                     max_depth=3,
                                     subsample=1.0,
                                     criterion='mae',
                                     random_state=11).fit(train_data_X,train_data_y)



import numpy as np
from sklearn.metrics import mean_squared_error

def nmae(true, pred):

    mae = np.mean(np.abs(true-pred))
    score = mae / np.mean(np.abs(true))
    
    return score

y_hat_rf = model_rf.predict(val_data_X) # y예측
print(f'모델 NMAE: {nmae(val_data_y,y_hat_rf)}')

y_hat_gb = model_gb.predict(val_data_X) # y예측
print(f'모델 NMAE: {nmae(val_data_y,y_hat_gb)}')

y_hat_best = model_best.predict(val_data_X) # y예측
print(f'모델 NMAE: {nmae(val_data_y,y_hat_gb)}')


# 교차검증 데이터 셋을 분할하기 위한 클래스
from sklearn.model_selection import KFold
# 교차검증을 수행할 수 있는 함수
from sklearn.model_selection import cross_val_score

from sklearn.ensemble import GradientBoostingClassifier

X, y = load_iris(return_X_y=True)

print(X.shape)
print(y.shape)

# 전체 데이터를 훈련 및 테스트 세트로 분할
X_train,X_test,y_train,y_test=train_test_split(
    X, y,
    test_size=0.2,
    random_state=1,
    stratify=y)

# 모델의 학습에 사용할 파라메터의 정의
param_grid = {'learning_rate':[0.1, 0.2, 0.3, 1., 0.01],
              'max_depth':[1, 2, 3],
              'subsample':[0.1, 0.2, 0.3, 1., 0.01],
              'n_estimators':[100, 200, 300, 10, 50]}

# 교차 검증 점수를 기반으로 최적의 하이퍼 파라메터를 
# 검색할 수 있는 GridSearchCV 클래스
from sklearn.model_selection import GridSearchCV

# 교차 검증 수행을 위한 데이터 분할 객체
cv=KFold(n_splits=5,shuffle=True,random_state=1)
# 교차 검증에 사용할 기본 머신러닝 모델
base_model=GradientBoostingRegressor(random_state=1)


grid_model = GridSearchCV(estimator=base_model,
                          param_grid=param_grid,
                          cv=cv,
                          n_jobs=-1)
grid_model.fit(X_train,y_train)

# 모든 하이퍼 파라메터를 조합하여 평가한 
# 가장 높은 교차검증 SCORE 값을 반환
print(f'best_score -> {grid_model.best_score_}')
# 가장 높은 교차검증 SCORE 가 어떤 
# 하이퍼 파라메터를 조합했을 때 만들어 졌는지 확인
print(f'best_params -> {grid_model.best_params_}')
# 가장 높은 교차검증 SCORE의 
# 하이퍼 파라메터를 사용하여 생성된 모델 객체를 반환
print(f'best_model -> {grid_model.best_estimator_}')

score = grid_model.score(X_train, y_train)
print(f'SCORE(TRAIN) : {score:.5f}')
score = grid_model.score(X_test, y_test)
print(f'SCORE(TEST) : {score:.5f}')


# training 데이터에서 피쳐 추출
train_X = train.drop(['id', 'target'], axis = 1) 
# training 데이터에서 target 추출
train_y = train.target 


# 모델 학습
model = RandomForestRegressor(n_estimators=300,
                                     max_depth=3,
                                     subsample=1.0,
                                     criterion='mae',
                                     random_state=11)
model.fit(train_X, train_y) 

y_pred = model_gb.predict(train_X)
y_pred[:5]

train_y[:5]
y_hat = model.predict(train_X) # y예측
print(f'모델 NMAE: {nmae(train_y,y_hat)}')


# test 데이터 전처리
test = test.drop('id', axis = 1)


test.head()
test.info()



test['title'].value_counts()[:20]

test['brand'] = test['title'].apply(lambda x : x.split(" ")[0])
test.head()


print('title의 unique 카테고리 개수 : ', len(test['title'].value_counts()))
print('brand의 unique 카테고리 개수 : ', len(test['brand'].value_counts()))

test['paint'].value_counts()[:20]


def clean_text(texts): 
    corpus = [] 
    for i in range(0, len(texts)): 
        review = re.sub(r'[@%\\*=()/~#&\+á?\xc3\xa1\-\|\.\:\;\!\-\,\_\~\$\'\"\n\]\[\>\<]', '',texts[i]) #@%*=()/+ 와 같은 문장부호 제거
        review = re.sub(r'\d+','',review)#숫자 제거
        review = review.lower() #소문자 변환
        review = re.sub(r'\s+', ' ', review) #extra space 제거
        review = re.sub(r'<[^>]+>','',review) #Html tags 제거
        review = re.sub(r'\s+', ' ', review) #spaces 제거
        review = re.sub(r"^\s+", '', review) #space from start 제거
        review = re.sub(r'\s+$', '', review) #space from the end 제거
        review = re.sub(r'_', ' ', review) #space from the end 제거
        #review = re.sub(r'l', '', review)
        corpus.append(review) 
        
    return corpus

print('정제 전 brand의 unique 카테고리 개수 : ', len(test['paint'].unique()))

temp = clean_text(test['paint']) #메소드 적용
test['paint'] = temp

print('정제 후 brand의 unique 카테고리 개수 : ', len(test['paint'].unique()))

test['paint'].value_counts()[:20]
test['paint'] = test['paint'] = test['paint'].apply(lambda x : 'blue' if x.find('blue') >= 0 else x)
test['paint'] = test['paint'] = test['paint'].apply(lambda x : 'red' if x.find('red') >= 0 else x)
test['paint'] = test['paint'] = test['paint'].apply(lambda x : 'green' if x.find('green') >= 0 else x)
test['paint'] = test['paint'] = test['paint'].apply(lambda x : 'white' if x.find('white') >= 0 else x)
test['paint'] = test['paint'] = test['paint'].apply(lambda x : 'grey' if x.find('grey') >= 0 else x)
test['paint'] = test['paint'] = test['paint'].apply(lambda x : 'grey' if x.find('gery') >= 0 else x)
test['paint'] = test['paint'] = test['paint'].apply(lambda x : 'grey' if x.find('gray') >= 0 else x)
test['paint'] = test['paint'] = test['paint'].apply(lambda x : 'ash' if x.find('ash') >= 0 else x)
test['paint'] = test['paint'] = test['paint'].apply(lambda x : 'brown' if x.find('brown') >= 0 else x)
test['paint'] = test['paint'] = test['paint'].apply(lambda x : 'silver' if x.find('silver') >= 0 else x)
test['paint'] = test['paint'] = test['paint'].apply(lambda x : 'silver' if x.find('sliver') >= 0 else x)
test['paint'] = test['paint'] = test['paint'].apply(lambda x : 'black' if x.find('black') >= 0 else x)
test['paint'] = test['paint'] = test['paint'].apply(lambda x : 'gold' if x.find('gold') >= 0 else x)
test['paint'] = test['paint'] = test['paint'].apply(lambda x : 'wine' if x.find('whine') >= 0 else x)

train['paint'].value_counts()


print('paint의 unique 카테고리 개수 : ', len(test['paint'].value_counts()))


test.info()

# 라벨 인코딩
obj_columns = [cname for cname in test.columns if test[cname].dtype=='object']
num_columns = [cname for cname in test.columns if test[cname].dtype=='int64']
print(obj_columns)

from sklearn.preprocessing import LabelEncoder

# - 각 문자열 컬럼에 대한 인코더를 저장하기 위해서 딕셔너리 생성
dict_encoder = {}

# - 문자열 컬럼의 개수만큼 반복을 수행하며 인코더를 생성 및 학습
for cname in obj_columns : 
    encoder = LabelEncoder()
    encoder.fit(test[cname])
    dict_encoder[cname] = encoder
    
    # 원본 데이터를 라벨 인코딩 결과로 대체
    test[cname] = encoder.transform(test[cname])

print(test[obj_columns].head())
# print(dict_encoder)



y_pred = model.predict(test)
y_pred[0:5]


submission = pd.read_csv('data/sample_submission.csv')
submission.head()

# 위에서 구한 예측값을 그대로 넣어줍니다.
submission['target'] = y_pred

# 데이터가 잘 들어갔는지 확인합니다.
submission.head()

submission.to_csv('data/submit1.csv', index=False)

