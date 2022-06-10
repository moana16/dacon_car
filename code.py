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


train[train['year'] < 1900]
train = train[train['year'] > 1900]
train = train.drop('id', axis = 1).reset_index().drop('index', axis = 1).reset_index().rename({'index':'id'}, axis = 'columns')
train.shape


train['title'].value_counts()[:20]

train['brand'] = train['title'].apply(lambda x : x.split(" ")[0])
train.head()


print('title의 unique 카테고리 개수 : ', len(train['title'].value_counts()))
print('brand의 unique 카테고리 개수 : ', len(train['brand'].value_counts()))

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




from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor



# 5. 각 머신러닝 클래스 별 모델 객체를 생성하고 학습을 진행



# 앙상블을 구현하기 위한 내부 모델의 클래스 로딩
model_rf = RandomForestRegressor(bootstrap=True, ccp_alpha=0.0, criterion='mse',
                       max_depth=None, max_features='auto', max_leaf_nodes=None,
                       max_samples=None, min_impurity_decrease=0.0,
                       min_impurity_split=None, min_samples_leaf=1,
                       min_samples_split=2, min_weight_fraction_leaf=0.0,
                       n_estimators=100, n_jobs=-1, oob_score=False,
                       random_state=42, verbose=0, warm_start=False).fit(train_data_X,train_data_y)

model_gb = GradientBoostingRegressor(n_estimators=300,
                                     max_depth=3,
                                     subsample=1.0,
                                     criterion='mae',
                                     random_state=11).fit(train_data_X,train_data_y)

model_best = GradientBoostingRegressor(alpha=0.9, ccp_alpha=0.0, criterion='friedman_mse',
                           init=None, learning_rate=0.1, loss='ls', max_depth=3,
                           max_features=None, max_leaf_nodes=None,
                           min_impurity_decrease=0.0, min_impurity_split=None,
                           min_samples_leaf=1, min_samples_split=2,
                           min_weight_fraction_leaf=0.0, n_estimators=100,
                           n_iter_no_change=None, presort='deprecated',
                           random_state=42, subsample=1.0, tol=0.0001,
                           validation_fraction=0.1, verbose=0, warm_start=False).fit(train_data_X,train_data_y)


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





train_X = train.drop(['id', 'target'], axis = 1) #training 데이터에서 피쳐 추출
train_y = train.target #training 데이터에서 target 추출



model = RandomForestRegressor(bootstrap=True, ccp_alpha=0.0, criterion='mse',
                       max_depth=None, max_features='auto', max_leaf_nodes=None,
                       max_samples=None, min_impurity_decrease=0.0,
                       min_impurity_split=None, min_samples_leaf=1,
                       min_samples_split=2, min_weight_fraction_leaf=0.0,
                       n_estimators=100, n_jobs=-1, oob_score=False,
                       random_state=42, verbose=0, warm_start=False)
model.fit(train_X, train_y) # 모델 학습

y_pred = model_gb.predict(train_X)
y_pred[:5]

train_y[:5]
y_hat = model.predict(train_X) # y예측
print(f'모델 NMAE: {nmae(train_y,y_hat)}')

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

# 테스트 데이터 전처리
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

