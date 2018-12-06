#dummy for categorical variable
import pandas as pd

#drop attribute 'Id'
train=pd.read_csv('../Data/train.csv',na_values=['NaN'])
train=train.drop(['Id'],axis=1)
test=pd.read_csv('../Data/test.csv',na_values=['NaN'])
test=test.drop(['Id'],axis=1)

dataset=pd.concat([train,test],sort=False,keys=['train','test'])

#drop top 5 missing value attributes
#dataset.drop(['Id'],axis=1)
dataset=dataset.drop(['PoolQC'],axis=1)
dataset=dataset.drop(['MiscFeature'],axis=1)
dataset=dataset.drop(['Alley'],axis=1)
dataset=dataset.drop(['Fence'],axis=1)
dataset=dataset.drop(['LotFrontage'],axis=1)

#fill missing value
dataset=dataset.fillna(dataset.mean())

dataset=dataset.fillna(dataset.mode().iloc[0])

#convert categorical to dummy variables
dataset = pd.get_dummies(dataset)
tmp1= dataset.loc['test']
tmp1.to_csv('dummytest.csv',sep='\t',index=False)
tmp2= dataset.loc['train']
tmp2.to_csv('dummytrain.csv',sep='\t',index=False)
