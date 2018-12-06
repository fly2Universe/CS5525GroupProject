#PCA
import numpy as np
from sklearn.decomposition import PCA
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
from sklearn.preprocessing import MinMaxScaler
from sklearn import decomposition


dataset=pd.read_csv('dummytrain.csv',na_values=['NaN'],sep='\t')
subdata=dataset[['SalePrice','BedroomAbvGr','YearBuilt',
        'GrLivArea','TotalBsmtSF','OverallQual','FullBath','GarageArea']]

#Scaling
ss = MinMaxScaler()
subdata=ss.fit_transform(subdata)


#PCA
pca = decomposition.PCA(n_components=4)
pc=pca.fit_transform(subdata)
pc_df=pd.DataFrame(data=pc,columns=['PC1','PC2','PC3','PC4'])

df = pd.DataFrame({'var':pca.explained_variance_ratio_,
             'PC':['PC1','PC2','PC3','PC4']})

columns=['PC1','PC2','PC3','PC4']
sns.barplot(x='PC',y='var',data=df)
plt.xlabel('Principal Components')
plt.ylabel('Variance Explained')
plt.savefig('../Figure/4Report/PCA.png')

#Clustering

sns.lmplot(x='PC1',y='PC2',data=pc_df,fit_reg=False,legend=True,
            scatter_kws={"s":80})
plt.show()
