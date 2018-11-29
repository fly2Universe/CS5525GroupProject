import numpy as np
import pandas as pd;
import matplotlib.pyplot as plt
import seaborn as sns

dataset=pd.read_csv('../Data/train.csv',na_values=['NaN'])


#histogram of integer attributes
numerics = ['int','float']

numeric_df = dataset.select_dtypes(include=numerics)
"""
for column in numeric_df:
    plt.hist(numeric_df[column].dropna(),color='green')
    plt.xlabel(column)
    plt.ylabel('Frequency')
    plt.title('Historgram of '+column)
    plt.grid(False)
    tmp='../Figure/NumericHist/'+column+'.png'
    plt.savefig(tmp)
    plt.clf()


#bar plot of categorical attributes
category = dataset.select_dtypes(exclude=numerics)

for column in category:
    ax = dataset[column].value_counts().plot(kind='bar',alpha=0.75,rot=40)
    ax.set_ylabel('Count')
    plt.title('Barplot of '+ column)
    tmp = '../Figure/CategoryBar/'+column+'.png'
    plt.savefig(tmp)
    plt.clf()

#scatter plot of numerical attributes and house Sale Price
for column in numeric_df:
    dataset.plot.scatter(column,'SalePrice',color='green')
    plt.title('Scatter plot of '+ column + 'Vs SalePrice')
    tmp = '../Figure/ScatterPrice/'+column+'.png'
    plt.savefig(tmp)
    plt.clf()

#representative scatter plots for report
fig, axes = plt.subplots(2,2)
attr=['GrLivArea','YearBuilt','HalfBath','OverallQual']
index = 0
for i in range(2):
    for j in range(2):
        column = attr[index]
        axes[i,j].scatter(dataset[column],dataset['SalePrice']/1000,s=.5,color='g')
        axes[i,j].set_title(column+' Vs '+'SalePrice')
        axes[i,j].set_xlabel(column)
        axes[i,j].set_ylabel('SalePrice (K)')
        index=index+1
plt.subplots_adjust(wspace=.5,hspace=.5)
plt.savefig('../Figure/4Report/NumericScatter.png')
plt.clf()

#representative histogram for report
fig, axes = plt.subplots(2,2)
attr=['2ndFlrSF','YearBuilt','EnclosedPorch','OverallQual']
index = 0
for i in range(2):
    for j in range(2):
        column = attr[index]
        axes[i,j].hist(dataset[column],color='g')
        axes[i,j].set_xlabel(column)
        axes[i,j].set_ylabel('Frequency')
        index=index+1
plt.subplots_adjust(wspace=.5,hspace=.5)
plt.savefig('../Figure/4Report/NumericaHist.png')
plt.clf()

fig, axes = plt.subplots(2,2,figsize=(7,7))
attr=['BldgType','HeatingQC','GarageType','MiscFeature']
index = 0
for i in range(2):
    for j in range(2):
        column = attr[index]
        sns.countplot(dataset[column],ax=axes[i,j])
        axes[i,j].set_xlabel(column)
        axes[i,j].set_ylabel('Frequency')
        index=index+1

for ax in fig.axes:
    plt.sca(ax)
    plt.xticks(rotation=30)

plt.subplots_adjust(wspace=.5,hspace=.5)
plt.savefig('../Figure/4Report/CategoryBar.png')
plt.clf()

fig, axes = plt.subplots(2,2,figsize=(7,7))
attr=['BldgType','HeatingQC','GarageType','MiscFeature']
dataset.fillna('Apple',inplace=True)
index = 0
for i in range(2):
    for j in range(2):
        column = attr[index]
        sns.catplot(x=column,y='SalePrice',data=dataset,ax=axes[i,j])
        axes[i,j].set_xlabel(column)
        axes[i,j].set_ylabel('Sale Price')
        index=index+1

for ax in fig.axes:
    plt.sca(ax)
    plt.xticks(rotation=30)

plt.subplots_adjust(wspace=.5,hspace=.5)
plt.savefig('../Figure/4Report/CategoryScatter.png')
plt.clf()
"""
dataset.fillna('NotGiven',inplace=True)
category = dataset.select_dtypes(exclude=numerics)

for column in category:
    ax=sns.catplot(x=column,y='SalePrice',data=dataset)
    ax.set(ylabel='Sale Price')
    ax.set_xticklabels(rotation=40)
    tmp = '../Figure/CategoryScatter/'+column+'.png'
    plt.savefig(tmp)
    plt.clf()
#box plot of categorical attributes and house Sale Price
