#CS5525 Data Analytics 2018 Fall Group Project



## Data and Goal 

 We chose one of the available data sets on Kaggle  [House Prices: Advanced Regression Techniques](https://www.kaggle.com/c/house-prices-advanced-regression-techniques), which provides information describing houses in Ames, Iowa. The goal here is to utilize data mining techniques to build models predicting the house price.



## Data exploration



## Data cleaning

#### Filtering out missing value

It turns out that every single record contains missing values, which are denoted by the sentinel value 'NaN', in this data set. The 6 attributes with highest missing value percentage are filtered out.


| Attributes  | Missing Value Percentage |
| ----------- | :----------------------: |
| PoolQC      |           99.5           |
| MiscFeature |           96.3           |
| Alley       |           93.8           |
| Fence       |           80.8           |
| FireplaceQu |           47.3           |
| LotFrontage |           17.8           |

#### Filling in missing value

The remaining missing value are filled in with certain value

