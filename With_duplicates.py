#!/usr/bin/env python
# coding: utf-8

# # Project Data Science: Housing Price in USA
# ## Prepared by Melina BASHA

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px


# ### Overview of the dataset

# In[2]:


housing = pd.read_csv("housing.csv")
housing.head()


# In[3]:


housing.shape


# In[4]:


housing.info()


# #### As seen above the housing dataset has 9 variables. The target variable is price and the other variables are its features. Among these 9 features we can distinguish 5 numerical variables and 4 categorical variables. Note that there are some variables that are in numerical format, but actually should be categorial variables, like zip code.

# In[5]:


print(f"Numerical variables in dataset: {housing.select_dtypes(exclude = ['object']).columns.tolist()}")
print(f"Categorical variables in dataset: {housing.select_dtypes(include = ['object']).columns.tolist()}")


# #### Below are presented statistics indicators for numerical and categorical variables (in separated tables).

# In[6]:


housing.describe()


# In[7]:


housing.describe(include = 'object')


# ### Data Cleaning

# #### In order of conducting Exploratory Data Analysis and Model Development we need to be sure that the dataset can be usable, otherwise it will produce problems. So we have to do checks related to missing values in the dataset and ways of handling with it. Based on the variables we will decide which of them to keep for this project, which to drop, which to fill with imputations mode.

# In[8]:


values_missing = housing.isna().sum()*100/len(housing)
print('Percentage Missing Values %')
values_missing


# #### As seen above there are variables that have missing values. First we will remove columns  city, zip_code and prev_sold_date, because they will be not used.

# In[9]:


housing = housing.drop(["city","zip_code","prev_sold_date"], axis=1)


# #### From Out[8] we can see that the target variables price has 0.78% missing values, so it is better to just remove the missing values from it.

# In[10]:


housing = housing.drop(housing[housing['price'].isnull()].index)


# #### Also we will remove the rows that have 2 or more missing values.

# In[11]:


housing = housing[~(housing.isna().sum(axis=1) >= 2)]


# #### Below we will make a check to see the state of missing values in the variables.

# In[12]:


((housing.isna().sum() / len(housing)) * 100).sort_values(ascending=False)


# #### Also we will remove the missing values from variables bed and bath as they are < 1 %.

# In[13]:


housing = housing.drop(housing[housing['bed'].isnull()].index, axis=0)
housing = housing.drop(housing[housing['bath'].isnull()].index, axis=0)


# #### Now we will do a check related to status variable and we will remove it as it has only one value "for_sale".

# In[14]:


housing["status"].value_counts()


# In[15]:


housing = housing.drop("status", axis=1)


# #### For variables like acre_lot and house_size we will use Median imputation to handle missing values. Median is more reccommended to be used with numeric variables.

# In[16]:


housing['acre_lot'].fillna(housing['acre_lot'].median(), inplace=True)
housing['house_size'].fillna(housing['house_size'].median(), inplace=True)


# In[17]:


housing.info()


# #### We will make another check if we have missing values.

# In[18]:


housing.isnull().sum()


# #### Data cleaning has not finished yet. We need to detect for outliers and remove them.

# In[19]:


var_num = ['bed','bath','acre_lot','house_size','price']
plt.boxplot(housing[var_num])
plt.xticks([1, 2, 3, 4, 5], var_num)
plt.title('Outlier Before Remove')
plt.show()
print(f'Total Row With Outlier: {housing.shape[0]}')


# #### As seen outliers are detected for price in the visualization above, so we will remove.

# In[20]:


Q1 = housing[var_num].quantile(0.25)
Q3 = housing[var_num].quantile(0.75)
IQR = Q3 - Q1

housing = housing[~((housing[var_num] < (Q1 - 1.5 * IQR)) | (housing[var_num] > (Q3 + 1.5 * IQR))).any(axis=1)]


# In[21]:


var_num = ['bed','bath','acre_lot','house_size','price']
plt.boxplot(housing[var_num])
plt.xticks([1, 2, 3, 4, 5], var_num)
plt.title('Outlier After Remove')
plt.show()
print(f'Total Row Without Outlier: {housing.shape[0]}')


# ### Exploratory Data Analysis and Visualizations

# In[22]:


housing.describe()


# #### Below it is presented the pairplot of each of 2 variables which show the relationship  between them. As presented there is no indication for lienar regression between the variables.

# In[23]:


sns.pairplot(housing, 
             kind='scatter', 
             plot_kws={'alpha':0.4}, 
             diag_kws={'alpha':0.55, 'bins':40})


# #### Below it is presented the correlation matrix for the variables. Considering price as target, there is no evidence for strong correlation between its features.

# In[24]:


sns.heatmap(housing.corr(), annot=True, linewidths=0.5);


# #### Below it is presented the Distribution of Price.

# In[25]:


fig = px.histogram(housing, x="price", nbins=25, template="plotly")
fig.update_layout(title="Distribution of Price")
fig.show()


# #### Below it is presented the chart of Top 10 States with Highest Mean Price

# In[26]:


df_mean = housing.groupby('state')['price'].mean().reset_index()

df_mean_sort = df_mean.sort_values(by='price', ascending=False)

fig = px.bar(df_mean_sort, x='state', y='price',
             title='Top 10 States with Highest Mean Price',
             labels={'state': 'State', 'price': 'Mean Price'})
fig.update_traces(texttemplate='%{y:.1f}', textposition='outside')
fig.show()


# #### Below it is presented the chart of House Size, Bed, Bath to Price.

# In[27]:


fig = px.scatter(housing, x='house_size', y='price', color='bed', size='bath',trendline='ols')
fig.update_layout(title='House Size vs Price',
                  xaxis_title='House Size',
                  yaxis_title='Price')
fig.show()


# #### Keeping the unique values of state, removing the ones that have less than 50 counts and fitting and transforming the "state" column to obtain numeric labels.

# In[28]:


housing["state"].value_counts()


# In[29]:


housing = housing.drop(housing[housing["state"].map(housing["state"].value_counts()) < 50]["state"].index)


# In[30]:


housing["state"].unique()


# In[31]:


from sklearn.preprocessing import LabelEncoder

label_encoder = LabelEncoder()
housing['state_numeric'] = label_encoder.fit_transform(housing['state'])
housing = housing.drop("state", axis=1)
housing


# In[32]:


housing.describe()


# #### Training the Model with multivariable regression using Scikit Learn 

# In[33]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder, StandardScaler, MinMaxScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import math


# #### Standardizing data

# In[34]:


housing['house_size'] = StandardScaler().fit_transform(housing['house_size'].values.reshape(len(housing), 1))
housing['price'] = StandardScaler().fit_transform(housing['price'].values.reshape(len(housing), 1))


# In[35]:


housing['bed'] = MinMaxScaler().fit_transform(housing['bed'].values.reshape(len(housing), 1))
housing['bath'] = MinMaxScaler().fit_transform(housing['bath'].values.reshape(len(housing), 1))
housing['acre_lot'] = MinMaxScaler().fit_transform(housing['acre_lot'].values.reshape(len(housing), 1))


# #### Splitting the data
# ##### X are the predictores, and y is the output. What we want to do is create a model that will take in the values in the X variable and predict y with a linear regression algorithm and random forest regressor algorithm. We will use the SciKit Learn library to create the model.
# ##### In the model we will use even categarorical variable state, by using One-hot encoding, in order to use them as dummy varuables in the model.

# In[36]:


X = housing[['bed', 'bath', 'acre_lot', 'house_size', 'state_numeric']]
y = housing['price']

X = pd.get_dummies(X, columns=['state_numeric'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# #### Linear Regression

# In[37]:


lm = LinearRegression()


# In[38]:


lm.fit(X_train, y_train)


# In[39]:


lm.coef_


# In[40]:


lm.score(X, y)


# In[41]:


# The coefficients in a dataframe
cdf = pd.DataFrame(lm.coef_,X.columns,columns=['Coef'])
print(cdf)


# In[42]:


predictions = lm.predict(X_test)


# In[43]:


print('Mean Absolute Error:',mean_absolute_error(y_test, predictions))
print('Mean Squared Error:',mean_squared_error(y_test, predictions))
print('Root Mean Squared Error:',math.sqrt(mean_squared_error(y_test, predictions)))


# In[45]:


print(predictions)


# #### Residuals
# ##### Distribution plot of the residuals of the model's predictions. They should be normally distributed.

# In[46]:


residuals = y_test-predictions
sns.distplot(residuals, bins=30)


# #### The Linear Regression shows a value of R^2 in the value 39.9% (that is not a very good value). On the other hand the value of MAE and MSE are near 0, which is a very good value.

# #### Random Forest Regressor

# In[47]:


# creating a Random Forest model and train it using training data
model_RF = RandomForestRegressor(n_estimators=100, random_state=42)
model_RF.fit(X_train, y_train)

# Make predictions using testing data
y_pred = model_RF.predict(X_test)

# calculate average error value using MSE metrics
mse_RF = mean_squared_error(y_test, y_pred)
rmse_RF = mean_squared_error(y_test, y_pred, squared=False)
mae_RF = mean_absolute_error(y_test, y_pred)
r2_RF = r2_score(y_test, y_pred)


# In[48]:


result = {'Random Forest': {'MSE': mse_RF, 'RMSE': rmse_RF, 'MAE': mae_RF, 'R^2': r2_RF}
}     


# In[49]:


data = pd.DataFrame.from_dict(result, orient='index')
data = data.applymap(lambda x: f'{x:.2f}')
print(data)


# #### The Random Forest Regressor shows that all indicators MSE, RMSE, MAE, R^2 are very good and this algorithm is appropriate for the relationship of the varibles.

# #### Training the model with multivariable regression using OLS

# In[50]:


import  statsmodels.api as sm
X = sm.add_constant(X_train)
model = sm.OLS(y_train, X)
model_fit = model.fit()
print(model_fit.summary())


# In[ ]:




