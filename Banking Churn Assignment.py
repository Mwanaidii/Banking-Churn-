#!/usr/bin/env python
# coding: utf-8

# In[91]:


import pandas as pd
import numpy as np
import seaborn as sns
import plotly.figure_factory as ff
import statsmodels.api as sm
import matplotlib.pyplot as plt


# ### 1.Understanding the Data

# In[92]:


# Load the dataset
df = pd.read_csv("banking_churn.csv.")
print(f"\n\033[1mFindings:\033[0m The dataset consists of {df.shape[0]} rows and {df.shape[1]} columns")
df.head()


# In[93]:


# Check data summary
df.info()


# In[94]:


df.isnull().sum()


# * Comments: We have no missing values in our data set so we can conclude that our data is clean.

# In[95]:


df.describe()


# In[96]:


# Checking the number of unique values in every column

df.nunique().sort_values()


# In[97]:


## Checking out the unique Geography.

df.Geography.unique()


# In[98]:


## the number of customers in each location using the unique ID.

## df.groupby('Geography')['CustomerId'].count() (This code could aso work)

data=df.groupby(['Geography'])['CustomerId'].aggregate('count').reset_index().sort_values('CustomerId', ascending=False)
data.head()

ff.create_table(data) # Creating a more presentable table.


# In[99]:


## Checking out the minimum and maximum age of the customers.

min_age = df.Age.min()
max_age = df.Age.max()

print(f"The dataset contains customers whose age ranges from {min_age} to {max_age} years old.")


# In[100]:


## Check the minimum and maximum balance in the data set, assume Euros which is the currency used in the above countries.

min_balance = df.Balance.min()
max_balance = df.Balance.max()

print(f"The dataset contains customers whose balance ranges from {min_balance} to {max_balance} Euros.")


# In[101]:


# Check Customers whose bank balance is zero.

df.query("Balance == 0")


# In[103]:


# Customers who have the highest bank balance.

df.query(f"Balance in {df.Balance.nlargest(10).to_list()}")


# ### 2.Exploratory Data Analysis
# 

# In[105]:


# What variable is most correlated to Exiting?

sns.heatmap(df.corr(), annot=True, annot_kws={"size": 7.5}, cmap='Oranges', square=True)


# ##### Comments: 
# * Age has the highest correlation with Exiting with a positive corr of 0.29.
# * Active members also have the highest correlation with Exiting with a negative corr of 0.16 
# * Balance has a positive corr of 0.12 with Exiting. 
# * We also see that the number of products is highest correlated with the account Balance with negative 0.3 corr.
# * We can conclude that older customers and those with higher balances and are active are more likely to Exit.

# In[83]:


## What is the distribution of Exited customers along various variables?

plt.figure(figsize = (17, 17))

plt.subplot(3,2,1)
sns.countplot(x = 'Exited', hue='IsActiveMember', data = df)

plt.subplot(3,2,2)
sns.countplot(x = 'Exited', hue='Gender', data = df)

plt.subplot(3,2,3)
sns.countplot(x = 'Exited', hue='Tenure', data = df)

plt.subplot(3,2,4)
sns.countplot(x = 'Exited', hue='NumOfProducts',data = df)


# ##### Comments
# * Non Active customers are more likely to exit.
# * A higher number of female customers is more likely to Exit compared to male customers.
# * Customers with only one Product are more likely to Exit.

# In[ ]:





# * We saw that France has the highest number of customers i.e
#     France     5014
#     Germany    2509
#     Spain      2477
# we can plot a histogram to get the visual variation.
# 

# In[109]:


plt.figure(figsize = (20, 15))

plt.subplot(3,2,1)
sns.countplot(x = 'Exited', hue='Geography', data = df)


# ##### Comments:
#     
# * Spain and Germany have almost the same number of customers.
# * France has the highest number of customers and most are less likely to Exit.
# * Germany has more customers that are likely to Exit compared to the other two Geographical locations.

# ### 3.Data Preprocessing

# In[47]:


#Detect and remove outliers in our variables of interest: 

fig, axes = plt.subplots(3, 1)

sns.boxplot(data=df, x = 'Exited', y='EstimatedSalary', ax=axes[0]) 
sns.boxplot(data=df, x = 'Exited', y='Balance', ax=axes[1]) 
sns.boxplot(data=df,x = 'Exited', y = 'Age' , ax=axes[2])

fig.tight_layout()


# ### 4.Feature Engineering

# ### 5.Feature Selection

# ### 6. Modelling and Evaluation

# In[ ]:





# In[ ]:




