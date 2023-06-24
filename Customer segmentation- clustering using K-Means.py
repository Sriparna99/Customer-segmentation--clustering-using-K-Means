#!/usr/bin/env python
# coding: utf-8

# In[50]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


# In[2]:


df = pd.read_csv(r"C:\Users\shree\Downloads\Mall_Customers.csv")


# In[3]:


df.head(2)


# In[4]:


# customer id column is not needed hence dropped it

df = df.drop("CustomerID",axis =1 )


# In[5]:


# Renaming columns helps to call column names more simplier way and it will be easier to work

df.columns = ["Gender", "Age","Income","Score"]


# In[6]:


df.head(2)


# # distribution based on gender

# In[51]:


plt.figure(figsize = (20,4))
plt.subplot(1,3,1)
sns.distplot(df.Age[df["Gender"]=="Female"],color = "orange",hist = False, kde= True , label = "Female")
sns.distplot(df.Age[df["Gender"]=="Male"],color = "Blue",hist = False, kde= True , label = "Male")
plt.title("Age")


plt.subplot(1,3,2)
sns.distplot(df.Income[df["Gender"]=="Female"],color = "orange",hist = False, kde= True, label = "Female")
sns.distplot(df.Income[df["Gender"]=="Male"],color = "Blue",hist = False, kde= True, label = "Male")
plt.title("Income")

plt.subplot(1,3,3)
sns.distplot(df.Score[df["Gender"]=="Female"],color = "orange",hist = False, kde= True, label = "Female")
sns.distplot(df.Score[df["Gender"]=="Male"],color = "Blue",hist = False, kde= True, label = "Male")
plt.title("Score")

plt.show



# ##### The differences that show on above plot that slightly more females age between 28-35 shops more and slightly more female's spending score is 50

# # Differences in Age, Score and income by gender

# #### box plot visualizations will be used to see outliers, quartiles , distribution and median.

# In[20]:


plt.figure(figsize = (20,5))
plt.subplot(1,3,1)
sns.boxplot(x = df.Gender , y = df.Age, data = df)
plt.title("Age")

plt.subplot(1,3,2)
sns.boxplot(x = df.Gender , y = df.Income, data = df)
plt.title("Income")

plt.subplot(1,3,3)
sns.boxplot(x = df.Gender , y = df.Score, data = df)
plt.title("Score")



# ### Relation between variables
# 
# ##### Income increases with age and score decreases with age

# # scatter plot

# In[25]:


plt.figure(figsize = (20,5))
plt.subplot(1,3,1)
sns.scatterplot( x = df.Age , y = df.Income , hue = df.Gender)
plt.title("Age vs Income")

plt.subplot(1,3,2)
sns.scatterplot( x = df.Age , y = df.Score , hue = df.Gender)
plt.title("Age vs score")

plt.subplot(1,3,3)
sns.scatterplot( x = df.Income , y = df.Score , hue = df.Gender)
plt.title("Income vs Score")

plt.show()




# # insights
# 
# there seems to be 2 groups of customers by age vs score(top right quarter and bottom right quarter) , where diagonal is delimiting them
# 
# The most important thing is actually the insight that we are getting from  income vs score. we can see 5 different groups of customers 

# # barplot
# 
# ### income and score by age 

# In[29]:


plt.figure(figsize = (20,8))
plt.subplot(2,1,1)
sns.barplot(x = df.Age, y = df.Income , hue = df.Gender, ci= 0)
plt.title("Income by Age")

plt.subplot(2,1,2)
sns.barplot(x = df.Age, y = df.Score , hue = df.Gender ,ci = 0)
plt.title("Score by Age")

plt.show()


# ### it reflects that income seems to be higher between age of 25 to 50 and spending score is higher between age of 18 to 40 comparing to otheres

# # Data Preparation

# ### gender column will be encoded with 0 and 1 values and age income and score column needs to be normalized

# In[32]:


df = pd.get_dummies(data = df , columns = ["Gender"],drop_first = True)
df = df.rename(columns = {"Gender_Male" : "Gender"})


# In[33]:


from sklearn.preprocessing import StandardScaler


# In[35]:


# Create new dataframe with transformed values
df_t = df.copy()
ss = StandardScaler()

df_t = ss.fit_transform(df["Age"].values.reshape(-1,1))
df_t = ss.fit_transform(df["Income"].values.reshape(-1,1))
df_t = ss.fit_transform(df["Age"].values.reshape(-1,1))


# # Clustering Using KMeans

# #### First we will try to find ideal numbers of clusters using elbow method, where we draw inertia for number of clusters in range 1 to 10 and try to find one with highest one gap angle.
# #### Then we will use these values to draw clusters and decide which one is most valuable for us
# #### We draw inertia of KMeans on raw as well as normalized data, just to see if it makes any difference :)

# In[43]:


from sklearn.cluster import KMeans
# untransformed data
inertia = []
for i in range(1, 10):
    km = KMeans(n_clusters=i).fit(df)
    inertia.append(km.inertia_)


# In[45]:



# tranformed data 
inertia_t = []
for i in range(1,10):
    km = KMeans(n_clusters = i).fit(df_t)
    inertia_t.append(km.inertia_)


# In[46]:


# plot results
plt.figure(figsize=(20,5))
plt.subplot(1,2,1)
sns.lineplot(x=range(1,10), y=inertia)
plt.title('KMeans inertia on original data')

plt.subplot(1,2,2)
sns.lineplot(x=range(1,10), y=inertia_t)
plt.title('KMeans inertia on transformed data')

plt.show()


# #### Elbow results
# ### When looking on inertia for original data, 2, 3 and 5 seems to be our candidates for number of clusters. When looking on inertia in transformed data, 2 and 4 seems to be best... so we simply check how clustering looks like when using 2, 3, 4 and 5 clusters.

# In[48]:


# collect cluster labels as well as cluster centers
clusters = [2,3,4,5]
cluster_centers = {}

for c in clusters:
    km = KMeans(n_clusters=c).fit(df[['Age', 'Income', 'Score', 'Gender']])
    df['cluster' + str(c)] = km.labels_
    cluster_centers[str(c)] = km.cluster_centers_


# # Select best cluster number
# Now, we will draw 4 charts showing clustering on chart income vs spending. Cluster centers are drawn as well. Based on business needs, we should choose one that we can describe bests and will be useful for our business.
# 
# From my point of view, KMeans outperforms other when number of clusters is 2 or 5.
# 
# I think 2 is not enough clusters and just divide customers into 2 groups - score under 50 and score over 50 so does not look well.
# 
# On the other side, when using 5 clusters, we are getting 5 different groups of customers that separates well from each other and we could run different campaigns on each customer group!

# In[52]:


plt.figure(figsize=(20,15))
for i, c in enumerate(clusters):
    plt.subplot(2,2,i+1)
    sns.scatterplot(df.Income, df.Score, df['cluster' + str(c)], s=120, palette=sns.color_palette("hls", c))
    sns.scatterplot(cluster_centers[str(c)][:,1], cluster_centers[str(c)][:,2], color='black', s=300)
    plt.title('Number of clusters: ' + str(c))
    
plt.show()


# In[53]:


plt.figure(figsize=(20,6))
plt.subplot(1,3,1)
sns.scatterplot(df.Income, df.Score, df['cluster5'], s=120, palette=sns.color_palette("hls", 5))
plt.title('Income vs Score')
   
plt.subplot(1,3,2)
sns.scatterplot(df.Age, df.Score, df['cluster5'], s=120, palette=sns.color_palette("hls", 5))
plt.title('Age vs Score')

plt.subplot(1,3,3)
sns.scatterplot(df.Age, df.Income, df['cluster5'], s=120, palette=sns.color_palette("hls", 5))
plt.title('Age vs Income')

plt.show()


# # Conclusion
# We have selected to have 5 clusters, meaning 5 customer groups. But who are people in these groups? Let's try to describe them for marketing team!
# 
# Poor and not-spender - customers with low income and low spending score (cluster #4)
# Poor and spender - customers with low income, but spending a lot (cluster #1)
# Neutral - customers with mid income and mid spending score (cluster #0)
# Rich and not-spender - customers with high income and low spending score (cluster #2)
# Rich and spender - customers with high income and high spending score (cluster #3)

# In[ ]:




