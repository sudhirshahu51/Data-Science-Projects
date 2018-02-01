
# coding: utf-8

# # Rehabilitaion Program

# ##  1. Problem statement

# ###### There is a dataset of patients available those are treated for cardiology problem and for whom there is a kind\
#  rehabilitation program goes on. Since as due to different reasons many were unable to participate in the program. \
#  So a survey carried in which data of all patients were taken regarding reason for not joining the program, \
#  there gender, what is there age, whether they have car or not and what is the distance of  house/home of particular\
#  individual from the rehabilitation program centre.  Using this it is required to build a model using logistic regression \
#  method to predict whether new patient based upon his background  will join the program or not.

# ## Importing Libraries

# In[1]:


# data analysis

import numpy as np
import pandas as pd
import sklearn as sk


#Visualization
import seaborn as sns
import matplotlib.pyplot as plt

# machine learning

get_ipython().magic('matplotlib notebook')
from sklearn.preprocessing import MinMaxScaler
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LogisticRegression as lr


# ### Acquire data
# 

# In[2]:


data = pd.read_csv('Data.csv', sep = ";", encoding = 'utf-16') #Reading csv files.

data.info()

data.head()


# ###### Converting data into suitable data types

# In[3]:


data['Age'] = data['Age'].str.replace(',', '.')
data['Distance'] = data['Distance'].str.replace(',', '.')
data['Age'] = pd.to_numeric(data['Age'])
data['Distance'] = pd.to_numeric(data['Distance'])


# In[4]:


data.info()


# In[5]:


print(data)


# ###### Since only 'Reason' is missing in half of the data set, so i'm filling it with 'No reason'.

# In[6]:


data.Reason = data.Reason.fillna('No reason') #Filling missing values of reason


# ## 2. Descriptive analysis of the data and explanation what the data is about.
# 
# 

# #### Features Available in dataset

# In[7]:


print(data.columns.values)


# As from the data it is clear that 5 features are available such as Reason for not joining the program, Gender of patient, Age of patient, means of Mobility for patient, Distance of rehabilitation centre from there living location. 

# #### Numerical features: Age, Distance 
# 
# Continuous: Age, Distance
# 
# 

# ##### Categorical features: Reason, Gender, Mobiity     
# 

# ##### Feature having blank values : Reason
# 

# ##### Data types of features
# 
# 

# In[8]:


data.info()


# ######  No. of people who actually joined the program.

# In[9]:


data.loc[data['Participation']== 1 , 'Participation'].sum()


# In[10]:


data.describe(include=['O'])


# 1. No. of males patients are more as compared to females.
# 2. There are 10 unique(excluding no reason) reasons given by patients regarding not joining the program/or having problemm to continue.
# 3. Almost half of patients don't have car.  

# In[11]:


data[['Gender', 'Participation']].groupby(['Gender'], as_index=False).mean().sort_values(by='Participation', ascending=False)


# - No. of female patients who joined the program is more than the no. of male patients.

# In[12]:


data[['Reason', 'Participation']].groupby(['Reason'], as_index=False).mean().sort_values(by='Reason', ascending=False)


# - People having own facility, or moved, forgot and disliked therapist are ones who mostly didn't participated in the program.

# In[13]:


data[['Mobility', 'Participation']].groupby(['Mobility'], as_index=False).mean().sort_values(by='Mobility', ascending=False)


# - People having car are most likely to participate in program.

# In[14]:


### Participation based on age
g = sns.FacetGrid(data, col='Participation')
g.map(plt.hist, 'Age', bins=18)


# - **Most of people are in 50-80 age range. **
# - **People mainly of age > 60 didn't participated in the program**
# - **People mainly from age 40 to 60 participated in the program.**

# In[15]:


g = sns.FacetGrid(data, col='Participation')
g.map(plt.hist, 'Distance', bins= 20)


# **Distance** from histrogram it clear that distance is a measure factor for people to join rehabitliaion programs as distance below 50 - 55km  are most likely to join. thus I will create a new band for it.

# - **Most people who didn't participate the program were living at distance of 40 and above**

# - **Most people to participate the program were living at distnce less than 60.**

# ## 3. Steps taken in order to perform the task and optimizing the model.

# ##### Add **Gender** and Mobility feature to model training and converting them to numeric values.

# In[16]:


data['Gender'].replace(['F','M'],[0,1],inplace=True)
#data['Gender'].apply({'M':1, 'F':0}.get) #faster as compared to replace.
data['Mobility'].replace(['No car','Car'],[0,1],inplace=True)
data.head()


# ###### Replacing reasons with numeric values and adding column reason values for model training.

# In[17]:


data['Reason'].replace(['No reason','Hospital readmission','Other obligations','Resumed work','Medical reasons','Own facilities','Lost interest','Disliked therapist','Forgot','Moved','Other program'],[0,1,2,3,4,5,6,7,8,9,10],inplace=True)
data.head()


# In[18]:


print(data)


# In[19]:


data.describe()


# ###### Nomalizing and scaling the data using Min, Max values

# - **Scaling** : Reason, age and distance

# In[20]:


scaler = MinMaxScaler()
data[['Reason','Age','Distance']] = scaler.fit_transform((data[['Reason','Age','Distance']]))


# In[21]:


data.head()


# In[22]:


data.describe()


# ##### Distributing data     train: test: cross_validation = 60:20:20

# In[23]:


import sklearn
from sklearn.utils import shuffle  
data = shuffle(data)              #Shuffling values of data

train_x, test_x = sklearn.cross_validation.train_test_split(data, train_size = 0.8, random_state = 1)


# In[24]:


train_x, val_x = sklearn.cross_validation.train_test_split(train_x, train_size = 0.75, random_state = 1)


# In[25]:


train_x.head()


# In[26]:


test_x.head()


# In[27]:


val_x.head()


# In[28]:


train_x.shape, test_x.shape, val_x.shape


# In[29]:


train_y = train_x['Participation']
train_y.shape


# In[30]:


test_y = test_x['Participation']
test_y.shape


# In[31]:


val_y = val_x['Participation']
val_y.shape


# ### Using Logistic regression model to predict the values.

# In[32]:


# clf = LogisticRegression.fit(train_x, train_y)
from sklearn.linear_model import LogisticRegression 
lr = LogisticRegression(C = 100, max_iter=10000)
lr.fit(train_x, train_y)
pred_yt= lr.predict(test_x)
pred_yv= lr.predict(val_x)
acc_log = round(lr.score(train_x, train_y) * 100, 2)
print("Accuracy of Logistic regression classifier on training set: ",acc_log)
test_log = round(lr.score(test_x, pred_yt) * 100, 2)
print("Accuracy of Logistic regression classifier on test set: ",test_log)
val_log = round(lr.score(val_x, pred_yv) * 100, 2)
print("Accuracy of Logistic regression classifier on cross validation set: ",val_log)


# ## 4. Interpretating the output

# - From the results it is clear that the logistic regression classifier gives optimum for given data set.

# ## 5. Description of performance and efficiency.

# Obtaining the performance scores for both test set and cross validation set.
# 1. F1 score
# 2. Precision score
# 3. Recall score

# In[33]:


from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report
#Importing score methods


# ###### Performance scores for test set

# In[34]:


from sklearn.metrics import precision_recall_fscore_support as score



precision, recall, fscore, support = score(test_y, pred_yt)

print('precision: {}'.format(precision))
print('recall: {}'.format(recall))
print('fscore: {}'.format(fscore))
print('support: {}'.format(support))


# ###### Performance scores for cross validation set

# In[35]:


precision, recall, fscore, support = score(val_y, pred_yv)

print('precision: {}'.format(precision))
print('recall: {}'.format(recall))
print('fscore: {}'.format(fscore))
print('support: {}'.format(support))


# ###### Classification report of test set

# In[36]:


# Combined report with all above metrics
from sklearn.metrics import classification_report

print(classification_report(test_y, pred_yt, target_names=['0', '1']))


# ###### Classification report of cross validation set

# In[37]:


# Combined report with all above metrics
from sklearn.metrics import classification_report

print(classification_report(val_y, pred_yv, target_names=['0', '1']))


# In[38]:


from sklearn.metrics import roc_curve, auc

lr = LogisticRegression().fit(train_x, train_y)
y_score_lr = lr.fit(train_x, train_y).decision_function(test_x)
fpr_lr, tpr_lr, _ = roc_curve(test_y, y_score_lr)
roc_auc_lr = auc(fpr_lr, tpr_lr)

plt.figure()
plt.xlim([-0.01, 1.00])
plt.ylim([-0.01, 1.01])
plt.plot(fpr_lr, tpr_lr, lw=3, label='LogRegr ROC curve (area = {:0.2f})'.format(roc_auc_lr))
plt.xlabel('False Positive Rate', fontsize=16)
plt.ylabel('True Positive Rate', fontsize=16)
plt.title('ROC curve (1-of-10 digits classifier)', fontsize=16)
plt.legend(loc='lower right', fontsize=13)
plt.plot([0, 1], [0, 1], color='navy', lw=3, linestyle='--')
plt.axes().set_aspect('equal')
plt.show()


# Area under curve(AUC) have value 1 which confirms the performance of logistic regression algorithm for the given data is 100%.

# ## Conclusion

# - My hypothesis was that logistic regression model will give train and test accuracy above 95% as patient data for rehabilition program\
#   clearly shows distinict differnce for the people of differnt age group those have particpated in program and those who didn't participate.\
#   Simultaneosly majority of people having car particpated in prgram those lived in medium range of distance from program centre, whereas \ 
#   people without car and living at far distances were the most who didn't participate. 
# - Also the test set result and cross validation set results support my hypothesis and instead of 95% accurancy it shows 100% accuracy.\
#  Which is clear in its self based upon how the features Age, Distance, Mobility and few reason easily provide distinction to predict 
#  those who will participate in the program and those will not. The results are well support by F1 score, ROC Curve and Area Under Curve 
#  as their values are 1(not for ROC curve).

# In[ ]:




