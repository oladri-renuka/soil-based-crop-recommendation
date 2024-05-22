#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import model_selection
from sklearn import preprocessing
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.ensemble import BaggingClassifier


# In[2]:


df = pd.read_csv('../input/smart-agricultural-production-optimizing-engine/Crop_recommendation.csv')


# In[3]:


df.head()


# In[4]:


df.isnull().sum()


# In[5]:


df.info()


# In[6]:


df.describe()


# In[7]:


df.shape


# In[10]:


sns.displot(x=df['N'], bins=20,kde=True,edgecolor="black",color='black',facecolor='pink')
plt.title("Nitrogen",size=20)
plt.show()


# In[16]:


sns.displot(x=df['P'],bins=20,color='gold',edgecolor='black',kde=True,facecolor='violet')
plt.title("Phosphorus", size=20)
plt.xticks(range(0,150,20))
plt.show()


# In[15]:


sns.displot(x=df['K'],kde=True, bins=20, facecolor='purple',edgecolor='white', color='gold')
plt.title("Potassium",size=20)
plt.show()


# In[17]:


sns.displot(x=df['temperature'], bins=20,kde=True,edgecolor="black",color='black',facecolor='green')
plt.title("Temperature",size=20)
plt.show()


# In[18]:


sns.displot(x=df['humidity'], color='black',facecolor='yellow',kde=True,edgecolor='black')
plt.title("Humidity",size=20)
plt.show()


# In[23]:


sns.displot(x=df['rainfall'], color='black',facecolor='blue',kde=True,edgecolor='black')
plt.title("Rainfall",size=20)
plt.show()


# In[30]:



crops = df['label'].unique()
print(len(crops))
print(crops)
print(pd.value_counts(df['label']))


# In[31]:


# Filtering each unique label and store it in a list df2 for to plot the box plot

df2=[]
for i in crops:
    df2.append(df[df['label'] == i])
df2[1].head()


# In[32]:


sns.catplot(data=df, x='label', y='temperature', kind='box', height=10, aspect=20/8.27)
# plt.xticks(rotation='vertical')
plt.title("Temperature", size=20)
plt.show()


# In[33]:


sns.catplot(data=df, x='label', y='humidity', kind='box', height=10, aspect=20/8.27)
# plt.xticks(rotation='vertical')
plt.title("Humidity", size=20)
plt.show()


# In[34]:


sns.catplot(data=df, x='label', y='temperature', kind='box', height=10, aspect=20/8.27)
plt.show()


# In[35]:


sns.catplot(data=df, x='label', y='N', kind='box', height=10, aspect=20/8.27)
# plt.xticks(rotation='vertical')
plt.show()


# In[36]:


sns.catplot(data=df, x='label', y='ph', kind='box', height=10, aspect=20/8.27)
# plt.xticks(rotation='vertical')
plt.title("Nitrogen",size=20)
plt.show()


# In[37]:


sns.catplot(data=df, x='label', y='P', kind='box', height=10, aspect=20/8.27)
# plt.xticks(rotation='vertical')
plt.title("Phosphorus",size=20)
plt.show()


# In[38]:


sns.catplot(data=df, x='label', y='K', kind='box', height=10, aspect=20/8.27)
# plt.xticks(rotation='vertical')
plt.title("Potassium",size=20)
plt.show()


# In[39]:


def detect_outlier(x):
    q1 = x.quantile(0.25)
    q3 = x.quantile(0.75)
    IQR = q3-q1
    lower_limit = q1 - (1.5*IQR)
    upper_limit = q3 + (1.5*IQR)
    print(f"Lower limit: {lower_limit} Upper limit: {upper_limit}")
    print(f"Minimum value: {x.min()}   MAximum Value: {x.max()}")
    for i in [x.min(),x.max()]:
        if i == x.min():
            if lower_limit > x.min():
                print("Lower limit failed - Need to remove minimum value")
            elif lower_limit < x.min():
                print("Lower limit passed - No need to remove outlier")
        elif i == x.max():
            if upper_limit > x.max():
                print("Upper limit passed - No need to remove outlier")
            elif upper_limit < x.max():
                print("Upper limit failed - Need to remove maximum value")
detect_outlier(df['K'][df['label']=='grapes'])


# In[40]:


for i in df['label'].unique():
    detect_outlier(df['K'][df['label']==i])
    print('---------------------------------------------')


# In[41]:


x = df.drop(['label'], axis=1)
x.head()


# In[42]:


Y = df['label']
encode = preprocessing.LabelEncoder()
y = encode.fit_transform(Y)
print("Label length: ",len(y))


# In[43]:


x_train,x_test,y_train,y_test = model_selection.train_test_split(x,y)
print(len(x_train),len(y_train),len(x_test),len(y_test))


# In[44]:


from sklearn.ensemble import GradientBoostingClassifier, AdaBoostClassifier
import xgboost as xgb

a = {
    'decision_tree': {
        'model': DecisionTreeClassifier(criterion='gini'),
        'params': {'decisiontreeclassifier__splitter': ['best', 'random']}
    },
    'svm': {
        'model': SVC(gamma='auto', probability=True),
        'params': {
            'svc__C': [1, 10, 100, 1000],
            'svc__kernel': ['rbf', 'linear']
        }
    },
    'random_forest': {
        'model': RandomForestClassifier(),
        'params': {
            'randomforestclassifier__n_estimators': [1, 5, 10]
        }
    },
    'k_classifier': {
        'model': KNeighborsClassifier(),
        'params': {
            'kneighborsclassifier__n_neighbors': [5, 10, 20, 25],
            'kneighborsclassifier__weights': ['uniform', 'distance']
        }
    },
    'xgboost': {
        'model': xgb.XGBClassifier(),
        'params': {
            'xgbclassifier__n_estimators': [50, 100, 150],
            'xgbclassifier__learning_rate': [0.05, 0.1, 0.2],
            'xgbclassifier__max_depth': [3, 4, 5]
        }
    },
    'gradient_boosting': {
        'model': GradientBoostingClassifier(),
        'params': {
            'gradientboostingclassifier__n_estimators': [50, 100, 150],
            'gradientboostingclassifier__learning_rate': [0.05, 0.1, 0.2],
            'gradientboostingclassifier__max_depth': [3, 4, 5]
        }
    },
    'adaboost': {
        'model': AdaBoostClassifier(),
        'params': {
            'adaboostclassifier__n_estimators': [50, 100, 150],
            'adaboostclassifier__learning_rate': [0.05, 0.1, 0.2]
        }
    }
}


# In[45]:


score=[]
details = []
best_param = {}
for mdl,par in a.items():
    pipe = make_pipeline(preprocessing.StandardScaler(),par['model'])
    res = model_selection.GridSearchCV(pipe,par['params'],cv=5)
    res.fit(x_train,y_train)
    score.append({
        'Model name':mdl,
        'Best score':res.best_score_,
        'Best param':res.best_params_
    })
    details.append(pd.DataFrame(res.cv_results_))
    best_param[mdl]=res.best_estimator_
pd.DataFrame(score)


# In[46]:


details[0]


# In[47]:


details[1]


# In[48]:


details[2]


# In[49]:


details[3]


# In[50]:


score


# In[51]:


pd.DataFrame(score)


# In[52]:


for i in best_param.keys():
    print(f'{i} : {best_param[i].score(x_test,y_test)}')


# In[53]:


predicted = best_param['random_forest'].predict(x_test)


# In[54]:


plt.figure(figsize=(12,8))
sns.heatmap(confusion_matrix(y_test,predicted),annot=True)
plt.xlabel("Original")
plt.ylabel("Predicted")
plt.show()


# In[55]:


pipe1 = make_pipeline(preprocessing.StandardScaler(),RandomForestClassifier(n_estimators = 10))
bag_model = BaggingClassifier(base_estimator=pipe1,n_estimators=100,
                              oob_score=True,random_state=0,max_samples=0.8)


# In[56]:


bag_model.fit(x_train,y_train)


# In[57]:


bag_model.score(x_test,y_test)


# In[58]:


predict = bag_model.predict(x_test)


# In[59]:


plt.figure(figsize=(12,8))
sns.heatmap(confusion_matrix(y_test,predict),annot=True)
plt.show()


# In[60]:


dha2 =pd.DataFrame(Y)
code = pd.DataFrame(dha2['label'].unique())


# In[61]:


dha = pd.DataFrame(y)
encode = pd.DataFrame(dha[0].unique())
refer = pd.DataFrame()
refer['code']=code
refer['encode']=encode
refer


# In[62]:


print(classification_report(y_test,predict))


# In[ ]:




