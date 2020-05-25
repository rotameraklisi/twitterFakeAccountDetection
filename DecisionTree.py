#!/usr/bin/env python
# coding: utf-8

# ## DECISION TREE

# In[1]:


import sys
import csv
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn import tree
from sklearn.impute import SimpleImputer
from sklearn.model_selection import cross_validate
#import sexmachine.detector as gender
from sklearn import metrics
from sklearn import preprocessing
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.model_selection import learning_curve
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
get_ipython().magic(u'matplotlib inline')


# Öğrenme Eğrisi için Fonksiyon

# In[3]:


def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5)):
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    return plt


# Confusion Matrix çizmek için Fonksiyon

# In[4]:


def plot_confusion_matrix(cm, title='Confusion matrix', cmap=plt.cm.Blues):
    target_names=['Fake','Genuine']
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(target_names))
    plt.xticks(tick_marks, target_names, rotation=45)
    plt.yticks(tick_marks, target_names)
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


# ROC eğrisini çizmek için fonksiyon

# In[5]:


def plot_roc_curve(y_test, y_pred):
    false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y_pred)
    print ("False Positive rate: ",false_positive_rate)
    print ("True Positive rate: ",true_positive_rate)
    
    roc_auc = auc(false_positive_rate, true_positive_rate)

    plt.title('Receiver Operating Characteristic')
    plt.plot(false_positive_rate, true_positive_rate, 'b',
    label='AUC = %0.2f'% roc_auc)
    plt.legend(loc='lower right')
    plt.plot([0,1],[0,1],'r--')
    plt.xlim([-0.1,1.2])
    plt.ylim([-0.1,1.2])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()


# In[6]:


print("Veri Seti Okunuyor.....\n")
genuine_users = pd.read_csv(r"C:\Users\Asus\Desktop\users.csv")
fake_users = pd.read_csv(r"C:\Users\Asus\Desktop\fusers.csv")
print("Gerçek Kullanıcı sütunları")
print(genuine_users.columns)
print("Gerçek Kullanıcı")
print(genuine_users.describe())
print("Sahte Kullanıcı")
print(fake_users.describe())
print(genuine_users)
x=pd.concat([genuine_users,fake_users], ignore_index=True)
print(len(x))
##genuine kadar 0 ve fake kadar 1 
y=len(fake_users)*[0] + len(genuine_users)*[1]



# In[7]:


print(x.columns)


# In[8]:


print(len(x))
df = pd.DataFrame(x)
print (df)
print (df.dtypes)
df.fillna(0)


# In[9]:


print("Öznitelik Çıkarımı.....\n")


name_list = list(enumerate(np.unique(x['name']))) 
name_dict = { name : i for i, name in name_list }   
x.loc[:,'name_code'] = x['name'].map( lambda x: name_dict[x]).astype(int)    
print(x.loc[:,'name_code'])

screenName_list = list(enumerate(np.unique(x['screen_name']))) 
screenName_dict = { name : i for i, name in screenName_list }   
x.loc[:,'screenName_code'] = x['screen_name'].map( lambda x: screenName_dict[x]).astype(int)    
print(x.loc[:,'screenName_code'])


createdAt_list = list(enumerate(np.unique(x['created_at']))) 
lang_dict = { name : i for i, name in createdAt_list }   
x.loc[:,'createdAt_code'] = x['created_at'].map( lambda x: lang_dict[x]).astype(int)    
print(x.loc[:,'createdAt_code'])

url_list = list(enumerate(np.unique(x['url']))) 
url_dict = { name : i for i, name in url_list }   
x.loc[:,'url_code'] = x['url'].map( lambda x: url_dict[x]).astype(int)    
print(x.loc[:,'url_code'])

time_zone_list = list(enumerate(np.unique(x['time_zone']))) 
time_zone_dict = { name : i for i, name in time_zone_list }   
x.loc[:,'time_zone_code'] = x['time_zone'].map( lambda x: time_zone_dict[x]).astype(int)    
print(x.loc[:,'time_zone_code'])

location_list = list(enumerate(np.unique(x['location']))) 
location_dict = { name : i for i, name in location_list }   
x.loc[:,'location_code'] = x['location'].map( lambda x: location_dict[x]).astype(int)    
print(x.loc[:,'location_code'])


description_list = list(enumerate(np.unique(x['description']))) 
description_dict = { name : i for i, name in description_list }   
x.loc[:,'description_code'] = x['description'].map( lambda x: description_dict[x]).astype(int)    
print(x.loc[:,'description_code'])


updated_list = list(enumerate(np.unique(x['updated']))) 
updated_dict = { name : i for i, name in updated_list }   
x.loc[:,'updated_code'] = x['updated'].map( lambda x: updated_dict[x]).astype(int)    
print(x.loc[:,'updated_code'])

dataset_list = list(enumerate(np.unique(x['dataset']))) 
dataset_dict = { name : i for i, name in dataset_list }   
x.loc[:,'dataset_code'] = x['dataset'].map( lambda x: dataset_dict[x]).astype(int)    
print(x.loc[:,'dataset_code'])





# In[10]:


lang_list = list(enumerate(np.unique(x['lang']))) 
##lang kodları sayısallaştırılıyor
print(lang_list)
lang_dict = { name : i for i, name in lang_list }   
print(lang_dict)
x.loc[:,'lang_code'] = x['lang'].map( lambda x: lang_dict[x]).astype(int)    
print(x.loc[:,'lang_code'])




# In[11]:


feature_columns_to_use=['name_code','screenName_code','statuses_count','followers_count','friends_count','favourites_count',
'listed_count', 'createdAt_code','url_code','lang_code','time_zone_code',
'location_code','default_profile', 'default_profile_image','geo_enabled',
'profile_use_background_image','profile_background_tile','utc_offset',
'protected','verified','updated_code','dataset_code']

feature_columns_to_use=feature_columns_to_use
x=x.loc[:,feature_columns_to_use]
print(x)
df = pd.DataFrame(x)
x=df.fillna(0)
print(x)


# In[12]:


print("Veriseti train ve test verisi olarak ayrılıyor...\n")
X_train,X_test,y_train,y_test = train_test_split(x, y, test_size=0.20, random_state=44)


# In[13]:


print("Veri seti Eğitiliyor.......\n")
clf = tree.DecisionTreeClassifier()
clf.fit(X_train, y_train)

y_pred =clf.predict(X_test)

feat_importance = clf.tree_.compute_feature_importances(normalize=False)
print("feature importance = " + str(feat_importance))

from sklearn.tree.export import export_graphviz
from sklearn.feature_selection import mutual_info_classif
from StringIO import StringIO

out = StringIO()
out = export_graphviz(clf, out_file='test')


# In[14]:


print("Eski Test Veriseti Accuracy: ") ,accuracy_score(y_test, y_pred)


# In[15]:


print("Veri seti Eğitiliyor.......\n")
clf = tree.DecisionTreeClassifier()
clf.fit(X_train, y_train)

y_pred =clf.predict(X_test)

feat_importance = clf.tree_.compute_feature_importances(normalize=False)
print("feature importance = " + str(feat_importance))

from sklearn.tree.export import export_graphviz
from sklearn.feature_selection import mutual_info_classif
from StringIO import StringIO

out = StringIO()
out = export_graphviz(clf, out_file='test')


# In[16]:


feature_columns_to_use=['name_code','screenName_code','statuses_count','followers_count','friends_count','favourites_count',
'listed_count', 'createdAt_code','url_code']

feature_columns_to_use=feature_columns_to_use
x=x.loc[:,feature_columns_to_use]
print(x)
df = pd.DataFrame(x)
x=df.fillna(0)
print(x)


# In[17]:


print("Veriseti train ve test verisi olarak ayrılıyor...\n")
X_train,X_test,y_train,y_test = train_test_split(x, y, test_size=0.20, random_state=44)


# In[18]:


print("Veri seti Eğitiliyor.......\n")
clf = tree.DecisionTreeClassifier()
clf.fit(X_train, y_train)

y_pred =clf.predict(X_test)

feat_importance = clf.tree_.compute_feature_importances(normalize=False)
print("feature importance = " + str(feat_importance))

from sklearn.tree.export import export_graphviz
from sklearn.feature_selection import mutual_info_classif
from StringIO import StringIO

out = StringIO()
out = export_graphviz(clf, out_file='test')


# In[19]:


print("Test Veriseti Accuracy: ") ,accuracy_score(y_test, y_pred)


# In[20]:


cm=confusion_matrix(y_test, y_pred)
print("Confusion matrix->Normalizasyon Olmadan")
print(cm)
plot_confusion_matrix(cm)


# In[21]:


cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
print("Normalize edilince ->  confusion matrix")
print(cm_normalized)
plot_confusion_matrix(cm_normalized, title='Normalized confusion matrix')


# In[22]:


print(classification_report(y_test, y_pred, target_names=['Fake','Genuine']))


# In[23]:


plot_roc_curve(y_test, y_pred)


# In[ ]:





# In[ ]:




