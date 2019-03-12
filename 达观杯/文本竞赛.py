
# coding: utf-8

# In[28]:


import pandas as pd
from sklearn.model_selection import train_test_split
import gensim
import pickle
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
import lightgbm as lgb

get_ipython().run_line_magic('matplotlib', 'inline')


# ## 数据处理

# In[2]:


train_df = pd.read_csv('./new_data/train_set.csv')
test_df = pd.read_csv('./new_data/test_set.csv')


# In[3]:


columns=train_df.columns
columns=columns.drop('class')
columns


# In[4]:


train_x, test_x, train_y, test_y = train_test_split(train_df[columns], train_df['class'], test_size=0.3, random_state=2019)


# In[5]:


train_x.info()


# ## TF-IDF

# In[6]:


vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_df=0.9, sublinear_tf=True)
vectorizer.fit(train_x['word_seg'])


# In[7]:


train_X = vectorizer.transform(train_x['word_seg'])   


# In[8]:


test_X = vectorizer.transform(test_x['word_seg'])   


# In[12]:


from sklearn.svm import SVC, LinearSVC
classifier = LinearSVC()
classifier.fit(train_X, train_y)


# In[13]:


test_Y = classifier.predict(test_X)


# In[14]:


from sklearn.metrics import f1_score
score = f1_score(test_y, test_Y, average='macro')
print("验证集分数：{}".format(score))


# In[21]:


from sklearn.linear_model import LogisticRegression
lr=LogisticRegression(C=1.0, penalty='l1', tol=1e-6)


# In[24]:


lr.fit(train_X,train_y)


# In[25]:


test_Y = lr.predict(test_X)


# In[26]:


score = f1_score(test_y, test_Y, average='macro')
print("验证集分数：{}".format(score))


# In[15]:


x_test = vectorizer.transform(test_df['word_seg'])


# In[16]:


y_test = classifier.predict(x_test)


# In[17]:


test_df['class'] = y_test.tolist()
df_result = test_df.loc[:, ['id', 'class']]


# In[18]:


# df_result.to_csv('./beginner2.csv', index=False)


# ## LGB

# In[ ]:


def f1_score_vali(preds, data_val):
    
    labels = data_vali.get_label()
    preds = np.argmax(preds.reshape(20, -1), axis=0)
    score_vali = f1_score(y_true=labels, y_pred=preds, average='macro')
    return 'f1_score', score_vali, True


# In[32]:


params = {
        'boosting': 'gbdt',
        'application': 'multiclass',
        'num_class': 20,
        'learning_rate': 0.1,
        'num_leaves':31,
        'max_depth':-1,
        'lambda_l1': 0,
        'lambda_l2': 0.5,
        'bagging_fraction' :1.0,
        'feature_fraction': 1.0
        }
bst = lgb.train(params, train_X, num_boost_round=800, valid_sets=train_y,feval=None, early_stopping_rounds=None,
                verbose_eval=True)



# In[ ]:


test_y = np.argmax(bst.predict(test_x), axis=1) + 1

df_result = pd.DataFrame(data={'id':range(102277), 'class': y_test.tolist()})
result_path = '../results/' + features_path.split('/')[-1] + '_lgb' + '.csv'
df_result.to_csv(result_path, index=False)

