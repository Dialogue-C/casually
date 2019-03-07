
# coding: utf-8

# In[1]:


import pandas as pd
from sklearn.model_selection import train_test_split
import gensim
import pickle
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

get_ipython().run_line_magic('matplotlib', 'inline')


# ## 数据处理

# In[2]:


train_df = pd.read_csv('./new_data/train_set.csv')
test_df = pd.read_csv('./new_data/test_set.csv')


# In[4]:


columns=train_df.columns
columns=columns.drop('class')
columns


# In[5]:


train_x, test_x, train_y, test_y = train_test_split(train_df[columns], train_df['class'], test_size=0.3, random_state=2019)


# In[6]:


train_x.info()


# ## TF-IDF

# In[7]:


vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_df=0.9, sublinear_tf=True)
vectorizer.fit(train_x['word_seg'])


# In[8]:


train_X = vectorizer.transform(train_x['word_seg'])   


# In[9]:


test_X = vectorizer.transform(test_x['word_seg'])   


# In[10]:


def sentence2list(sentence):
    return sentence.strip().split()
#组成语料库
sentences_train = list(train_x['word_seg'].apply(sentence2list))


# In[11]:


sentences_test = list(test_x['word_seg'].apply(sentence2list))


# In[12]:


sentences = sentences_train + sentences_test


# In[14]:


from sklearn.svm import SVC, LinearSVC
classifier = LinearSVC()
classifier.fit(train_X, train_y)


# In[17]:


test_Y = classifier.predict(test_X)


# In[19]:


from sklearn.metrics import f1_score
score = f1_score(test_y, test_Y, average='macro')
print("验证集分数：{}".format(score))


# In[22]:


x_test = vectorizer.transform(test_df['word_seg'])


# In[23]:


y_test = classifier.predict(x_test)


# In[27]:


test_df['class'] = y_test.tolist()
df_result = test_df.loc[:, ['id', 'class']]


# In[28]:


df_result.to_csv('./beginner.csv', index=False)

