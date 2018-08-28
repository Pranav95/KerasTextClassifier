
# coding: utf-8

# In[1]:


import keras 
import pandas as pd 
import numpy as np

import csv 
table = []
header = ['user', 'text','label']
with open('finale_pro.csv', 'r') as csvfile:
    spamreader = csv.reader(csvfile, delimiter = ',', quotechar= '|')
    for row in spamreader:
        if(len(row)>3):
          print(row)
        table.append(row)



# In[2]:


df  = pd.DataFrame(table, columns=header)
print(df.size)
#df.label.replace(['anti'], ['pro'], inplace=True)

#df = df.drop_duplicates(subset =['text'], keep = False)
table = []
with open('finale_anti.csv','r') as csvfile:
    spamreader = csv.reader(csvfile,delimiter = ',', quotechar = '|')
    for row in spamreader:
        table.append(row)


# In[3]:




df2 = pd.DataFrame(table, columns = header)

print(df2.size)




# In[4]:


from sklearn.utils import shuffle
df = shuffle(df)
df2 = shuffle(df2)


# In[5]:


df = df[:df2.size]
df2 = shuffle(df2)
df = df.append(df2, ignore_index=True)
df = shuffle(df)


# In[8]:


from nltk.corpus import stopwords


# In[10]:


import nltk
nltk.download('stopwords')
df


# In[11]:


stop = stopwords.words('english')


# In[12]:


df['text2'] = df['text'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))


# In[13]:
from keras.preprocessing import text
tokenize = text.Tokenizer(num_words=5000)

temp_text = df['text2']
tokenize.fit_on_texts(temp_text)
temp_text = tokenize.texts_to_matrix(temp_text)


temp_label = df['label']
from sklearn.preprocessing import LabelBinarizer

encoder = LabelBinarizer()
encoder.fit(temp_label)





temp_label = encoder.transform(temp_label)


from keras.utils import to_categorical
temp_label = to_categorical(temp_label)


print(temp_label.shape)



train_size= int(len(df)* .8)
train_text = temp_text[:train_size]
train_label = temp_label[:train_size]

test_text = temp_text[train_size+1:]
test_label = temp_label[train_size+1:]



print(train_label.shape)
exit()

# In[14]:





# In[15]:


#tokenize.fit_on_texts(train_text)
#x_train = tokenize.texts_to_matrix(train_text)


# In[16]:


'''from sklearn.preprocessing import LabelBinarizer

encoder = LabelBinarizer()
encoder.fit(train_label)


train_label = encoder.transform(train_label)


from keras.utils import to_categorical
train_label = to_categorical(train_label)


    
#tokenize.fit_on_texts(test_text)
#x_test = tokenize.texts_to_matrix(test_text)

from sklearn.preprocessing import LabelBinarizer
encoder = LabelBinarizer()
encoder.fit(test_label)

test_label = encoder.transform(test_label)


from keras.utils import to_categorical
test_label = to_categorical(test_label)
'''

# In[ ]:


from keras import Sequential
from keras.layers.core import Dense, Activation, Dropout

model = Sequential()
model.add(Dense(512, input_shape=(5000,)))
model.add(Activation('relu'))
model.add(Dropout(0.3))

model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.3))

model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.3))

model.add(Dense(2))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy', 
              optimizer='adam', 
              metrics=['accuracy'])
history = model.fit(train_text, train_label, 
                    batch_size=100, 
                    epochs=50, 
                    verbose=1, 
                    validation_split=0.1)


# In[69]:


score = model.evaluate(test_text, test_label, 
                       batch_size=100, verbose=1)
print(score[0])
print(score[1])
for s in score:
    print(score)

