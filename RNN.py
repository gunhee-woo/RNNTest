# LSTM with dropout for sequence classification
import os
import numpy
import tensorflow as tf
#import tensorflow.keras.models import Sequential
#import tensorflow.keras.layers import Dense, Activation
#import tensorflow.keras.layers import LSTM
#from keras.preprocessing import sequence,text
#from keras.layers.embeddings import Embedding
import pandas as pd
from sklearn.model_selection import train_test_split


# fix random seed for reproducibility
numpy.random.seed(7)

#fetching sms spam dataset
url = os.getcwd() + "\\data5.csv"
#sms = pd.read_table(url, header=None, names=['data', 'label'])
sms = pd.read_table(url, sep=",", engine='python', header=None, encoding="utf-8", names=['data', 'label'])
re = sms.isna()
sms.dropna(inplace=True)

#sms.drop(sms.index[0])
#sms.reset_index()

#binarizing
sms['label_num'] = sms.label.map({0:0, 1:1})
sms.head()

X = sms.data #원본 데이터
y = sms.label_num # 원본 라벨

#count = int(len(X) * 0.8) # 8:2로 나누는 간단한 방법

print(X.shape)
print(y.shape)

###################################
tk = tf.keras.preprocessing.text.Tokenizer(nb_words=3000, lower=False)
tk.fit_on_texts(X.values)
x = tk.texts_to_sequences(X.values)

###################################
max_len = 3000
x = tf.keras.preprocessing.sequence.pad_sequences(x, maxlen=max_len)



max_features = 3000
model = tf.keras.Sequential()
print('Build model...')

model = tf.keras.Sequential()
model.add(tf.keras.layers.Embedding(max_features, 128, input_length=max_len))
model.add(tf.keras.layers.LSTM(128, dropout=0.2))
model.add(tf.keras.layers.Dense(1))
model.add(tf.keras.layers.Activation('sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

model.fit(x, y=y, batch_size=100, nb_epoch=1, verbose=1, validation_split=0.2, shuffle=True)

#model_load = tf.keras.models.load_model(os.getcwd() + "\\result.h5")
#scores = model_load.evaluate(X_test_t, y_test_labels, verbose=2)
#predict = model_load.predict(X_test_t)
#predict_lst = []
#    for item in predict:
#        predict_lst.append(np.argmax(item))
