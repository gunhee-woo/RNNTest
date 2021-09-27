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
from sklearn.metrics import classification_report

# fix random seed for reproducibility
numpy.random.seed(7)

#fetching sms spam dataset
url = os.getcwd() + "\\data5.csv"
#sms = pd.read_table(url, header=None, names=['data', 'label'])
sms = pd.read_table(url, sep=",", engine='python', header=None, encoding="utf-8", names=['data', 'label'])
re = sms.isna()
sms.dropna(inplace=True)

# sms.drop(sms.index[0])
# sms.reset_index()

#binarizing
sms['label_num'] = sms.label.map({0:0, 1:1})
sms.head()

X = sms.data #원본 데이터
y = sms.label_num # 원본 라벨

# 원본데이터 8:2 비율로 나눔
# X_train, X_test = train_test_split(X, test_size=0.2, random_state=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# 전처리과정 => 입력할 수 있는 데이터 형태(수치형 텐서)로 변환 => 텍스트 벡터화(토큰화)
###################################
# 가장 빈도가 높은 최대 3000개 단어 선택
tk = tf.keras.preprocessing.text.Tokenizer(nb_words=3000, lower=False)
tk.fit_on_texts(X_train.values)
# 단어 인덱스 구축
x_train = tk.texts_to_sequences(X_train.values)
# 문자열을 정수 인덱스의 리스트로 변환

tk.fit_on_texts(X_test.values)
x_test = tk.texts_to_sequences(X_test.values)

###################################
# 패딩 작업 => 모두 같은 길이의 정수 텐서를 만듬 => 이 텐서를 다루는 임베딩 층을 신경망의 첫번째 층으로 사용
max_len = 3000
# 모든 정수 배열의 길이를 3000으로 맞춘다
x_train = tf.keras.preprocessing.sequence.pad_sequences(x_train, maxlen=max_len)
x_test = tf.keras.preprocessing.sequence.pad_sequences(x_test, maxlen=max_len)

checkpoint_path = "training_1/cp.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

# 체크포인트 콜백 만들기
cp_callback = tf.keras.callbacks.ModelCheckpoint(checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=1)
max_features = 3000
model = tf.keras.Sequential()
print('Build model...')

model = tf.keras.Sequential()
model.add(tf.keras.layers.Embedding(max_features, 128, input_length=max_len))
model.add(tf.keras.layers.LSTM(128, dropout=0.2))
model.add(tf.keras.layers.Dense(1))
model.add(tf.keras.layers.Activation('sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

# model.fit(x, y=y, batch_size=100, nb_epoch=1, verbose=1, validation_split=0.2, shuffle=True)
# model.fit(x_train, y_train, batch_size=200, epochs=1, verbose=1, validation_data=(x_test, y_test), shuffle=True, callbacks=[cp_callback])

model.load_weights(checkpoint_path)
loss, acc = model.evaluate(x_test, y_test, verbose=2)
print("복원된 모델의 정확도: {:5.2f}%".format(100*acc))

# model.save("\\result.h5")
# model.save('my_model.h5')
# model_load = tf.keras.models.load_model(os.getcwd() + "\\result.h5")
# scores = model_load.evaluate(x_test, y_test, verbose=2)
# predict = model_load.predict(x_test)
predict = model.predict(x_test)

# predict_lst = []
#    for item in predict:
#        predict_lst.append(np.argmax(item))
predict_bool = numpy.argmax(predict, axis=1)
print(classification_report(y_test, predict_bool))
