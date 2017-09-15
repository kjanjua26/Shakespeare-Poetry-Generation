'''
	S1: Input data
	S2: Normalize it. 
	S3: Split.
	S4: Model Train using Keras.
	S5: Predict

	Output: 
		Generated Shakespeare writing. 
'''
import numpy as np 
import keras 
from keras.models import Sequential
from keras.layers import Dense, LSTM, Activation, TimeDistributed, Dropout
from keras.models import load_model
from keras.utils import np_utils, to_categorical
#Reading the data
data = open('input.txt').read()
data = data.lower()
num_c = len(data)
print "The length of text is: ", num_c

#Normalizing it. 
chars = sorted(list(set(data))) #Distinct characters
vocab = len(chars)
print "The total characters are: ", vocab
ci = dict((c,i) for i,c in enumerate(chars)) #chars to int
ic = dict((i,c) for i,c in enumerate(chars)) #int to chars

seq_len = 100 #Sequence Length, constant
gen_chars = 200 #Essay to generate
x = []
y = []

#formatting the data
for i in range(0, num_c - seq_len, 1):
	s_in = data[i:i + seq_len]
	s_out = data[i + seq_len]
	x.append([ci[char] for char in s_in])
	y.append(ci[s_out])

x_len = len(x)
print "The x_len is: ", x_len
x_train = np.reshape(x, (x_len, seq_len,1)) #LSTM Input Reshaping
x_train = x_train / float(vocab) #Further normalization
y_train = np_utils.to_categorical(y)
print "X_train Shape: ", x_train.shape 
print "Y_train Shape: ", y_train.shape 

def model():
	model = Sequential()
	model.add(LSTM(256, input_shape=(x_train.shape[1], x_train.shape[2]), return_sequences=True))
	model.add(Dropout(0.2))
	model.add(LSTM(256))
	model.add(Dropout(0.2))
	model.add(Dense(y_train.shape[1]))
	model.add(Activation('softmax'))
	return model

model = model()
model.compile(loss='categorical_crossentropy', optimizer='adam')
model.fit(x_train,y_train,epochs=40,batch_size=64,verbose=1)
model.save('weights.h5')

print "Saved Model."

