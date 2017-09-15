import numpy as np 
import keras 
from keras.models import Sequential
from keras.layers import Dense, LSTM, Activation, TimeDistributed, Dropout
from keras.models import load_model
from keras.utils import np_utils, to_categorical
import sys

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
gen_chars = 10 #Essay to generate
x = []
y = []

#formatting the data
for i in range(0, num_c - seq_len, 1):
	s_in = data[i:i + seq_len]
	s_out = data[i + seq_len]
	x.append([ci[char] for char in s_in])
	y.append(ci[s_out])

x_len = len(x)

model = load_model('weights.h5')
model.compile(loss='categorical_crossentropy', optimizer='adam')

#Random Seed
seed = np.random.randint(0, x_len-1)
p = x[seed]

for i in range(gen_chars):
	x_valid = np.reshape(p, (1, len(p), 1))
	x_valid = x_valid / float(vocab)
	out = model.predict(x_valid, verbose=0)
	index = np.argmax(out) #inverting
	res = ic[index]
	s_in = [ic[value] for value in p]
	res = ''.join(res)
	print res, 
	file = open('res.txt', 'w')
	file.write(res)
	p.append(index)
	p = p[1:len(p)]
print "\nOutput Generated."

