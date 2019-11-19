
from __future__ import absolute_import, division, print_function

import tensorflow as tf
from tensorflow import keras

import numpy as np
#import matplotlib.pyplot as plt

import time


PUNCTUATIONS = {".PERIOD": 1, ",COMMA": 2}
PUNCTUATIONS_RE = {0: " ", 1: ".PERIOD", 2:",COMMA"}


vocab_list = []
index_list = []

punc_index = []
word_index = []


vocab_dictionary = {}
#readvocb = open("./raw_data/vocab",'r')
readvocb = open("./raw_light/vocab",'r')


#vocab_dictionary.update({"Three":3})
#print(vocab_dictionary["Three"])

vocab_list.append("*DEFAUT")
vocab_list.append(",COMMA")
vocab_list.append(".PERIOD")
vocab_list.append("?QUESTION")

index_list.append(0)
index_list.append(1)
index_list.append(2)
index_list.append(4)




index = 3
line = readvocb.readline()
while line != "":
    vocab_list.append(line.replace("\n",""))
    index_list.append(index)
    line = readvocb.readline()
    index+=1



vocab_dictionary = dict(zip(vocab_list,index_list))
#print(vocab_dictionary)
#readtrain = open("./raw_data/train.txt",'r')
readtrain = open("./raw_light/general_out.txt",'r')

num_punc = 0
current_index = 0
mark_index = 0


punctuation_temp = " "



word_set_array = [[]]

#X = np.reshape(word_array, (int(len(word_array)/longest_seq), longest_seq, 1))



train_text = []

line = readtrain.readline()

while line != "":
    line = line.replace("\n","")
    train_text.append(line)
    line = readtrain.readline()


#Array Forming
text_array = [[]]
ref_array = []
longest_seq = 0
seq_length = 8



for line in train_text:
    line_split = line.split(" ")
    line_length = len(line_split)
    if line_length > longest_seq:
        longest_seq = line_length



    for i in range(seq_length , line_length - seq_length):
        line_array = []
        word = line_split[i]
        if word in vocab_dictionary:
            ref_array.append(vocab_dictionary[word])
            print(word,vocab_dictionary[word])
        else:
            ref_array.append(vocab_dictionary["*DEFAUT"])
            print(word,vocab_dictionary["*DEFAUT"])

        for j in range(i - seq_length, i):
            word = line_split[j]
            if word in vocab_dictionary:
                line_array.append(vocab_dictionary[word])
                #print(word,vocab_dictionary[word])
            else:
                line_array.append(vocab_dictionary["*DEFAUT"])
                #print(word,vocab_dictionary["*DEFAUT"])
        print(line_array)
        text_array.append(line_array)


text_array.pop(0)
'''

for i in range(len(cancer_array)):
    while len(cancer_array[i]) < longest_seq:
        cancer_array[i].append(0)
    tag_array.append(0)

for i in range(len(ref_array)):
    while len(ref_array[i]) < longest_seq:
        ref_array[i].append(0)
    tag_array.append(1)
'''



text_np = np.array(text_array)
tag_np = np.array(ref_array)
print(text_np)
print(tag_np)




#LSTM
X = text_np
X = np.reshape(X, (len(X), seq_length, 1))

Y = tag_np
#print(X.shape,X)
#print(len(Y),Y)

#for i in range(len(Y)):
#    print(Y[i])

#    print(word_set_array[i])

model = keras.Sequential([
    #keras.layers.Flatten(input_shape=(28, 28)),
    #keras.layers.LSTM(784, activation='tanh', recurrent_activation='hard_sigmoid', use_bias=True, kernel_initializer='glorot_uniform', recurrent_initializer='orthogonal', bias_initializer='zeros', unit_forget_bias=True, kernel_regularizer=None, recurrent_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, recurrent_constraint=None, bias_constraint=None, dropout=0.0, recurrent_dropout=0.0, implementation=1, return_sequences=False, return_state=False, go_backwards=False, stateful=False, unroll=False),
    #keras.layers.Dense(128, activation=tf.nn.relu),
    #keras.layers.Dense(10, activation=tf.nn.softmax)
    keras.layers.LSTM(250,input_shape=(seq_length,1),return_sequences=True),
    keras.layers.LSTM(150,return_sequences=False,go_backwards = True),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(128, activation=tf.nn.relu),
    keras.layers.Dense(len(vocab_list), activation=tf.nn.softmax)

])


#Save the trained model

saved_model = "./saved_model/lstm_model_web"


save_callback=tf.keras.callbacks.ModelCheckpoint(
    filepath=saved_model,
    save_weights_only=True)



print(X.shape)

print(Y.shape)

#print(type(X[1][0][0]))
#print(type(Y[1][0][0]))

model.compile(optimizer='adam', 
              loss='sparse_categorical_crossentropy')

#Load from saved model
latest = tf.train.latest_checkpoint("./saved_model")
print(latest)
model.load_weights(latest)
#model.load_weights(tf.train.latest_checkpoint(saved_model))

#model.build(tf.TensorShape([1,1,1]))

'''
model.fit(X, Y, epochs=20, callbacks=[save_callback])
'''


test_acc = model.evaluate(X, Y)


prediction = model.predict(X)
for i in range(len(prediction)):
    print(text_array[i], "---", ref_array[i], "---", prediction[i])


print('Test ', model.metrics_names, " : ", test_acc)




