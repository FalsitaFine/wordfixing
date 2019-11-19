import sys
import numpy as np
import gensim
import string
import tensorflow as tf
from keras.callbacks import LambdaCallback
from keras.layers.embeddings import Embedding
from keras.layers import Dense, Activation, Bidirectional, Dropout, Input, LSTM
from keras.models import Sequential, Model
from keras.utils.data_utils import get_file
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from keras.metrics import categorical_accuracy


from pyphonetics import Soundex
from pyphonetics import Metaphone



def get_levenshtein_distance(word1, word2):

    word2 = word2.lower()
    word1 = word1.lower()
    matrix = [[0 for x in range(len(word2) + 1)] for x in range(len(word1) + 1)]

    for x in range(len(word1) + 1):
        matrix[x][0] = x
    for y in range(len(word2) + 1):
        matrix[0][y] = y

    for x in range(1, len(word1) + 1):
        for y in range(1, len(word2) + 1):
            if word1[x - 1] == word2[y - 1]:
                matrix[x][y] = min(
                    matrix[x - 1][y] + 1,
                    matrix[x - 1][y - 1],
                    matrix[x][y - 1] + 1
                )
            else:
                matrix[x][y] = min(
                    matrix[x - 1][y] + 1,
                    matrix[x - 1][y - 1] + 1,
                    matrix[x][y - 1] + 1
                )

    return matrix[len(word1)][len(word2)]



max_sentence_len_fwd = 8
max_sentence_len_bwd = 4
readf = open("./general_out.txt",'r')
line = readf.readline()

sentences = []
sentences_bwd = []
while(line):
  linex = line.replace("\n",'')
  linex = linex.lower()
  if linex != '':
    #if len(line.split()) < max_sentence_len:
        #print(len(line.split()))
    line_spilt = linex.split()
    index = max_sentence_len_fwd
    while(index < len(line_spilt)):
      sentences.append(line_spilt[index - max_sentence_len_fwd : index])
      bwd_index = max_sentence_len_bwd

      #print(line_spilt[index - max_sentence_len : index])
      while(index + bwd_index > len(line_spilt)):
        bwd_index -= 1
      sentences_bwd.append(line_spilt[index : index + bwd_index])
      index+=1

  line = readf.readline()

#print(sentences)
'''
docs = readf.readlines()
sentences = [[word for word in doc.lower().translate(string.punctuation).split()[:max_sentence_len]] for doc in docs]
'''
#print(len(sentences))



#word_model = gensim.models.Word2Vec(sentences, size=128, min_count=3, window=15, iter=300)
#word_model.save("word2vec_bid.model")

word_model = gensim.models.Word2Vec.load("word2vec.model")



pretrained_weights = word_model.wv.syn0
#print(pretrained_weights)
vocab_size, emdedding_size = pretrained_weights.shape

#print("___",len(sentences))
def word2idx(word):
  if word in word_model.wv.vocab:
    return word_model.wv.vocab[word].index
  else:
    return 0
def idx2word(idx):
  return word_model.wv.index2word[idx]

#print(len(sentences))
train_x = np.zeros([len(sentences), max_sentence_len_fwd-1], dtype=np.int32)
train_xb = np.zeros([len(sentences_bwd), max_sentence_len_bwd], dtype=np.int32)
train_xc = np.zeros([len(sentences), max_sentence_len_fwd-1 + max_sentence_len_bwd], dtype=np.int32)
train_y = np.zeros([len(sentences)], dtype=np.int32)


print("XB,X", len(sentences_bwd),len(sentences))


for i, sentence in enumerate(sentences):
  for t, word in enumerate(sentence[:-1]):
    #print("x:", i,t,word)
    train_x[i, t] = word2idx(word) 
    train_xc[i, t] = word2idx(word)  
  #print("y:", i,sentence[-1])
  train_y[i] = word2idx(sentence[-1])
for i, sentence in enumerate(sentences_bwd):
     
    for t, word in enumerate(sentence):
      train_xb[i, t] = word2idx(word)
      train_xc[i, t+max_sentence_len_fwd-1] = word2idx(word)

      index_blank = t
      #print("xb:", i,t,sentence[-1] )   
    while (index_blank < max_sentence_len_bwd - 1):
      index_blank += 1
      train_xb[i,index_blank] = -1
      train_xc[i, index_blank+max_sentence_len_fwd-1] = word2idx(word)

      #print("ADDING BLANK")


for i in range(10):
  print(train_x[i])
  for tx in range(len(train_x[i])):
    print(idx2word(train_x[i,tx]))
  print(">>>")
  print(train_y[i])

  print(idx2word(train_y[i]))
  
  print("<<<")
  print(train_xb[i])
  for tx in range(len(train_xb[i])):
    print(idx2word(train_xb[i,tx]))
print(">>>>",emdedding_size)


'''
def bidirectional_lstm_model(seq_length, vocab_size):
    print('Build LSTM model.')
    model = Sequential()
    model.add(Bidirectional(LSTM(rnn_size, activation="relu"),input_shape=(seq_length, vocab_size)))
    model.add(Dropout(0.6))
    model.add(Dense(vocab_size))
    model.add(Activation('softmax'))
    
    optimizer = Adam(lr=learning_rate)
    callbacks=[EarlyStopping(patience=2, monitor='val_loss')]
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=[categorical_accuracy])
    print("model built!")
    return model
'''
'''

rnn_size = 256 # size of RNN
seq_length = 30 # sequence length
learning_rate = 0.001 #learning rate

md=model = bidirectional_lstm_model(seq_length, vocab_size)
'''
model = Sequential()

model.add(Embedding(input_dim=vocab_size, output_dim=emdedding_size, weights=[pretrained_weights]))
#model.add(Bidirectional(LSTM(rnn_size, activation="relu"),input_shape=(seq_length, vocab_size)))
#model.add(LSTM(units=emdedding_size))

model.add(Bidirectional(LSTM(units=emdedding_size)))

model.add(Dropout(0.6))
model.add(Dense(units=vocab_size))
model.add(Activation('softmax'))


model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')


def generate_next(text,text_bwd, topn, ref_word):
  metaphone = Metaphone()

  text_add = text + text_bwd

  text_split = text_add.split()
  word_idxs = []
  for word in text_split:
    if word == "-blank":
      word_idxs.append(-1)
    else:
      word_idxs.append(word2idx(word.lower()))

  #word_idxs = [word2idx(word) for word in text_add.lower().split()]
  #word_idxs = [word2idx(word) for word in text.lower().split()]
  #word_bwd_idxs = [word2idx(word) for word in text_bwd.lower().split()]
  #print("word,wordidx",text,word_idxs)
  result_list = []

  prediction = model.predict(x=np.array(word_idxs))

  #print(prediction.shape)
  


  '''
  for x in range(6):
    print(prediction[x])
    debug_resu = np.argmax(prediction[x])
    print(idx2word(debug_resu))
  '''


  prediction = prediction[-1].ravel()
  #print(prediction.shape)
  prediction = prediction[-vocab_size:]
  #print(prediction.shape)

  for i in range(len(prediction)):
    prediction[i] /= ((8**metaphone.distance(ref_word,idx2word(i))))

  pre_result = np.argmax(prediction)
  #pre_result = pre_result % vocab_size
  result_list.append([idx2word(pre_result),prediction[pre_result]])

  #print(pre_result,prediction[pre_result])
  #print(idx2word(pre_result))

  #print(pre_result,prediction[pre_result])
  firstresult = idx2word(pre_result)
  #print(idx2word(pre_result))

  prediction_x = prediction
  prediction_x[pre_result] = 0


  pre_result = np.argmax(prediction_x)
  pre_result = pre_result % vocab_size
  #print(pre_result,prediction_x[pre_result])
  #print(idx2word(pre_result))
  result_list.append([idx2word(pre_result),prediction_x[pre_result]])

  #print(pre_result,prediction_x[pre_result])
  #print(idx2word(pre_result))

  while(topn >= 2):
    prediction_x[pre_result] = 0
    #print(prediction_x.shape,vocab_size)
    pre_result = np.argmax(prediction_x)
    pre_result = pre_result % vocab_size

    #print(pre_result,prediction_x[pre_result])
    #print(idx2word(pre_result))
    result_list.append([idx2word(pre_result),prediction_x[pre_result]])

    topn += -1

  #print(result_list)


  return result_list, firstresult

def predicting(text_in, text_bwd, topn, ref):
  result,firstresult = generate_next(text_in, text_bwd, topn, ref)
  return result, firstresult



saved_model = "./model_new/lstm_model_web_bid_dualinput"


save_callback=tf.keras.callbacks.ModelCheckpoint(
    filepath=saved_model,
    save_weights_only=True)


#latest = tf.train.latest_checkpoint("./saved_model_combine")

'''
try:
  model.load_weights(latest)
  print(latest, " loaded")
except:
  print("No model loaded")
'''


#print(latest, " loaded")

try:
  print("Detected model, training...")
  model.load_weights(saved_model)
except:
  print("No existed model")


print("MODE",len(sys.argv))
if len(sys.argv) == 2:
  if sys.argv[1] == "testbench":

    testf = open("mutated.txt",'r')
    testr = open("mutated_ref.txt",'r')

    testtxt = testf.readline()
    #testtxt = testtxt.split()

    testtxtr = testr.readline()
    #testtxtr = testtxtr.split()

    current_word = ''

    while(testtxt):
        testtxt = testtxt.split()
        testtxtr = testtxtr.split()

        for i in range(len(testtxt)):
            if testtxt[i] != testtxtr[i]:
                current_word = testtxt[:i]
                word_to_correct = i
                if (i != 0):

                  bwd_index = 0
                  bwd_word = ""
                  while i + bwd_index + 1 < len(testtxt):
                    bwd_index += 1
                    if (bwd_index == 1):
                      bwd_word = testtxt[i+1]
                    elif (bwd_index <= 4):
                      bwd_word = bwd_word + " " + testtxt[i + bwd_index]
                  while (bwd_index <= 4):
                    bwd_index += 1
                    bwd_word = bwd_word + " -blank" 
                  
                  print("To fixing: FWD>>>", current_word, " WRONG>>>", testtxt[word_to_correct], " BWD>>>",bwd_word)

                  result,resu = predicting(" ".join(current_word),bwd_word,20,testtxt[word_to_correct])
                  print("Correct word: ", testtxtr[i], " Fixing result: ", resu, "         Confidence: " ,result[0][1]/result[1][1])
                  if testtxtr[i] == resu:
                      print("CORRECT")
                      testtxt[i] = resu
                  else:
                      inlistflag = 0
                      for ins in result:
                        if ins[0] == testtxtr[i]:
                          print("NOT CORRECT, BUT IN FIRST 20 PREDICTIONS( RANK ",i,")")
                          inlistflag = 1
                          break
                      if inlistflag == 0:
                        print("NOT CORRECT") 

        testtxt = testf.readline()
        testtxtr = testr.readline()

  else:
    result,firstresult = predicting(sys.argv[1],30,'')
    print("-------->",result)

if len(sys.argv) == 3:
  result,firstresult = predicting(sys.argv[1],30,sys.argv[2])
  print("-------->",result)

if(len(sys.argv) == 1):
  #for i in range(len(train_x)):
  #  print(train_x[i],train_y[i])
  
  model.fit(train_xc, train_y,
            batch_size=128,
            epochs=1000,
            callbacks=[save_callback])
  
  model.summary()


        