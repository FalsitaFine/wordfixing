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
import re
import sets
import random

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



max_sentence_len = 10
readf = open("./general_out.txt",'r')
line = readf.readline()

sentences = []
while(line):
  linex = line.replace("\n",'')
  linex = linex.lower()
  if linex != '':
    #if len(line.split()) < max_sentence_len:
        #print(len(line.split()))
    line_spilt = linex.split()
    index = max_sentence_len
    while(index < len(line_spilt)):
      sentences.append(line_spilt[index - max_sentence_len : index])
      #print(line_spilt[index - max_sentence_len : index])
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
print(pretrained_weights)
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
train_x = np.zeros([len(sentences), max_sentence_len-1], dtype=np.int32)
train_y = np.zeros([len(sentences)], dtype=np.int32)
for i, sentence in enumerate(sentences):
  for t, word in enumerate(sentence[:-1]):
    #print("x:", i,t,word)
    train_x[i, t] = word2idx(word)
  #print("y:", i,sentence[-1])
  train_y[i] = word2idx(sentence[-1])


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






model_dual = Sequential()
model_dual.add(Embedding(input_dim=vocab_size, output_dim=emdedding_size, weights=[pretrained_weights]))
#model.add(Bidirectional(LSTM(rnn_size, activation="relu"),input_shape=(seq_length, vocab_size)))
#model.add(LSTM(units=emdedding_size))
model_dual.add(Bidirectional(LSTM(units=emdedding_size)))

model_dual.add(Dropout(0.6))
model_dual.add(Dense(units=vocab_size))
model_dual.add(Activation('softmax'))


model_dual.compile(optimizer='adam', loss='sparse_categorical_crossentropy')





def generate_next(text, topn, ref_word):
  metaphone = Metaphone()
  word_idxs = [word2idx(word) for word in text.lower().split()]
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
    prediction[i] /= ((16**metaphone.distance(ref_word,idx2word(i)))+1)

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


def generate_next_dual(text,text_bwd, topn, ref_word):
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


def predicting(text_in, text_bwd, topn, ref, mode):
    if mode == "dual":
        result,firstresult = generate_next_dual(text_in, text_bwd, topn, ref)
    else:
        result,firstresult = generate_next(text_in, topn, ref)

    return result, firstresult



saved_model = "./model_new/lstm_model_web_bid"
saved_model_dual = "./model_new/lstm_model_web_bid_dualinput"


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
  model.load_weights(saved_model)
except:
  print("No existed model")


try:
  model.load_weights(saved_model_dual)
except:
  print("No existed model")








f = open("general_out.txt", 'r')
text = f.read()
f.close()

text = text.lower()
text = text.replace('\n','')

words = re.split('[^A-Za-z]+', text)



def contains(small, big):
    for i in xrange(len(big)-len(small)+1):
        for j in xrange(len(small)):
            if big[i+j] != small[j]:
                break
        else:
            return i, i+len(small)
    return False



    
def get2GramSentenceRandom(word, n = 50):
    for i in xrange(n):
        #print word,
        choices = [element for element in gram2 if element[0][0] == word]
        if not choices:
            break
        
        word = weighted_choice(choices)[1]





def generateNgram(n):
    gram = dict()
        
    for i in range(len(words)-(n-1)):
        key = tuple(words[i:i+n])
        if key in gram:
            gram[key] += 1
        else:
            gram[key] = 1

    if n == 6:
        #print(gram)
        #print(gram)
        gram = sorted(gram.items(), key=lambda x: -x[1])
        #print(gram)
    else:
        gram = sorted(gram.items(), key=lambda x: -x[1])

    return gram


def getNGramSentenceRandom(gram, word, n = 50):
    for i in range(n):
        #print word,
        choices = [element for element in gram if element[0][0] == word]
        #print choices
        #print (element)
        #print (">>>>")
        #print (choices)

        if not choices:
            break
        #word = weighted_choice(choices)[1]


def longSentence(gramlist, words, n = 50):
    gramlen = len(gramlist)
    sentenlen = len(words)

    choice = []
    #choice[i]: [(word), point]

    longest_n = min(sentenlen+1,gramlen)
    while(longest_n > 1):
    #print longest_n, gramlist[longest_n][0]
    #choices = [element for element in gramlist[longest_n-1] if contains(words[sentenlen - longest_n : sentenlen], list(element[0]))]
    #print len(element[0]), len(gramlist[longest_n-2][0][0]),len(words[sentenlen - longest_n : sentenlen])
    #print list(element[0]), gramlist[longest_n-2][0],list(words[sentenlen - longest_n : sentenlen])
        for gram in gramlist[longest_n-2]:
            #print(gram)
            #print gram[0][:longest_n], words[-longest_n:]
            flag = 1
            for i in range(longest_n-1):
                #print(gram[0][i])
                #print(words[len(words)-longest_n+1+i])
                if gram[0][i] != words[len(words)-longest_n+1+i]:
                    flag = 0
                #else:
                #    print(gram[0][i])
            if flag == 1:
                #print gram[0][:longest_n], words[-longest_n:]
                #if gram[0] in choice[:][0]:
                choice.append(gram)

            #print len(gram[0][:longest_n-1]), len(words[-longest_n+1:])
            #if(gram[0][:longest_n] == words[-longest_n:]):
        #if len(choice)>0:
        #    break
        longest_n -= 1
    return choice


#FIND ONE BLOCK FOR THE WORD
def longSentenceBiDir(gramlist, words, tofixindex, n = 50):
    gramlen = len(gramlist)
    sentenlen = len(words)

    choice = []
    choicebkg = []

    #choice[i]: [(word), point]

    longest_n = min(tofixindex+2,gramlen)


    #FORWARD
    while(longest_n > 1):
    #print longest_n, gramlist[longest_n][0]
    #choices = [element for element in gramlist[longest_n-1] if contains(words[sentenlen - longest_n : sentenlen], list(element[0]))]
    #print len(element[0]), len(gramlist[longest_n-2][0][0]),len(words[sentenlen - longest_n : sentenlen])
    #print list(element[0]), gramlist[longest_n-2][0],list(words[sentenlen - longest_n : sentenlen])
        for gram in gramlist[longest_n-2]:
            #print(gram)
            #print gram[0][:longest_n], words[-longest_n:]
            flag = 1
            for i in range(longest_n-1):
                #print(gram[0][i])
                #print(words[len(words)-longest_n+1+i])
                if gram[0][i] != words[tofixindex -longest_n+1+i]:
                    flag = 0
                #else:
                #    print(gram[0][i])
            if flag == 1:
                #print gram[0][:longest_n], words[-longest_n:]
                #if gram[0] in choice[:][0]:
                choice.append(gram)
                #print("FWD adding", gram)

            #print len(gram[0][:longest_n-1]), len(words[-longest_n+1:])
            #if(gram[0][:longest_n] == words[-longest_n:]):
        #if len(choice)>0:
        #    break
        longest_n -= 1


    #BACKWARD
    longest_n = min((sentenlen-tofixindex),gramlen) 

    while(longest_n > 1):
    #print longest_n, gramlist[longest_n][0]
    #choices = [element for element in gramlist[longest_n-1] if contains(words[sentenlen - longest_n : sentenlen], list(element[0]))]
    #print len(element[0]), len(gramlist[longest_n-2][0][0]),len(words[sentenlen - longest_n : sentenlen])
    #print list(element[0]), gramlist[longest_n-2][0],list(words[sentenlen - longest_n : sentenlen])
        for gram in gramlist[longest_n-2]:
            #print(gram)
            #print gram[0][:longest_n], words[-longest_n:]
            flag = 1

            for i in range(longest_n-1):
                #print(gram[0][i])
                #print(words[len(words)-longest_n+1+i])
                if gram[0][i+1] != words[tofixindex+i+1]:
                    flag = 0
                #else:
                #    print(gram[0][i])
            if flag == 1:
                #print gram[0][:longest_n], words[-longest_n:]
                #if gram[0] in choice[:][0]:
                #print("BKG adding", gram)
                choicebkg.append(gram)

            #print len(gram[0][:longest_n-1]), len(words[-longest_n+1:])
            #if(gram[0][:longest_n] == words[-longest_n:]):
        #if len(choice)>0:
        #    break
        longest_n -= 1
    return choice,choicebkg



def genePoint(choices,word_to_fix):
    metaphone = Metaphone()
    wordlist = []
    pointList = [[]]
    for choice in choices:
        if not choice[0][-1] in wordlist:
            wordlist.append(choice[0][-1])
            pointList.append([choice[0][-1],0])

        for point in pointList:
            #print(point)
            if (len(point)>0):
                if choice[0][-1] == point[0]:
                    #print(len(choice[0]), " ",choice[1])
                    #print(word_to_fix," ", choice[0][-1]," ",metaphone.distance(word_to_fix,choice[0][-1])**2+1)

                    point[1] += (8**len(choice[0]))*choice[1]/((4**metaphone.distance(word_to_fix,choice[0][-1]))+1);
    pointList.pop(0)
    #print(pointList)
    pointList.sort(key=lambda x: x[1], reverse = True)
    print(pointList[0:10])
    if len(pointList)>0:
        if len(pointList[0])>1:
            try:
                return pointList[0][0],pointList[0][1]/pointList[1][1]
            except:
                return "NONE",0

        elif len(pointList)>0:
            if len(pointList[0])>0:
                try:
                    return pointList[0][0],pointList[0][1]
                except:
                    return "NONE",0

    return "NONE",0



def genePointList(choices,word_to_fix):
    metaphone = Metaphone()
    wordlist = []
    pointList = [[]]
    for choice in choices:
        if not choice[0][-1] in wordlist:
            wordlist.append(choice[0][-1])
            pointList.append([choice[0][-1],0])

        for point in pointList:
            #print(point)
            if (len(point)>0):
                if choice[0][-1] == point[0]:
                    #print(len(choice[0]), " ",choice[1])
                    #print(word_to_fix," ", choice[0][-1]," ",metaphone.distance(word_to_fix,choice[0][-1])**2+1)

                    point[1] += (8**len(choice[0]))*choice[1]/((4**metaphone.distance(word_to_fix,choice[0][-1]))+1);
    pointList.pop(0)
    #print(pointList)
    pointList.sort(key=lambda x: x[1], reverse = True)
    if(len(pointList)>20):
        return pointList[0:20]
    elif len(pointList)>1:
        if len(pointList[0])>0:
            try:
                return pointList[0][0],pointList[0][1]/pointList[1][1]
            except:
                return "NONE",0
    elif len(pointList)>0:
        if len(pointList[0])>0:
            try:
                return pointList[0][0],pointList[0][1]
            except:
                return "NONE",0


    return "NONE",0



def genePointBidir(choices,choicesbkg,word_to_fix):
    metaphone = Metaphone()
    wordlist = []
    pointList = [[]]
    for choice in choices:
        if not choice[0][-1] in wordlist:
            wordlist.append(choice[0][-1])
            pointList.append([choice[0][-1],0])

        for point in pointList:
            #print(point)
            if (len(point)>0):
                if choice[0][-1] == point[0]:
                    #print(len(choice[0]), " ",choice[1])
                    #print(word_to_fix," ", choice[0][-1]," ",metaphone.distance(word_to_fix,choice[0][-1])**2+1)

                    point[1] += (8**len(choice[0]))*choice[1]/((6**metaphone.distance(word_to_fix,choice[0][-1]))+1);

    
    for choice in choicesbkg:
        if not choice[0][0] in wordlist:
            wordlist.append(choice[0][0])
            pointList.append([choice[0][0],0])

        for point in pointList:
            #print(point)
            if (len(point)>0):
                if choice[0][0] == point[0]:
                    #print(len(choice[0]), " ",choice[1])
                    #print(word_to_fix," ", choice[0][-1]," ",metaphone.distance(word_to_fix,choice[0][-1])**2+1)

                    point[1] += (4**len(choice[0]))*choice[1]/((8**metaphone.distance(word_to_fix,choice[0][0]))+1)/4;
    
    pointList.pop(0)
    #print(pointList)
    pointList.sort(key=lambda x: x[1], reverse = True)
    print(pointList[0:10])
    if len(pointList)>1:
        if len(pointList[0])>0:
            try:
                return pointList[0][0],pointList[0][1]/pointList[1][1]
            except:
                return "NONE",0
    elif len(pointList)>0:
        if len(pointList[0])>0:
            try:
                return pointList[0][0],pointList[0][1]
            except:
                return "NONE",0


    return "NONE",0





gramlist = []
max_gram = 12
for i in range(2, max_gram):
    gramlist.append(generateNgram(i))





error_counter = 0
fix_counter_lstm = 0
fix_counter_lstm_dual = 0
inlist_counter_lstm = 0
inlist_counter_lstm_dual = 0
fix_counter_ng = 0
fix_counter_ng_bi = 0

#MAIN
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
                error_counter += 1
                current_word_duallstm = testtxt[:i]
                word_to_correct_duallstm = i
                if i <= (max_gram - 1):
                  current_word_ng = testtxt[:i]
                  word_to_correct = i
                  word_to_correct = testtxt[i]
                  word_to_correct_fwdng = testtxt[i]

                else:
                  current_word_ng = testtxt[i-max_gram+1:i]
                  word_to_correct_fwdng = testtxt[i]

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



                  print("To fixing: ", current_word,testtxt[word_to_correct])
                  result,resu = predicting(" ".join(current_word),"",20,testtxt[word_to_correct],"fwd")
                  choice,choicebkg = longSentenceBiDir(gramlist, testtxt, word_to_correct, 1)
                  resu_ng,confi_ng = genePointBidir(choice, choicebkg, testtxt[word_to_correct])
                  choice_fwdng = longSentence(gramlist, current_word_ng, 1)
                  resu_fwdng,confi_fwdng = genePoint(choice,word_to_correct_fwdng)
                  result_duallstm,resu_duallstm = predicting(" ".join(current_word_duallstm),bwd_word,20,testtxt[word_to_correct_duallstm],"dual")


                  correctresult = testtxtr[i].lower()
                  correctresult = correctresult.replace(" ",'')
                  print("Correct word: ", correctresult)

                  print(" Fixing result(Bi-LSTM): ", resu, "         Confidence: " ,result[0][1])
                  print(" Fixing result(Bi-LSTM-Dual): ", resu_duallstm, "         Confidence: " ,result_duallstm[0][1])
                  print(" Fixing result(Bi-NG): ", resu_ng, "         Confidence: " ,confi_ng)
                  print(" Fixing result(NG): ", resu_fwdng, "         Confidence: " ,confi_fwdng)


                  if correctresult == resu:
                      print("CORRECT(LSTM)")
                      fix_counter_lstm += 1
                  else:
                      inlistflag = 0
                      for ins in result:
                        if ins[0] == correctresult:
                          print("NOT CORRECT(LSTM), BUT IN FIRST 20 PREDICTIONS( RANK ",i,")")
                          inlistflag = 1
                          inlist_counter_lstm += 1
                          break
                      if inlistflag == 0:
                        print("NOT CORRECT(LSTM)")

                  if correctresult == resu_duallstm:
                      print("CORRECT(LSTM-dual)")
                      fix_counter_lstm_dual += 1
                  else:
                      inlistflag = 0
                      for ins in result_duallstm:
                        if ins[0] == correctresult:
                          print("NOT CORRECT(LSTM-dual), BUT IN FIRST 20 PREDICTIONS( RANK ",i,")")
                          inlistflag = 1
                          inlist_counter_lstm_dual += 1
                          break
                      if inlistflag == 0:
                        print("NOT CORRECT(LSTM-dual)")


                  if correctresult == resu_ng:
                    print("CORRECT(Bi-NG)")
                    fix_counter_ng_bi += 1
                  else:
                    print("NOT CORRECT(Bi-NG)") 

                  if correctresult == resu_fwdng:
                    print("CORRECT(NG)")
                    fix_counter_ng += 1
                  else:
                    print("NOT CORRECT(NG)") 

                  if confi_ng > 2:
                      print("fix-->",testtxt[i] ," to ",resu_ng)
                      testtxt[i] = resu


        testtxt = testf.readline()
        testtxtr = testr.readline()




  print("RESULT ON TESTBENCH ------------>>>>>>>    ")

  print("TOTAL ERR:" , error_counter)
  print("LSTM FIX:" , fix_counter_lstm)
  print("LSTM INLIST:" , inlist_counter_lstm)
  print("DUAL-LSTM FIX:" , fix_counter_lstm_dual)
  print("DUAL-LSTM INLIST:" , inlist_counter_lstm_dual)
  print("NG FIX:" , fix_counter_ng)
  print("BI-NG FIX:" , fix_counter_ng_bi)


if len(sys.argv) == 3:
  result,firstresult = predicting(sys.argv[1],30,sys.argv[2])
  print("-------->",result)

if(len(sys.argv) == 1):
  #for i in range(len(train_x)):
  #  print(train_x[i],train_y[i])
  
  model.fit(train_x, train_y,
            batch_size=128,
            epochs=1000,
            callbacks=[save_callback])
  
  model.summary()


