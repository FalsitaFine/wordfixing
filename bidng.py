import re
import sets
import random
from pyphonetics import Soundex
from pyphonetics import Metaphone
model_ngram = open("model_ngram",'a')

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
        if len(pointList[0])>0:
            return pointList[0][0],pointList[0][1]/pointList[1][1]
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
    elif len(pointList)>0:
        if len(pointList[0])>0:
            return pointList[0][0],pointList[0][1]/pointList[1][1]
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

                    point[1] += (8**len(choice[0]))*choice[1]/((4**metaphone.distance(word_to_fix,choice[0][-1])));

    
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

                    point[1] += (8**len(choice[0]))*choice[1]/((8**metaphone.distance(word_to_fix,choice[0][0])))/4;
    
    pointList.pop(0)
    #print(pointList)
    pointList.sort(key=lambda x: x[1], reverse = True)
    print(pointList[0:10])
    if len(pointList)>0:
        if len(pointList[0])>0:
            return pointList[0][0],pointList[0][1]/pointList[1][1]
    return "NONE",0





gramlist = []
max_gram = 12
for i in range(2, max_gram):
    gramlist.append(generateNgram(i))


#model_ngram.write(gramlist)
'''
test_sentence = "a doctor can help determine".split()
word_to_correct = "weather"
print("To fixing: ", test_sentence,word_to_correct)


choice = longSentence(gramlist, test_sentence, 1)
genePoint(choice,word_to_correct)


test_sentence = "the latest study published this".split()
word_to_correct = "weak"
print("To fixing: ", test_sentence,word_to_correct)


choice = longSentence(gramlist, test_sentence, 1)
genePoint(choice,word_to_correct)


test_sentence = "a number of risk factors for breast".split()
word_to_correct = "cancel"

'''
#getNGramSentenceRandom(gramlist[2], word, 10)

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
            if i <= (max_gram - 1):
                current_word = testtxt[:i]
                word_to_correct = i
            else:
                current_word = testtxt[i-max_gram+1:i]
                word_to_correct = i

            print("To fixing: ", current_word,testtxt[word_to_correct])
            choice,choicebkg = longSentenceBiDir(gramlist, testtxt, word_to_correct, 1)
            resu,confi = genePointBidir(choice, choicebkg, testtxt[word_to_correct])
            print("Correct word: ", testtxtr[i])
            if testtxtr[i] == resu:
                print("CORRECT",confi)
            else:
                print("NOT CORRECT") 
            if confi > 2:
                print("fix-->",testtxt[i] ," to ",resu)
                testtxt[i] = resu
    testtxt = testf.readline()
    testtxtr = testr.readline()



'''
for testword in testtxt:
    current_word = current_word + " " + testword
    current_word_split = current_word.split()
    if len(current_word_split) >= 4:
        word_to_correct = current_word_split[-1]
        test_sentence = current_word_split[:-1]
        print("To fixing: ", test_sentence,word_to_correct)
        choice = longSentence(gramlist, test_sentence, 1)
        genePoint(choice,word_to_correct)
    if len(current_word_split) >= max_gram - 1:
        current_word = current_word.replace(current_word_split[0],'')

'''



#testcase:





choice = longSentence(gramlist, test_sentence, 1)
genePoint(choice,word_to_correct)

print("To fixing: " ,test_sentence,word_to_correct)