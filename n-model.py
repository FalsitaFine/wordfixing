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





def generateNgram(n=1):
    gram = dict()
        
    for i in xrange(len(words)-(n-1)):
        key = tuple(words[i:i+n])
        if gram.has_key(key):
            gram[key] += 1
        else:
            gram[key] = 1

    gram = sorted(gram.items(), key=lambda (_, count): -count)
    return gram

trigram = generateNgram(3)

def getNGramSentenceRandom(gram, word, n = 50):
	for i in xrange(n):
		#print word,
		choices = [element for element in gram if element[0][0] == word]
		#print choices
		print element
		print ">>>>"
		print choices

		if not choices:
		    break
		#word = weighted_choice(choices)[1]


def longSentence(gramlist, words, n = 50):
    gramlen = len(gramlist)
    sentenlen = len(words)

    choice = []
    longest_n = min(sentenlen+1,gramlen)
    while(longest_n > 1):
    #print longest_n, gramlist[longest_n][0]
    #choices = [element for element in gramlist[longest_n-1] if contains(words[sentenlen - longest_n : sentenlen], list(element[0]))]
    #print len(element[0]), len(gramlist[longest_n-2][0][0]),len(words[sentenlen - longest_n : sentenlen])
    #print list(element[0]), gramlist[longest_n-2][0],list(words[sentenlen - longest_n : sentenlen])
        for gram in gramlist[longest_n-2]:
            #print gram
            #print gram[0][:longest_n], words[-longest_n:]
            flag = 1
            for i in range(longest_n-1):
                if gram[0][i] != words[len(words)-longest_n+1+i]:
                    flag = 0
            if flag == 1:
                #print gram[0][:longest_n], words[-longest_n:]
                choice.append(gram)

            #print len(gram[0][:longest_n-1]), len(words[-longest_n+1:])
            #if(gram[0][:longest_n] == words[-longest_n:]):
        if len(choice)>0:
        longest_n -= 1
    print ">>>",choice





gramlist = []
max_gram = 12
for i in range(2, max_gram):
	gramlist.append(generateNgram(i))

#test_sentence = ['possessive', 'behaviors', 'are', 'the', 'ones', 'people']
test_sentence = "A doctor can help determing".split()
word = "cancer"
#getNGramSentenceRandom(gramlist[2], word, 10)

longSentence(gramlist, test_sentence, 1)

