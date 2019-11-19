import os
import re
import sys
import random
import csv


#path = sys.argv[1]
path = "info_test/"


file_list = []
vocab = []
vocabr = []


punc = [".",",","?",":","(",")","*","'",'"','\t']
spe = ["/","<",">",'=',"["]
pathDir = os.listdir(path)
for d in pathDir:
	child = os.path.join("%s%s%s%s" % ("./",path,"/",d))
	print ("loading data:", child)
	if d != ".DS_Store":
		file_list.append(child)


#writef = open(sys.argv[2],'a')
writef = open("./general_out.txt",'a')
writefp = open("./general_ref.txt",'a')
#trainwrite = open("./raw_data/train_ref.txt",'a')

writev = open("./vocab",'a')


count = 0
lines = 0
files = 0

for child in file_list:
	#readf = open(child,'r')
	readf = open(child,'r')

	print("converting data:",child,files)
	line = readf.readline()
	while(line):
		line = line.replace("\n","")


		line_ref = line.replace("."," .PERIOD")
		line_ref = line_ref.replace(","," ,COMMA")
		line_ref = line_ref.replace("?"," ?QUESTION")

		line_no_punc = line

		for redu in punc:
			line_no_punc = line_no_punc.replace(redu,"")
		
		line_split = line_no_punc.split(" ")
		line_ref_split = line_ref.split(" ")

		for words in line_split:
			#word_alp = words.replace(".","")
			#word_alp = word_alp.replace(",","")
			#word_alp = word_alp.replace("?","")
			flag = 0
			for flags in spe:
				if flags in words:
					line_no_punc = line_no_punc.replace(words,"")
					flag = 1

			if (not(words in vocab)) and (flag == 0):
				vocab.append(words)
				writev.write(words + "\n")


		for words in line_ref_split:
			for flags in spe:
				if flags in words:
					line_ref = line_ref.replace(words,"")


		while("  " in line_no_punc):
			line_no_punc = line_no_punc.replace("  "," ")
		while("  " in line_ref):
			line_ref = line_ref.replace("  "," ")

		writef.write(line_no_punc + "\n")
		writefp.write(line_ref + "\n")

		#line_reduced_pause = line_reduced.replace(" "," <sil=0.000> ")
		lines += 1
		line = readf.readline()
	files = files + 1

print("vocabulary file generated with totally",len(vocab))
