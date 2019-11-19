import os
import re
import sys
import random
import csv


#path = sys.argv[1]
path = "info/"


file_list = []
vocab = []
vocabr = []


punc = [".",",","?"]
pathDir = os.listdir(path)
for d in pathDir:
	child = os.path.join("%s%s%s%s" % ("./",path,"/",d))
	print ("loading data:", child)
	if d != ".DS_Store":
		file_list.append(child)



#prelist_test_re = ["^Good","^Merry"]
#suflist_test_re = ["ful$"]
#prelist_test = ["Good","Merry"]
#suflist_test = ["ful"]

prelist = []
suflist = []
prelist_re = []
suflist_re = []

#Load patterns
'''
readpre = open("prefix.txt",'r')
line = readpre.readline()
while line != "":
	pre_split = line.split("+, ")
	for words in pre_split:
		prelist.append(words)
		prelist_re.append(("^"+words))
	line = readpre.readline()

readsuf = open("suffix.txt",'r')
line = readsuf.readline()
while line != "":
	suf_split = line.split(", +")
	for words in suf_split:
		suflist.append(words)
		suflist_re.append((words+"$"))
	line = readsuf.readline()

suflist[0] = suflist[0].replace("+","")
suflist_re[0] = suflist_re[0].replace("+","")

#print(prelist)
#print(suflist)

#prelist = prelist_test;
#suflist = suflist_test;
#prelist_re = prelist_test_re;
#suflist_re = suflist_test_re;
'''
#Read file
#readf = open(sys.argv[1],'r')

#writef = open(sys.argv[2],'a')
writef = open("./raw_data/train.txt",'a')
writefp = open("./raw_data/train_pause.txt",'a')
#trainwrite = open("./raw_data/train_ref.txt",'a')

writefd = open("./raw_data/dev.txt",'a')
writefdp = open("./raw_data/dev_pause.txt",'a')
#devwrite = open("./raw_data/dev_ref.txt",'a')


writev = open("./raw_data/vocab",'a')
writevs = open("./raw_data/vocabs",'a')

writer = open("./raw_data/reference.txt",'a')
writer2 = open("./raw_data/nopunc.txt",'a')
count = 0
lines = 0

for child in file_list:
	#readf = open(child,'r')
	childreader = open(child, mode='r')
	csv_reader = csv.DictReader(childreader)

	print("converting data:",child)

	devname = child.split("/")[-1]

	devname = devname.replace(".csv","_trans.txt")

	devname = "./raw_data/" + devname

	trainname = devname.replace("_trans","_train")

	devwrite = open(devname,'a')

	trainwrite = open(trainname,'a')
	#line = readf.readline()
	#Abandon the first line of csv files


	#line = readf.readline()

	#lines += 1

	lines = 1

	for row in csv_reader:

		#print(f'\t{row["tweet"]}')

		line = row["tweet"]
		line = line.split("http://")[0]
		line = line.replace("http://",'')
		#line = line.replace("?",".")

		#accuracy
		#line_acc = line.split(",")
		#print(line_acc[-1])
		#line = line.replace(line_acc[-1],"")

		#if line[-1] == ",":
		#	line = line[:-1]
		#if not(line[-1] in punc):
		#	line = line + "."
		#
		writer.write(line)




		line_no_punc = line.replace(".","")
		line_no_punc = line_no_punc.replace(",","")
		line_no_punc = line_no_punc.replace(" "," <sil=0.000> ")
		writer2.write(line_no_punc)

		line_split = line.split(" ")
		line_reduced = ""
		for words in line_split:
			word_alp = words.replace(".","")
			word_alp = word_alp.replace(",","")
			word_alp = word_alp.replace("?","")

			if not(word_alp in vocab):
				vocab.append(word_alp)
				writevs.write(word_alp + "\n")

			pre_temp = ""
			suf_temp = ""
			pre_index = 0
			suf_index = 0
			for prefix in prelist_re:
				if re.search(prefix, words):
					#print(prefix)
					nPos = len(prelist[pre_index]) + 1
					#print(nPos)
					pre_temp += prelist[pre_index]
					count+=1
					for i in range(nPos,len(words)):
						pre_temp += "A"
					#print(pre_temp)
					break
				pre_index += 1

			for suffix in suflist_re:
				if re.search(suffix, words):
					#print(suffix)	
					nPos = words.find(suflist[suf_index])
					count+=1
					#print(nPos)
					for i in range(nPos):
						suf_temp += "A"
					suf_temp += suflist[suf_index]
					#print(suf_temp)
					break
				suf_index += 1

			if (pre_temp != ""):
				line_reduced += pre_temp + " "
			elif (suf_temp != ""):
				line_reduced += suf_temp + " "
			else:
				if not("https://"  in words or "/" in words):
					line_reduced += words + " "

		writer.write(line_reduced + "\n")
		devwrite.write(line_reduced + "\n")

		line_reduced = line_reduced.replace("\n","")
	
		line_reduced_split = line_reduced.split(" ")

		line_reduced = line_reduced.replace("."," .PERIOD")
		line_reduced = line_reduced.replace(","," ,COMMA")
		line_reduced = line_reduced.replace("?"," ?QUESTION")


		trainwrite.write(line_reduced + "\n")

		#line_reduced_pause = line_reduced.replace(" "," <sil=0.000> ")
		

		for words in line_reduced_split:
			word_alp = words.replace(".","")
			word_alp = word_alp.replace(",","")
			word_alp = word_alp.replace("?","")

			if not(word_alp in vocabr):
				vocabr.append(word_alp)
				writev.write(word_alp + "\n")

		line_reduced += "\n"


		#print(line_reduced)
		random_int = random.randint(1,10)
		if (random_int <= 8):
			writef.write(line_reduced)
			#writefp.write(line_reduced_pause)
		else:
			writefd.write(line_reduced)
			#writefdp.write(line_reduced_pause)

		#line = readf.readline()
		lines += 1

print("convertion done,",count,"words replaced in", lines, "lines")
print("vocabulary file generated with totally",len(vocab),"words, reduced to",len(vocabr),"words")
