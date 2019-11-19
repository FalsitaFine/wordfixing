import random
testf = open("general_out.txt",'r')
test_text = testf.readline()

random_fac_word = 0.2
random_fac_chara = 0.3
random_max_chara = 3

mutate_list = [['a','e','i','o','u','y','h','w'],['b','f','p','v'],['d','t'],['l','m','n','r'],['c','g','j','k','q','s','x','z']]
mutate_test = ""
ref_test = ""
index = 0
while(test_text):

	if(random.random() > 0.99):
		index += 1
		test_words = test_text.split()

		for word in test_words:
			fac = random.random()
			if fac <= random_fac_word:
				mutate_word = ""
				mutate_count = 0
				for chara in word:
					fac = random.random()
					if mutate_count > random_max_chara:
						continue
					if fac <= random_fac_chara and chara.isalpha():
						fac = random.random()
						if fac <= 0.3:
							#duplicate
							mutate_word = mutate_word + chara + chara
							mutate_count += 1
						elif fac <= 0.6:
							#lost
							mutate_word = mutate_word
							mutate_count += 1

						else:
							#mutate character
							for pru_list in mutate_list:
								if chara in pru_list:
									random_chara = random.choice(pru_list)
									#print(pru_list,random_chara)
									mutate_word = mutate_word + random_chara
									mutate_count += 1

					else:
						mutate_word = mutate_word + chara
			else:
				mutate_word = word
			mutate_test = mutate_test + mutate_word + " "
		mutate_test = mutate_test + "\n"
		ref_test = ref_test + test_text
	test_text = testf.readline()
	print("Generating: ",index)


#print(mutate_test)
#print(test_text)

outf = open("mutated.txt",'a')
outfr = open("mutated_ref.txt",'a')
outf.write(mutate_test)
outfr.write(ref_test)

