file_list = ['Cancer','Breast surgeon','Anastrozole','Femara','HER2']
file_location = './raw_data/'

file_out = open('./raw_data/cancer_combined_train.txt','a')

for file in file_list:
	file_pointer = file_location + file + "_train.txt"
	file_loader = open(file_pointer,'r')
	line = file_loader.readline()
	while line != '':
		file_out.write(line)
		line = file_loader.readline()




