import pdb
data_file = open("outputData_T2.txt", "r")

final_file = open("outputDataT2_final.txt", "w")
for line in data_file:
	words = line.strip().split(' ')
	string_to_write = ''
	for word in words:
		string_to_write += word + ', '
	final_file.write(string_to_write[:-2]+'\n')
	# pdb.set_trace()

final_file.close()