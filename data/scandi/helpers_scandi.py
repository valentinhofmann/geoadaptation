import codecs
import re
import csv
import pickle

def serialize(item, path):
	pickle.dump(item, open(path, "wb" ))

def deserialize(path):
	return pickle.load(open(path, "rb" ))

def load_file(filepath):
	return (codecs.open(filepath, 'r', encoding = 'utf8', errors = 'replace')).read()

def load_lines(filepath):
	return [l.strip() for l in list(codecs.open(filepath, "r", encoding = 'utf8', errors = 'replace').readlines())]

def write_list(path, list, append = False):
	f = codecs.open(path,'w' if not append else 'a',encoding='utf8')
	for l in list:
		f.write(str(l) + "\n")
	f.close()

def get_csv_lines(path, delimiter = ","):
	csved_lines = []
	with codecs.open(path, 'r', encoding = 'utf8', errors = 'replace') as csvfile:
		reader = csv.reader(csvfile, delimiter=',')
		for row in reader:
			csved_lines.append(row)
	return csved_lines

def clean_city(c):
    c = ''.join(i for i in c if i.isalpha() or i == ' ')
    c = c.strip()
    return c

def point2country(point, sv, no, da):
    if point.within(sv):
        return 'Sverige'
    elif point.within(no):
        return 'Norge'
    elif point.within(da):
        return 'Danmark'
    return ''

def is_retweet(text):
    return bool(re.search(r'RT @\w+', text))

def replace_links(text):
    return re.sub(r'(http|www)[\w.,;@?^=%&:\/\"~+#-]+', 'link', text)

def replace_usernames(text):
    return (re.sub(r'@\w+', '', text)).strip()
