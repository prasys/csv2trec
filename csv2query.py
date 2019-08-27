#!/usr/bin/python3

import pandas as pd
import sys
import numpy as numpy
import math
from gensim import corpora
from nltk.corpus import stopwords
from string import ascii_lowercase
import gensim, os, re, itertools, nltk, snowballstemmer



db_name = sys.argv[1] #get the file name for it
size = 200000 # Chunk Size to handle large file sizes 
chunk_list = []  # append each chunk df here 


# A sample of Stemming , obviously more words can be added or removed. This is just a very naiive way
def stemit():
	stemmer = snowballstemmer.EnglishStemmer()
	stop = stopwords.words('english')
	stop.extend(['may','also','zero','one','two','three','four','five','six','seven','eight','nine','ten','across','among','beside','however','yet','within']+list(ascii_lowercase))
	stoplist = stemmer.stemWords(stop)
	stoplist = set(stoplist)
	stop = set(sorted(stop + list(stoplist))) 
	return stop

# Read's our CSV File Here. This is based on Amazon's Review Dataset that was released. Please visit https://s3.amazonaws.com/amazon-reviews-pds/readme.html for more details 
def read_csv(filepath):
	#parseDate = ['review_date']
	#dateparse = lambda x: pd.datetime.strptime(x, '%Y-%m-%d')
	#colName = ['customer_id','product_category', 'review_id', 'star_rating','helpful_votes','total_votes','vine','verified_purchase','review_body','review_date']
	#colName = ['customer_id', 'review_id', 'product_title' , 'review_body']
	colName = ['review_id', 'review_body']
	column_dtypes = {#'marketplace': 'category',
                # 'customer_id': 'uint32',
                 'review_id': 'str',
                 #'product_id': 'str',
                 #'product_parent': 'uint32',
                # 'product_title' : 'str',
                 #'product_category' : 'category',
                 #'star_rating' : 'Int64',
                 #'helpful_votes' : 'Int64',
                 #'total_votes' : 'Int64',
                 #'vine' : 'category',
                 #'review_date' : 'str',
                 #'verified_purchase' : 'category',
                 #'review_headline' : 'str',
                 'review_body' : 'str'}
	#df_chunk = pd.read_csv(filepath, sep='\t', header=0, chunksize=500000, error_bad_lines=False,parse_dates=parseDate, dtype=column_dtypes, usecols=colName, date_parser=dateparse)
	df_chunk = pd.read_csv(filepath, sep='\t', header=0, chunksize=size, error_bad_lines=False, dtype=column_dtypes, usecols=colName)
	#df_chuck = df_chuck.fillna(0)
	return df_chunk

# Get the Category which we use to cluster from the file , if nothing is given then it just assumes that
def getCategory(input):
	re1='.*?'	# Non-greedy match on filler
	re2='(?:[a-z][a-z]+)'	# Uninteresting: word
	re3='.*?'	# Non-greedy match on filler
	re4='(?:[a-z][a-z]+)'	# Uninteresting: word
	re5='.*?'	# Non-greedy match on filler
	re6='(?:[a-z][a-z]+)'	# Uninteresting: word
	re7='.*?'	# Non-greedy match on filler
	re8='((?:[a-z][a-z]+))'	# Word 1
	rg = re.compile(re1+re2+re3+re4+re5+re6+re7+re8,re.IGNORECASE|re.DOTALL)
	m = rg.search(input)
	if m:
		word1=m.group(1)
		return word1
	return ("NOCAT")	

# The No of Columns that we want to define . It depends on how is your dataset

## A file format to convert to XML 
## Solution Taken From https://stackoverflow.com/questions/18574108/how-do-convert-a-pandas-dataframe-to-xml?rq=1



def removeWhiteSpace(text):
	return (re.sub(r'\s+', ' ', text).strip())



def to_txt(df, filename=None, mode='w'):

	if filename is None:
		return 0

	tfile = open(filename, mode)
	tfile.write(df.to_string(header=False,index=False,line_width=300))
	tfile.close()

pd.DataFrame.to_txt = to_txt # Converts to TXT

isRun = False 
df_chunk = read_csv(db_name)
for index, chunk in enumerate(df_chunk):
	print("Currently Processing " + str(index) + " out of ")


	if (isRun == False):
		cat_text = getCategory(db_name)
		print("Found Category Of " + cat_text)
		isRun = True


	#stop = stemit()
	print("Performing Minor Clean-Up of Documentation")
	chunk['review_body'].replace('[!"#%\'()*+,-./:;<=>?@\[\]^_`{|}~1234567890’”“′‘\\\]',' ',inplace=True,regex=True) # removes invalid character
	chunk['review_body'] = chunk['review_body'].str.lower()
	chunk['review_body'] = chunk['review_body'].astype(str) # Fix In program 
	chunk['review_body'] = chunk['review_body'].apply(removeWhiteSpace)
	print("Transforming it into TREC-XML Format")
	FileName = str(index) + "_" + cat_text + ".query"
	chunk.to_txt(FileName)
