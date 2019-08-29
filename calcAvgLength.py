#!/usr/bin/python3

from pymongo import MongoClient
from textblob import TextBlob
import pandas as pd
import sys
import textstat
import numpy as numpy
import math
import pickle
import gensim
from gensim import corpora
from pprint import pprint
from nltk.corpus import stopwords
from string import ascii_lowercase
import gensim, os, re, pymongo, itertools, nltk, snowballstemmer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from pandarallel import pandarallel
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.metrics.pairwise import cosine_similarity , linear_kernel
import matplotlib.pylab as plt
import seaborn as sns
import numexpr as ne
import os.path
import scipy.sparse as sparse

#from odo import odo
#import dask.dataframe as pd

#star_rating
#vine
#cat_text = 'verified_purchase'
cat_text = sys.argv[1] #category we want to extract
db_name = cat_text
flag_calc_scores = False
isMongo = False
isCalc = True

size = 500000
chunk_list = []  # append each chunk df here 
analyser = SentimentIntensityAnalyzer()
pandarallel.initialize(progress_bar=True,shm_size_mb=4096,nb_workers=8)
#cat_text = 'vine'
#db_name = "amazon_reviews_us_Books_v1_02.tsv"


def stemit():
	stemmer = snowballstemmer.EnglishStemmer()
	stop = stopwords.words('english')
	stop.extend(['may','also','zero','one','two','three','four','five','six','seven','eight','nine','ten','across','among','beside','however','yet','within']+list(ascii_lowercase))
	stoplist = stemmer.stemWords(stop)
	stoplist = set(stoplist)
	stop = set(sorted(stop + list(stoplist))) 
	return stop

def sentiment_analyzer_scores(sentence):
    score = analyser.polarity_scores(sentence)
    #print(score["compound"])
    return score
    #print("{:-<40} {}".format(sentence, str(score)))


def read_csv(filepath):
	#parseDate = ['review_date']
	#dateparse = lambda x: pd.datetime.strptime(x, '%Y-%m-%d')
	#colName = ['customer_id','product_category', 'review_id', 'star_rating','helpful_votes','total_votes','vine','verified_purchase','review_body','review_date']
	colName = ['customer_id', 'review_id' , 'review_body']
	column_dtypes = {'marketplace': 'category',
                 'customer_id': 'uint32',
                 'review_id': 'str',
                 #'product_id': 'str',
                 #'product_parent': 'uint32',
                 #'product_title' : 'str',
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



def establishMongoDB():
	client = MongoClient('mongodb://localhost:27017/')
	return clientb

def renameColumns(df):
	df.columns = ['a','b','c','d','e','f']
	return df

## A file format to convert to XML 
def to_xml(df, filename=None, mode='w'):
    def row_to_xml(row):
        xml = ['<DOC>']
        for i, col_name in enumerate(row.index):
            xml.append('  <{0}>{1}</{0}>'.format(col_name, row.iloc[i]))
        xml.append('</DOC>')
        return '\n'.join(xml)
    res = '\n'.join(df.apply(row_to_xml, axis=1))

    if filename is None:
        return res
    with open(filename, mode) as f:
        f.write(res)

pd.DataFrame.to_xml = to_xml


def sentiment_calc_polarity(text):
	#score = TextBlob(text).sentiment.polarity
	try:
		#if score > 0.4:
			#print (text)
		#if "delivery" in text:
		#	print ("[DELIVERY DETECTED] " + text)
		#print (TextBlob(text).sentiment.polarity)
		return TextBlob(text).sentiment.polarity
	except:
		return None

def sentiment_calc_subjectivity(text):
	#score = TextBlob(text).sentiment.subjectivity
	try:
		#print (TextBlob(text).sentiment.subjectivity)
		return TextBlob(text).sentiment.subjectivity
	except:
		0
		return None

def fetch_db(text):
	fetch_db = db[text].find()
	return fetch_db

def test(text):
	#print (text)
	score = textstat.automated_readability_index((str (text)))
	if math.isnan(score) == True:
		return 0.0
	else:
		return score

def strip_html(text):
    soup = BeautifulSoup(text, "html.parser")
    return soup.get_text()

def remove_between_square_brackets(text):
    return re.sub('\[[^]]*\]', '', text)

def denoise_text(text):
    text = strip_html(text)
    text = remove_between_square_brackets(text)
    return text

def memory_usage(df):
	return(round(df.memory_usage(deep=True).sum() / 1024 ** 2, 2))

def calculateDocumentSim(document):
	print("Count Vector")
	cv = TfidfVectorizer(stop_words='english',min_df=0.02, max_df=0.85, max_features=100000,use_idf=False,smooth_idf=True)
	#cv = CountVectorizer(stop_words='english',min_df=0.02, max_df=0.85, max_features=100000)
	dt_mat = cv.fit_transform(document)
	print("TFIDF Vector")
	#tfidf = TfidfTransformer(smooth_idf=True,use_idf=False)
	#dt_mat = tfidf.fit_transform(dt_mat)
	#pairwise_similarity = dt_mat * dt_mat.T
	dt_mat = dt_mat.astype("float32") #make things way faster
	print(dt_mat.dtype)
	print(dt_mat.shape)
	print("Calculate the Dot Product")
	#ne.evaluate(dt_mat.T)
	#dt_mat = linear_kernel(dt_mat,dense_output=False)
#	tsvd = TruncatedSVD(n_components=10)
	#X_sparse_tsvd = tsvd.fit(dt_mat).transform(dt_mat)
	#dt_mat = X_sparse_tsvd.dot(X_sparse_tsvd.T)
	#pairwise_similarity = dt_mat
	#pairwise_similarity = dt_mat * dt_mat.T # Multiple the matrix by it's transformation to get the identify matrix to find the similarity 
	print("Return")
	return dt_mat

def func(row):
    xml = ['<doc>']
    for field in row.index:
        xml.append('  <{0}>{1}</{0}>'.format(field, row[field]))
    xml.append('</doc>')
    return '\n'.join(xml)

#sample = denoise_text(sample)
#print(sample)

#Debug code for args 	
#print ("This is the name of the script: " + sys.argv[0])
#print ("Number of arguments: " +  str(sys.argv))
#print ("The arguments are: " + str(sys.argv))



if isMongo == True:
	client = establishMongoDB()
	db = client.amazon
	col = client['amazon_o'][db_name]
	#collection = fetch_db('sample') #STG
	print("Loading data from MongoDB")
	collection = fetch_db(db_name)
	print("Loading data into Pandas")
	df = pd.DataFrame(list(collection))
	#print(df.head())
else:
	#print(df.head())
	df_chunk = read_csv(db_name) # read our CSV file location - needs to be absolute file path. To be tested it out
	for index, chunk in enumerate(df_chunk):
		print("CONVERTING THEM")
		print(chunk.head())
		print(index)
		#chunk['star_rating'] = 
		#hunk['review_date'] = pd.to_datetime(chunk['review_date'])
	#	chunk['year'] = chunk['review_date'].dt.year
		stop = stemit()
		#print(chunk.dtypes)
		print("REMOVE INVALID CHARACTER")
		chunk['review_body'].replace('[!"#%\'()*+,-./:;<=>?@\[\]^_`{|}~1234567890’”“′‘\\\]',' ',inplace=True,regex=True) # removes invalid character
		chunk['review_body'] = chunk['review_body'].astype(str)
		wordlist = filter(None, " ".join(list(set(list(itertools.chain(*chunk['review_body'].str.split(' ')))))).split(" "))
		chunk['stemmed_text_data'] = [' '.join(filter(None,filter(lambda word: word not in stop, line))) for line in chunk['review_body'].str.lower().str.split(' ')]
		chunk['length_scores'] = chunk['stemmed_text_data'].str.len()
		if isCalc == True:
		#	print("Apply WSJ Formatting for it")
		#	loldongs = str(index) +  ".xml"
		#	chunk.to_xml(loldongs)
		#	print("APPLYING READABILITY SCORE")
		#	chunk['readscore'] = chunk['stemmed_text_data'].parallel_apply(test)
		#	print("APPLYING SENTIMENT SCORE")
			#chunk['sentiment'] = chunk['stemmed_text_data'].parallel_apply(sentiment_analyzer_scores)
			#df_r = chunk['sentiment'].parallel_apply(pd.Series)
			#chunk = pd.concat([chunk,df_r], axis=1).drop('sentiment',axis=1) #drops the sentiment score
			chunk = chunk.drop('review_body', axis=1)
			chunk = chunk.drop('stemmed_text_data', axis=1)
		chunk_list.append(chunk)
		#print(df_r.head())

		#print("APPLYING POLARITY SCORE")
		#chunk['polarity'] = chunk['review_body'].apply(sentiment_calc_polarity)
	
	print("CONCAT")
	df = pd.concat(chunk_list)
	#print("CALCULATING PAIRWISE SIMILARITY")
	#pairwise_similarity = calculateDocumentSim(df['review_body'].astype(str))


	#print("DISPLAY HEATMAP")
	#ax = sns.heatmap(pairwise_similarity.todense())
	#plt.show()

	#sns.plt.show()

	path = os.path.abspath(os.path.dirname(__file__))
	#print(path)
	#name = db_name + ".pickle"
	name1 = "/output/" + db_name + "_scores" +".csv"
	output_file = path + name1
	#print (output_file)
	#df.to_parquet(name,compression='gzip')
	#df.to_pickle(name)
	df.to_csv(output_file)
	#print(df.head())

	#print(df.dtypes())





# concat the list into dataframe 





#vine
#rev



#print("Calculating Readability Score")
#df['readscore'] = df['review_body'].apply(test)

#df['review_date'] = pd.to_datetime(df['review_date']) #convert time
#df['year'] = df['review_date'].year
#grouped = df.groupby(cat_text) #group by star rating for amazon here
#group1 = df.groupby(cat_text)['review_body'].apply(lambda x: x.str.split().str.len().mean()) std
#group1 = df.groupby(cat_text)['review_body'].apply(lambda x: x.str.split().str.len().mean())
#print(group1)
#group1 = df.groupby(cat_text)['review_body'].apply(lambda x: x.str.split().str.len().std())
#print(group1)
#group1 = df.groupby(cat_text)['review_body'].apply(lambda x: x.str.split().str.len().sum()) # Calculate the Total Number of Words
#print(group1)
#print ("Calculating Polarity")
#df['polarity'] = df['review_body'].apply(sentiment_calc_polarity)

#print ("Calculating Subjectivity")
#df['subjectivity'] = df['review_body'].apply(sentiment_calc_subjectivity)


#data = df.to_dict(orient='records')

#col.insert_many(data)

#odo(df,db[colle])


if flag_calc_scores == True:
	for group_name, df_group in grouped:
		print(cat_text + " " + str(group_name))
		#print("Total No of Reviews" + str(df_group['review_body'].count()))
		#df_group['test'] = df_group['review_body'].apply(sentiment_calc_polarity)
		#peek = df_group['review_body'].str.extract()
		#print(peek)
		avg_pol = df_group['polarity'].mean()
		stv_pol = df_group['polarity'].std()
		avg_sub = df_group['subjectivity'].mean()
		stv_sub = df_group['subjectivity'].std()
		print(avg_pol)
		print(stv_pol)
		print(avg_sub)
		print(stv_sub)
		#for single in temp:
	#	a = single['review_body']
	#	print(a)

#for entity in collection:
#	  sentiment_calc_polarity(entity['review_body'])
		#print('{0} {1}'.format(car['review_headline'], 
		#	car['review_body'])) 
