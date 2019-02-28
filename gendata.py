import os, sys
import glob
import argparse
import numpy as np
import pandas as pd
from collections import Counter

# gendata.py -- Don't forget to put a reasonable amount code comments
# in so that we better understand what you're doing when we grade!

# add whatever additional imports you may need here. You may not use the
# scikit-learn OneHotEncoder, or any related automatic one-hot encoders.

def open_file(filename):
    """Opens filename line by line with strings of words with part of speech tags. Returns
       list with lines.
    """
    with open(filename,'r') as f:
        lines = f.read().splitlines()
    
    return lines
        
def remove_pos(lines,start,end):
    """Returns list with words stripped from POS-tags.
    """
    """    
    words = []     
    for line in lines[start:end]:
        line = line.split()
        #print(line)
        #words = words + line
        words.append(line)
    #print(words)"""

    lines_no_pos_tags = []
    for line in lines:
        line = line.split()
        line_no_pos_tags = []
        for word in line:
            index = word.index('/')
            word = word[:index]
            line_no_pos_tags.append(word)
        lines_no_pos_tags.append(line_no_pos_tags)
            
    print(lines_no_pos_tags)
        #words_no_pos_tags.append(word)
        
    #return words_no_pos_tags

def build_one_hot_vectors(word_list):
    """Counts the occurences of each word in word_list and builds vocabulary. Makes one
       hot vectors of all words in vocabulary. Makes a list of one hot vectors, in the
       order they appear in the document. Returns vocabulary and the list of hot vectors.
    """
    vocabulary = Counter(word_list).most_common()
    one_hot_vectors = {}
    
    for i,word in enumerate(vocabulary):
        word = word[0]
        vector = np.zeros((len(vocabulary),), dtype=int)
        vector[i] = 1
        one_hot_vectors[word] = vector
    
    one_hot_document = []
    for word in word_list:
        one_hot_document.append(one_hot_vectors[word])
    
    return vocabulary,one_hot_document 

def build_ngrams(one_hot_document,word_list,n):
    ngram_vectors = []
    ngram_labels = []
    #one_hot_document = one_hot_document[:10]
    #word_list = word_list[:10]
    
    for i,vector in enumerate(one_hot_document[:-(n-1)]):
        ngram_vector = np.concatenate([one_hot_document[i],one_hot_document[i+1]])
        ngram_vectors.append(ngram_vector)
        
    ngram_labels = zip(*[word_list[i+2:] for i in range(n-2)])
    ngram_labels = [''.join(label) for label in ngram_labels]

    ngram_wordvectors = list(zip(ngram_labels,ngram_vectors))
    ngram_vectors_df = pd.DataFrame(ngram_wordvectors)
    ngram_vectors_df.columns = ["ngram","vector"]

    return ngram_vectors_df

def write_outputfile(ngram_wordvectors, outputfile):
    """Writes vectors_df to outputfile.
    """
    # Disables summary printing, to be able to do calculations on the whole matrix in 
    # outputfile.
    np.set_printoptions(threshold=np.nan) 
    ngram_wordvectors.to_csv(outputfile, index=False)
    
parser = argparse.ArgumentParser(description="Convert text to features")
parser.add_argument("-N", "--ngram", metavar="N", dest="ngram", type=int, default=3, help="The length of ngram to be considered (default 3).")
parser.add_argument("-S", "--start", metavar="S", dest="startline", type=int,
                    default=0,
                    help="What line of the input data file to start from. Default is 0, the first line.")
parser.add_argument("-E", "--end", metavar="E", dest="endline",
                    type=int, default=None,
                    help="What line of the input data file to end on. Default is None, whatever the last line is.")
parser.add_argument("inputfile", type=str,
                    help="The file name containing the text data.")
parser.add_argument("outputfile", type=str,
                    help="The name of the output file for the feature table.")

args = parser.parse_args()

print("Loading data from file {}.".format(args.inputfile))
lines = open_file(args.inputfile)
print("Starting from line {}.".format(args.startline))

if args.endline:
    word_list = remove_pos(lines,args.startline,args.endline)
    print("Ending at line {}.".format(args.endline))
else:
    print("Ending at last line of file.")
    word_list = remove_pos(lines,args.startline,len(lines))
    
#vocabulary,one_hot_document = build_one_hot_vectors(word_list)
#ngram_vectors = build_ngrams(one_hot_document,word_list,3)
#write_outputfile(ngram_vectors, args.outputfile)

print("Constructing {}-gram model.".format(args.ngram))
print("Writing table to {}.".format(args.outputfile))
    
# THERE ARE SOME CORNER CASES YOU HAVE TO DEAL WITH GIVEN THE INPUT
# PARAMETERS BY ANALYZING THE POSSIBLE ERROR CONDITIONS.
