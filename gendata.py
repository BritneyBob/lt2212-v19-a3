import sys
import argparse
import numpy as np
import pandas as pd
from collections import Counter

def open_file(filename):
    """Opens filename line by line with tokens and their part of speech tags. Returns
       list with lines containing tokens stripped from POS-tags.
    """
    with open(filename,'r') as f:
        lines = f.read().splitlines()
   
    lines_no_pos_tags = []
    for line in lines:
        line = line.split()
        line_no_pos_tags = []
        for token in line:
            index = token.index('/')
            token = token[:index]
            line_no_pos_tags.append(token)
        lines_no_pos_tags.append(line_no_pos_tags)
            
    return lines_no_pos_tags  

def build_one_hot_vectors(lines,start,end,n):
    """Counts the occurences of each oken in lines and builds full vocabulary. Adds 
       start token to full_vocabulary. Makes a list of tokens in lines from start to 
       end. Adds n-1 start tokens to the beginning of the token list. Builds one hot
       vectors of all tokens in vocabulary, with the same length as full vocabulary.
       Makes a list of only those one hot vectors that appear in lines from start to 
       end, in the order they appear. Returns list of tokens (from start to end) and 
       list of hot vectors.
    """   
    every_token = []
    for line in lines:
        every_token += line
        
    full_vocabulary = Counter(every_token).most_common()
    full_vocabulary.append(("START",0))
    
    tokens = []     
    for line in lines[start:end]:
        tokens += line
        
    counter = 0
    while counter < n-1:
        tokens.insert(0,"START")
        counter += 1
    
    one_hot_vectors = {}
    for i,token_count in enumerate(full_vocabulary):
        token = token_count[0]
        vector = np.zeros((len(full_vocabulary),), dtype=int)
        vector[i] = 1
        one_hot_vectors[token] = vector
    
    one_hot_document = []
    for token in tokens:
        one_hot_document.append(one_hot_vectors[token])
    
    return tokens,one_hot_document

def build_ngrams(one_hot_document,tokens,n):
    """Builds n-grams of length n. n-1 one hot vectors in one_hot_document are 
       concatenated to form one vector. The last word (word number n) in the ngram is 
       taken from the words list and becomes the label. Returns a dataframe with each 
       row containing one vector built from n-1 one hot vectors, and one label. 
    """

    ngram_vectors = []
    ngram_labels = []
    
    if n > 2:
        for i,vector in enumerate(one_hot_document[:-(n-1)]):
            ngram_vector = np.concatenate([one_hot_document[i],one_hot_document[i+1]])
            j = 2
            while j < (n-1):
                ngram_vector = np.concatenate([ngram_vector,one_hot_document[i+j]])
                j +=1
            ngram_vectors.append(ngram_vector)
    else:
        for i,vector in enumerate(one_hot_document[:-(n-1)]):
            ngram_vector = one_hot_document[i]
            ngram_vectors.append(ngram_vector)

    ngram_labels = zip(*[tokens[i+(n-1):] for i in range(n-(n-1))])
    ngram_labels = [''.join(label) for label in ngram_labels]

    ngram_tokenvectors = list(zip(ngram_vectors,ngram_labels))
    ngram_vectors_df = pd.DataFrame(ngram_tokenvectors)
    ngram_vectors_df.columns = ["vector","label"]

    return ngram_vectors_df

def write_outputfile(ngram_tokenvectors, outputfile):
    """Writes ngram_wordvectors to outputfile.
    """
    # Disables summary printing, to be able to do calculations on the whole matrix in 
    # outputfile.
    #np.set_printoptions(threshold=np.nan) # Doesn't seem to work while this is 
    # applied... I'm not sure if the calculations are done on the whole matrix.
    ngram_tokenvectors.to_csv(outputfile, index=False)
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Convert text to features")
    parser.add_argument("-N", "--ngram", metavar="N", dest="ngram", type=int, 
    default=3, help="The length of ngram to be considered (default 3).")
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
        tokens,one_hot_document = build_one_hot_vectors(lines,args.startline,args.endline,args.ngram)
        print("Ending at line {}.".format(args.endline))
    else:
        tokens,one_hot_document = build_one_hot_vectors(lines,args.startline,len(lines))
        print("Ending at last line of file.")

    if args.ngram < 2:
        print("Error: Size of n-gram can't be less than 2.")
        sys.exit()
              
    print("Constructing {}-gram model.".format(args.ngram))
    ngram_vectors = build_ngrams(one_hot_document,tokens,args.ngram)
    print("Writing table to {}.".format(args.outputfile))
    write_outputfile(ngram_vectors,args.outputfile)
