import sys
import argparse
import numpy as np
import pandas as pd
from joblib import dump, load
from sklearn.linear_model import LogisticRegression

def read_as_df(filename):
    """Reads filename as a dataframe. Removes brackets from all vectors. Converts
       vectors from strings to arrays. Returns the columns (vectors and labels) of 
       vectorfile_df. 
    """
    vectorfile_df = pd.read_csv(filename)
    vectors = [doc[1:-1] for doc in vectorfile_df['vector']]
    vectors = [np.fromstring(doc,sep=' ') for doc in vectors]
    vectorfile_df['vector'] = vectors
    
    return vectorfile_df['vector'],vectorfile_df['label']

def train_model(vectors,labels):
    """Trains a LogisticRegression model on vectors and labels. Returns model. 
    """
    trainer = LogisticRegression(solver="lbfgs",multi_class="multinomial")
    vector_block = np.stack(vectors)
    label_block = np.stack(labels)
    model = trainer.fit(vector_block,label_block)
    
    return model

def write_modelfile(model,outputfile):
    """Writes model to outputfile.
    """
    dump(model,outputfile)
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train a maximum entropy model.")
    parser.add_argument("-N", "--ngram", metavar="N", dest="ngram", type=int,
                        default=3, 
                        help="The length of ngram to be considered (default 3).")
    parser.add_argument("datafile", type=str,
                        help="The file name containing the features.")
    parser.add_argument("modelfile", type=str,
                    help="The name of the file to which you write the trained model.")

    args = parser.parse_args()

    print("Loading data from file {}.".format(args.datafile))
    vectors,labels = read_as_df(args.datafile)
    print("Training {}-gram model.".format(args.ngram))
    model = train_model(vectors,labels)
    print("Writing table to {}.".format(args.modelfile))
    write_modelfile(model,args.modelfile)