import sys
import argparse
import numpy as np
import pandas as pd
import math
from joblib import dump, load
from sklearn.linear_model import LogisticRegression

def load_data(datafile):
    """Reads filename as a dataframe. Removes brackets from all vectors. Converts
       vectors from strings to arrays. Returns the columns (vectors and labels) of 
       vectorfile_df. 
    """
    vectorfile_df = pd.read_csv(datafile)
    vectors = [doc[1:-1] for doc in vectorfile_df['vector']]
    vectors = [np.fromstring(doc,sep=' ') for doc in vectors]
    vectorfile_df['vector'] = vectors
    
    vectors = np.stack(vectorfile_df['vector'])
    labels = np.stack(vectorfile_df['label'])
    
    return vectors,labels

def load_model(modelfile):
    """Loads and returns model from modelfile.
    """
    model = load(modelfile)
    
    return model

def accuracy(vectors,actual_labels,model):
    """Predicts labels for vectors by model. Compares predicted_labels with
       actual_labels. Calculates and returns (rounded) accuracy.
    """
    predicted_labels = model.predict(vectors)

    similar_labels = [i for i, j in zip(predicted_labels,actual_labels) if i == j]
    
    accuracy = round((len(similar_labels) / len(predicted_labels)) * 100,2)

    return accuracy

def perplexity(vectors,actual_labels,model):
    """Gets the probabilities for each class in model. Finds the probability of the
       actual class (label) of the n-gram, and calculates entropy and perplexity by
       this value. Returns perplexity.
    """

    probabilities = model.predict_proba(vectors) 
    classes = model.classes_.tolist()
    
    probabilities_actual_labels = []
    for i,label in enumerate(actual_labels):
        if label in classes:
            index = classes.index(label)
            probability_actual_label = probabilities[i][index]
        else:
            probability_actual_label = 0.0000000001
        probabilities_actual_labels.append(probability_actual_label)
                       
    total = 0
    for probability in probabilities_actual_labels:
        part = math.log2(probability)
        total += part
    total = -total
    entropy = total / len(actual_labels)
    perplexity = math.pow(2,entropy)
    
    return perplexity
                  
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Test a maximum entropy model.")
    parser.add_argument("-N", "--ngram", metavar="N", dest="ngram", type=int,
                        default=3, 
                        help="The length of ngram to be considered (default 3).")
    parser.add_argument("datafile", type=str,
                    help="The file name containing the features in the test data.")
    parser.add_argument("modelfile", type=str,
                    help="The name of the saved model file.")

    args = parser.parse_args()

    print("Loading data from file {}.".format(args.datafile))
    vectors,labels = load_data(args.datafile)
    print("Loading model from file {}.".format(args.modelfile))
    model = load_model(args.modelfile)

    print("Testing {}-gram model.".format(args.ngram))
    accuracy = accuracy(vectors,labels,model)
    perplexity = perplexity(vectors,labels,model)

    print("Accuracy is {}.".format(accuracy))
    print("Perplexity is {}.".format(perplexity))