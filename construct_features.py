import pandas as pd
import os
import numpy as np
import math
from gensim.models.doc2vec import Doc2Vec

def main():
    # load data from csv files
    all_icd_10_fn = os.path.join('ICD-9-10_converters', 'all_icd_10.csv')
    all_icd_10_df = pd.read_csv(all_icd_10_fn)
    # load Pat2Vec Model (which will convert list of diagnoses into 10d vector)
    pat2vec_model = Doc2Vec.load('pat2vec_dim10.model')
    print(pat2vec_model.infer_vector(["M54.1", "J06.9", "401", "R51"]))


if __name__ == '__main__':
    main()