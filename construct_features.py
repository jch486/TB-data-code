import pandas as pd
import os
import numpy as np
import math
from gensim.models.doc2vec import Doc2Vec

# create diagnoses grouped by same patient id and date from all_icd_10.csv
def group_patient_date(all_icd_10_df):
    return all_icd_10_df.groupby(['patient_id', 'date'])['dx'].agg(', '.join).reset_index()

# create grouped vectors from grouped diagnoses
def convert_to_vector(pat2vec_model, dx_grouped):
    vectors_grouped = pd.DataFrame()

    for row in dx_grouped.itertuples(index=False):
        patient_id = row[0]
        date = row[1]
        dx = row[2]
        new_vect = pat2vec_model.infer_vector(dx.split(", "))
        new_row = pd.DataFrame({'patient_id': [patient_id], 'date': [date], 'vs': [new_vect]})
        vectors_grouped = pd.concat([vectors_grouped, new_row], ignore_index=True)
    
    return vectors_grouped

def create_feature_vectors(vectors_grouped, timespan, gamma):
    features_vectors = pd.DataFrame()
    # get first and last dates in dataset
    date_start = vectors_grouped['date'].to_numpy().min()
    date_end = vectors_grouped['date'].to_numpy().max()

    # for each patient and visit date combo
    for curr_id_date in vectors_grouped[['patient_id', 'date']].drop_duplicates().itertuples(index=False):
        # month_end = visit date
        month_end = curr_id_date[1]
        month_start = month_end - timespan
        curr_patient = curr_id_date[0]
        # if the timespan before the current visit date is within the dataset's range
        if(month_start >= date_start and month_end <= date_end):
            # get all vectors in the timespan before the current visit date, for the specified patient
            curr_vs = vectors_grouped.loc[
                (vectors_grouped['date'] >= month_start) &
                (vectors_grouped['date'] < month_end) &
                (vectors_grouped['patient_id'] == curr_patient)
            ].copy()

            # calculate discount and discount all vectors
            time_differences = month_end - curr_vs['date']
            discounts = gamma ** time_differences
            curr_vs['vs'] = discounts * curr_vs['vs']

            # add discounted vectors together
            total_vs = pd.DataFrame({'patient_id': [curr_patient], 'date': [month_end], 
                                     'vs': [np.sum(curr_vs['vs'].to_numpy(), axis=0)]})

            # add to dataframe
            features_vectors = pd.concat([features_vectors, total_vs], ignore_index=True)
    
    return features_vectors

def main():
    # set up file paths
    all_icd_10_fn = os.path.join('other_data', 'all_icd_10.csv')
    dx_grouped_fn = os.path.join('other_data', 'dx_grouped.csv')
    vectors_grouped_fn = os.path.join('other_data', 'vectors_grouped.pkl')
    features_vectors_fn = os.path.join('other_data', 'features_vectors.csv')
    
    # load data from csv file
    all_icd_10_df = pd.read_csv(all_icd_10_fn)

    ##### PART 1: group diagnoses by same patient id and date #####
    if not os.path.exists(dx_grouped_fn):
        # create diagnoses grouped by same patient id and date from all_icd_10.csv
        dx_grouped = group_patient_date(all_icd_10_df)
        dx_grouped.to_csv(dx_grouped_fn, index=False)
    ##### PART 1 #####

    ##### PART 2: convert grouped diagnoses to vectors #####
    if not os.path.exists(vectors_grouped_fn):
        # load Pat2Vec Model (which will convert list of diagnoses into 10d vector)
        # https://ai.jmir.org/2023/1/e40755
        # https://huggingface.co/zidatasciencelab/Pat2Vec
        pat2vec_model = Doc2Vec.load('pat2vec_dim10.model')
        # create grouped vectors from grouped diagnoses
        dx_grouped = pd.read_csv(dx_grouped_fn)
        vectors_grouped = convert_to_vector(pat2vec_model, dx_grouped)
        # save with pickle
        vectors_grouped.to_pickle(vectors_grouped_fn)
        # not using csv since it converts vectors to strings
        # vectors_grouped.to_csv(vectors_grouped_fn, index=False)
    ##### PART 2 #####

    ##### PART 3: combine vectors in the thirty days before each visit #####
    if not os.path.exists(features_vectors_fn):
        vectors_grouped = pd.read_pickle(vectors_grouped_fn)
        # not using csv since it converts vectors to strings
        # vectors_grouped = pd.read_csv(vectors_grouped_fn)
        features_vectors = create_feature_vectors(vectors_grouped, 30, 0.9)
        features_vectors.to_csv(features_vectors_fn, index=False)
    ##### PART 3 #####

    # to do:
    # need to remove feature vectors with 0 in the 'vs' column
    # those feature vectors represent patients with 30 days of no diagnoses before a visit

if __name__ == '__main__':
    main()