import pandas as pd
import os
import numpy as np
import math
# from gensim.models.doc2vec import Doc2Vec

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
        new_row = pd.DataFrame({'patient_id': [patient_id], 'date': [date]})
        new_row['f1'] = new_vect[0]
        new_row['f2'] = new_vect[1]
        new_row['f3'] = new_vect[2]
        new_row['f4'] = new_vect[3]
        new_row['f5'] = new_vect[4]
        new_row['f6'] = new_vect[5]
        new_row['f7'] = new_vect[6]
        new_row['f8'] = new_vect[7]
        new_row['f9'] = new_vect[8]
        new_row['f10'] = new_vect[9]
        vectors_grouped = pd.concat([vectors_grouped, new_row], ignore_index=True)
    
    return vectors_grouped

def create_features(vectors_grouped, timespan, gamma):
    features = pd.DataFrame(columns=['patient_id', 'date', 'f1', 'f2', 'f3', 'f4', 'f5', 'f6', 'f7', 'f8', 'f9', 'f10'])
    # get first and last dates in dataset
    date_start = vectors_grouped['date'].to_numpy().min()
    date_end = vectors_grouped['date'].to_numpy().max()

    # for each patient and visit date combo
    for curr_id_date in vectors_grouped[['patient_id', 'date']].drop_duplicates().itertuples(index=False):
        # month_end = visit date
        month_end = curr_id_date[1]
        month_start = month_end - timespan
        curr_patient = curr_id_date[0]
        # overlap = if there is overlap between the 30-day window before this visit and any dates in the final feature vector for this patient
        overlap = np.isin(np.arange(month_end - 30, month_end), features[features['patient_id'] == curr_patient]['date'].values).any()
        # if the timespan before the current visit date is within the dataset's range and there is no overlap
        if(month_start >= date_start and month_end <= date_end and not overlap):
            # get all vectors in the timespan before the current visit date, for the specified patient
            curr_vs = vectors_grouped.loc[
                (vectors_grouped['date'] >= month_start) &
                (vectors_grouped['date'] < month_end) &
                (vectors_grouped['patient_id'] == curr_patient)
            ].copy()

            # proceed with calculating feature vector if there are vectors to aggregate
            # 30 days of no diagnoses before a visit = no vectors to aggregate
            if(len(curr_vs) != 0):
                # calculate discount and discount all vectors
                time_differences = month_end - curr_vs['date']
                discounts = gamma ** time_differences
                curr_vs['f1'] = discounts * curr_vs['f1']
                curr_vs['f2'] = discounts * curr_vs['f2']
                curr_vs['f3'] = discounts * curr_vs['f3']
                curr_vs['f4'] = discounts * curr_vs['f4']
                curr_vs['f5'] = discounts * curr_vs['f5']
                curr_vs['f6'] = discounts * curr_vs['f6']
                curr_vs['f7'] = discounts * curr_vs['f7']
                curr_vs['f8'] = discounts * curr_vs['f8']
                curr_vs['f9'] = discounts * curr_vs['f9']
                curr_vs['f10'] = discounts * curr_vs['f10']

                # add discounted vectors together
                total_vs = pd.DataFrame({'patient_id': [curr_patient], 'date': [month_end], 
                                        'f1': [np.sum(curr_vs['f1'].to_numpy(), axis=0)], 
                                        'f2': [np.sum(curr_vs['f2'].to_numpy(), axis=0)], 
                                        'f3': [np.sum(curr_vs['f3'].to_numpy(), axis=0)], 
                                        'f4': [np.sum(curr_vs['f4'].to_numpy(), axis=0)], 
                                        'f5': [np.sum(curr_vs['f5'].to_numpy(), axis=0)], 
                                        'f6': [np.sum(curr_vs['f6'].to_numpy(), axis=0)], 
                                        'f7': [np.sum(curr_vs['f7'].to_numpy(), axis=0)], 
                                        'f8': [np.sum(curr_vs['f8'].to_numpy(), axis=0)], 
                                        'f9': [np.sum(curr_vs['f9'].to_numpy(), axis=0)], 
                                        'f10': [np.sum(curr_vs['f10'].to_numpy(), axis=0)], })

                # add to dataframe
                frames = [df for df in [features, total_vs] if not df.empty]
                features = pd.concat(frames, ignore_index=True)
    
    return features

def main():
    # set up file paths
    all_icd_10_fn = os.path.join('other_data', 'all_icd_10.csv')
    dx_grouped_fn = os.path.join('other_data', 'dx_grouped.csv')
    vectors_grouped_fn = os.path.join('other_data', 'vectors_grouped.pkl')
    features_fn = os.path.join('other_data', 'features.csv')
    features_formatted_fn = os.path.join('other_data', 'features_formatted.csv')
    
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
        # save with pickle (not using csv since it converts vectors to strings)
        vectors_grouped.to_pickle(vectors_grouped_fn)
    ##### PART 2 #####

    ##### PART 3: combine vectors in the thirty days before each visit #####
    if not os.path.exists(features_fn):
        vectors_grouped = pd.read_pickle(vectors_grouped_fn)
        features = create_features(vectors_grouped, 30, 0.95)
        features.to_csv(features_fn, index=False)
    ##### PART 3 #####

    ##### PART 4: vector formatting #####
    if not os.path.exists(features_formatted_fn):
        features = pd.read_csv(features_fn)
        features_formatted = features
        features_formatted['example_id']= features_formatted[['patient_id', 'date']].values.tolist()
        features_formatted[['example_id', 'f1', 'f2', 'f3', 'f4', 'f5', 'f6', 'f7', 'f8', 'f9', 'f10']].to_csv(features_formatted_fn, index=False)
    ##### PART 4 #####

if __name__ == '__main__':
    main()