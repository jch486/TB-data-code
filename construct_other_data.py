import pandas as pd
import os
import numpy as np
import math

# convert all icd-9 diagnoses in all_dx_visits_df into icd-10
def icd9_to_icd10(all_dx_visits_df, icd9_to_icd10_df):
    all_dx_visits_df = all_dx_visits_df.drop_duplicates()
    changed_df = pd.DataFrame()
    # split all_dx_visits_df into icd-9 and icd-10 diagnoses
    icd9_half = all_dx_visits_df[all_dx_visits_df['dx_ver'] == 9]
    icd10_half = all_dx_visits_df[all_dx_visits_df['dx_ver'] == 10]

    # for each icd-9 diagnosis, convert into icd-10 and add it to changed_df
    for row in icd9_half.itertuples(index=False):
        patient_id = row[0]
        inpatient = row[3]
        date = row[4]
        conversions = icd9_to_icd10_lookup(icd9_to_icd10_df, row[1])
        for new_dx in conversions:
            new_row = pd.DataFrame({'patient_id': [patient_id], 'dx': [new_dx], 'dx_ver': [10], 'inpatient': [inpatient], 'date': [date]})
            changed_df = pd.concat([changed_df, new_row], ignore_index=True)

    # combine newly mapped icd-10 diagnoses and original icd-10 diagnoses
    return pd.concat([changed_df, icd10_half], ignore_index=True)

# return all icd-10 codes that correspond to specified icd-9 code
def icd9_to_icd10_lookup(icd9_to_icd10_df, dx):
    return icd9_to_icd10_df.loc[(icd9_to_icd10_df['icd9cm'] == dx), 'icd10cm']

# create diagnoses grouped by same patient id and date from all_icd_10.csv
def group_patient_date(all_icd_10_df):
    return all_icd_10_df.groupby(['patient_id', 'date'])['dx'].agg(', '.join).reset_index()

# create grouped vectors from grouped diagnoses
def convert_to_vector(icd_10_embedding, dx_grouped):
    vectors_grouped = pd.DataFrame()

    for row in dx_grouped.itertuples(index=False):
        patient_id = row[0]
        date = row[1]
        dx = row[2]
        dx_conversions = icd_10_embedding[icd_10_embedding['code'].isin(dx.split(", "))]

        if(not dx_conversions.empty):
            new_row = pd.DataFrame({'patient_id': [patient_id], 'date': [date]})
            new_row['f1'] = [np.mean(dx_conversions['V1'].to_numpy(), axis=0)]
            new_row['f2'] = [np.mean(dx_conversions['V2'].to_numpy(), axis=0)]
            new_row['f3'] = [np.mean(dx_conversions['V3'].to_numpy(), axis=0)]
            new_row['f4'] = [np.mean(dx_conversions['V4'].to_numpy(), axis=0)]
            new_row['f5'] = [np.mean(dx_conversions['V5'].to_numpy(), axis=0)]
            new_row['f6'] = [np.mean(dx_conversions['V6'].to_numpy(), axis=0)]
            new_row['f7'] = [np.mean(dx_conversions['V7'].to_numpy(), axis=0)]
            new_row['f8'] = [np.mean(dx_conversions['V8'].to_numpy(), axis=0)]
            new_row['f9'] = [np.mean(dx_conversions['V9'].to_numpy(), axis=0)]
            new_row['f10'] = [np.mean(dx_conversions['V10'].to_numpy(), axis=0)]
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
                                        'f1': [np.mean(curr_vs['f1'].to_numpy(), axis=0)], 
                                        'f2': [np.mean(curr_vs['f2'].to_numpy(), axis=0)], 
                                        'f3': [np.mean(curr_vs['f3'].to_numpy(), axis=0)], 
                                        'f4': [np.mean(curr_vs['f4'].to_numpy(), axis=0)], 
                                        'f5': [np.mean(curr_vs['f5'].to_numpy(), axis=0)], 
                                        'f6': [np.mean(curr_vs['f6'].to_numpy(), axis=0)], 
                                        'f7': [np.mean(curr_vs['f7'].to_numpy(), axis=0)], 
                                        'f8': [np.mean(curr_vs['f8'].to_numpy(), axis=0)], 
                                        'f9': [np.mean(curr_vs['f9'].to_numpy(), axis=0)], 
                                        'f10': [np.mean(curr_vs['f10'].to_numpy(), axis=0)], })

                # add to dataframe
                frames = [df for df in [features, total_vs] if not df.empty]
                features = pd.concat(frames, ignore_index=True)
    
    return features

def construct_outcomes(features, tb_dx_visits_df):
    outcomes = pd.DataFrame()

    # for each patient (each 30-day window before a visit is a patient)
    for features in features.itertuples(index=False):
        patient_id = features[0]
        date = features[1]
        # find whether or not the patient was diagnosed with TB on their visit
        has_tb = ((tb_dx_visits_df['patient_id'] == patient_id) & (tb_dx_visits_df['date'] == date)).any()
        # add outcomes for that visit to dataframe
        new_row = pd.DataFrame({'example_id': [[patient_id, date]], 'has_tb': [int(has_tb)], 'has_no_tb': [int(not has_tb)]})
        outcomes = pd.concat([outcomes, new_row], ignore_index=True)
    
    return outcomes

def construct_metadata(outcomes):
    metadata = outcomes[['example_id']].copy()
    metadata['person_id'] = metadata['example_id']
    return metadata

def main():
    # set up file paths and load data
    icd9_to_icd10_fn = os.path.join('converters', 'icd9toicd10cmgem.csv')
    icd9_to_icd10_df = pd.read_csv(icd9_to_icd10_fn)
    all_dx_visits_fn = os.path.join('subsample', 'all_dx_visits.csv')
    all_dx_visits_df = pd.read_csv(all_dx_visits_fn)
    all_icd_10_fn = os.path.join('other_data', 'all_icd_10.csv')
    dx_grouped_fn = os.path.join('other_data', 'dx_grouped.csv')
    icd_10_embedding_fn = os.path.join('converters', 'icd-10-cm-2022-embedding.csv')
    icd_10_embedding = pd.read_csv(icd_10_embedding_fn)
    vectors_grouped_fn = os.path.join('other_data', 'vectors_grouped.pkl')
    features_fn = os.path.join('other_data', 'features.csv')
    features_formatted_fn = os.path.join('other_data', 'features_formatted.csv')
    tb_dx_visits_fn = os.path.join('subsample', 'tb_dx_visits.csv')
    tb_dx_visits_df = pd.read_csv(tb_dx_visits_fn)
    outcomes_fn = os.path.join('other_data', 'outcomes.csv')
    outcomes_undersampled_fn = os.path.join('other_data', 'outcomes_undersampled.csv')
    metadata_fn = os.path.join('other_data', 'metadata.csv')
    features_formatted_undersampled_fn = os.path.join('other_data', 'features_formatted_undersampled.csv')

    ##### PART 1: convert all icd-9 diagnoses in all_dx_visits_df into icd-10 #####
    if not os.path.exists(all_icd_10_fn):
        # all_icd_10 = icd9_to_icd10(all_dx_visits_df, icd9_to_icd10_df)
        # only using ICD-10 codes
        all_icd_10 = all_dx_visits_df[all_dx_visits_df['dx_ver'] == 10]
        all_icd_10.to_csv(all_icd_10_fn, index=False)
    ##### PART 1 #####

    all_icd_10_df = pd.read_csv(all_icd_10_fn)

    ##### PART 2: group diagnoses by same patient id and date #####
    if not os.path.exists(dx_grouped_fn):
        # create diagnoses grouped by same patient id and date from all_icd_10
        dx_grouped = group_patient_date(all_icd_10_df)
        dx_grouped.to_csv(dx_grouped_fn, index=False)
    ##### PART 2 #####

    dx_grouped = pd.read_csv(dx_grouped_fn)

    ##### PART 3: convert grouped diagnoses to vectors #####
    # using: https://doi.org/10.1186/s12859-023-05597-2
    # https://github.com/kaneplusplus/icd-10-cm-embedding
    if not os.path.exists(vectors_grouped_fn):
        # create grouped vectors from grouped diagnoses
        vectors_grouped = convert_to_vector(icd_10_embedding, dx_grouped)
        # save with pickle (not using csv since it converts vectors to strings)
        vectors_grouped.to_pickle(vectors_grouped_fn)
    ##### PART 3 #####

    vectors_grouped = pd.read_pickle(vectors_grouped_fn)

    ##### PART 4: combine vectors in the thirty days before each visit #####
    if not os.path.exists(features_fn):
        features = create_features(vectors_grouped, 30, 0.95)
        features.to_csv(features_fn, index=False)
    ##### PART 4 #####

    features = pd.read_csv(features_fn)

    ##### PART 5: vector formatting #####
    if not os.path.exists(features_formatted_fn):
        features = pd.read_csv(features_fn)
        features_formatted = features
        features_formatted['example_id']= features_formatted[['patient_id', 'date']].values.tolist()
        features_formatted[['example_id', 'f1', 'f2', 'f3', 'f4', 'f5', 'f6', 'f7', 'f8', 'f9', 'f10']].to_csv(features_formatted_fn, index=False)
    ##### PART 5 #####

    features_formatted = pd.read_csv(features_formatted_fn)

    ##### PART 6: construct outcomes #####
    if not os.path.exists(outcomes_fn):
        outcomes = construct_outcomes(features, tb_dx_visits_df)
        outcomes.to_csv(outcomes_fn, index=False)
    ##### PART 6 #####

    outcomes = pd.read_csv(outcomes_fn)

    '''
    ##### PART 7: balance dataset by removing some patients with no TB from outcomes #####
    if not os.path.exists(outcomes_undersampled_fn):
        outcomes_tb = outcomes[outcomes['has_tb'] == 1]
        outcomes_no_tb = outcomes[outcomes['has_no_tb'] == 1].sample(n=len(outcomes_tb), random_state=42)
        outcomes_undersampled = pd.concat([outcomes_no_tb, outcomes_tb], ignore_index=True)
        outcomes_undersampled.to_csv(outcomes_undersampled_fn, index=False)
    ##### PART 7 #####

    outcomes_undersampled = pd.read_csv(outcomes_undersampled_fn)

    ##### PART 8: balance dataset by removing same patients from features #####
    if not os.path.exists(features_formatted_undersampled_fn):
        features_formatted_undersampled = features_formatted[features_formatted['example_id'].isin(outcomes_undersampled['example_id'])]
        features_formatted_undersampled.to_csv(features_formatted_undersampled_fn, index=False)
    ##### PART 8 #####
    '''

    ##### PART 9: construct metadata #####
    if not os.path.exists(metadata_fn):
        metadata = construct_metadata(outcomes)
        metadata.to_csv(metadata_fn, index=False)
    ##### PART 9 #####

if __name__ == '__main__':
    main()