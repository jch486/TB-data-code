import pandas as pd
import os
import numpy as np
import math

def construct_outcomes(features_vectors, tb_dx_visits_df):
    outcomes = pd.DataFrame()

    # for each patient (each 30-day window before a visit is a patient)
    for features in features_vectors.itertuples(index=False):
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
    features_vectors_fn = os.path.join('other_data', 'features_vectors.csv')
    features_vectors = pd.read_csv(features_vectors_fn)
    tb_dx_visits_fn = os.path.join('subsample', 'tb_dx_visits.csv')
    tb_dx_visits_df = pd.read_csv(tb_dx_visits_fn)
    outcomes_fn = os.path.join('other_data', 'outcomes.csv')
    metadata_fn = os.path.join('other_data', 'metadata.csv')

    if not os.path.exists(outcomes_fn):
        outcomes = construct_outcomes(features_vectors, tb_dx_visits_df)
        outcomes.to_csv(outcomes_fn, index=False)

    outcomes = pd.read_csv(outcomes_fn)

    if not os.path.exists(metadata_fn):
        metadata = construct_metadata(outcomes)
        metadata.to_csv(metadata_fn, index=False)

if __name__ == '__main__':
    main()