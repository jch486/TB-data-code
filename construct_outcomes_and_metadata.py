import pandas as pd
import os
import numpy as np
import math

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
    features_fn = os.path.join('other_data', 'features.csv')
    features = pd.read_csv(features_fn)
    tb_dx_visits_fn = os.path.join('subsample', 'tb_dx_visits.csv')
    tb_dx_visits_df = pd.read_csv(tb_dx_visits_fn)
    outcomes_fn = os.path.join('other_data', 'outcomes.csv')
    outcomes_undersampled_fn = os.path.join('other_data', 'outcomes_undersampled.csv')
    metadata_fn = os.path.join('other_data', 'metadata.csv')
    features_formatted_fn = os.path.join('other_data', 'features_formatted.csv')
    features_formatted = pd.read_csv(features_formatted_fn)
    features_formatted_undersampled_fn = os.path.join('other_data', 'features_formatted_undersampled.csv')

    if not os.path.exists(outcomes_fn):
        outcomes = construct_outcomes(features, tb_dx_visits_df)
        outcomes.to_csv(outcomes_fn, index=False)

    outcomes = pd.read_csv(outcomes_fn)

    # balance dataset by removing patients with no TB
    if not os.path.exists(outcomes_undersampled_fn):
        outcomes_tb = outcomes[outcomes['has_tb'] == 1]
        outcomes_no_tb = outcomes[outcomes['has_no_tb'] == 1].sample(frac=len(outcomes_tb)/len(outcomes), random_state=42)
        outcomes_undersampled = pd.concat([outcomes_no_tb, outcomes_tb], ignore_index=True)
        outcomes_undersampled.to_csv(outcomes_undersampled_fn, index=False)

    outcomes_undersampled = pd.read_csv(outcomes_undersampled_fn)

    if not os.path.exists(features_formatted_undersampled_fn):
        features_formatted_undersampled = features_formatted[features_formatted['example_id'].isin(outcomes_undersampled['example_id'])]
        features_formatted_undersampled.to_csv(features_formatted_undersampled_fn, index=False)

    if not os.path.exists(metadata_fn):
        metadata = construct_metadata(outcomes)
        metadata.to_csv(metadata_fn, index=False)

if __name__ == '__main__':
    main()