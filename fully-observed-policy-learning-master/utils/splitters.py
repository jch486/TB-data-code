import logging
from sklearn.utils import shuffle
import pandas as pd

# split cohort into each outcome, then shuffles unique elements of the cohort 
# and splits it in two based on train_size
def split_pids(cohort_info_df, outcomes_df,
                seed, train_size=.7): 

    cohort_df = cohort_info_df.merge(outcomes_df, on='example_id')
    has_tb = cohort_df[cohort_df['has_tb'] == 1]
    has_no_tb = cohort_df[cohort_df['has_no_tb'] == 1]

    pids_tb = sorted(has_tb['person_id'].unique())
    shuffled_pids_tb = shuffle(pids_tb, random_state=seed)
    cutoff = int(len(shuffled_pids_tb)*train_size)
    train_pids_tb, val_pids_tb = shuffled_pids_tb[:cutoff], shuffled_pids_tb[cutoff:]

    pids_no_tb = sorted(has_no_tb['person_id'].unique())
    shuffled_pids_no_tb = shuffle(pids_no_tb, random_state=seed)
    cutoff = int(len(shuffled_pids_no_tb)*train_size)
    train_pids_no_tb, val_pids_no_tb = shuffled_pids_no_tb[:cutoff], shuffled_pids_no_tb[cutoff:]

    logging.info(f"Train TB proportion: {len(train_pids_tb) / (len(train_pids_tb) + len(train_pids_no_tb))}")
    logging.info(f"Val TB proportion: {len(val_pids_tb) / (len(val_pids_tb) + len(val_pids_no_tb))}")
    
    return train_pids_tb + train_pids_no_tb, val_pids_tb + val_pids_no_tb

    '''
    # unique() returns unique values
    pids = sorted(cohort_info_df['person_id'].unique())
    shuffled_pids = shuffle(pids, random_state=seed)
    cutoff = int(len(shuffled_pids)*train_size)
    train_pids, val_pids = shuffled_pids[:cutoff], shuffled_pids[cutoff:]

    return train_pids, val_pids
    '''


# cohort_info_df = metadata
def split_cohort_abx(cohort_df, outcomes_df, 
                     cohort_info_df,
                     seed, train_size=.7):

    '''
        Given a DataFrame containing cohort features and a DataFrame containing outcome
        labels, splits the features / labels data into train / validation sets on the basis
        of person ID. This ensures that there are no individuals with data in both the training
        and validation sets.
    '''

    train_pids, val_pids = split_pids(cohort_info_df, outcomes_df, seed, train_size)

    train_eids = cohort_info_df[cohort_info_df['person_id'].isin(set(train_pids))]['example_id'].values
    val_eids = cohort_info_df[cohort_info_df['person_id'].isin(set(val_pids))]['example_id'].values

    # Extract features for train / val example IDs
    train_cohort_df = cohort_df[cohort_df['example_id'].isin(set(train_eids))]
    val_cohort_df = cohort_df[cohort_df['example_id'].isin(set(val_eids))]

    logging.info(f"Train cohort size: {len(train_cohort_df)}")
    logging.info(f"Validation cohort size: {len(val_cohort_df)}")

    # Extract outcome labels for train / val cohorts - ensure same example ID order by merging
    train_outcomes_df = train_cohort_df[['example_id']].merge(outcomes_df, on='example_id', how='inner')
    val_outcomes_df =  val_cohort_df[['example_id']].merge(outcomes_df, on='example_id', how='inner')

    assert list(train_cohort_df['example_id'].values) == list(train_outcomes_df['example_id'].values)
    assert list(val_cohort_df['example_id'].values) == list(val_outcomes_df['example_id'].values)

    return train_cohort_df, train_outcomes_df, val_cohort_df, val_outcomes_df
