import pandas as pd
import numpy as np

from collections import defaultdict
import itertools
from datetime import datetime

import argparse
import logging
import os
import sys
import json
sys.path.append('../')

from utils.metrics import get_metrics
from utils.utils import convert_fnr_to_prob

# returns the outcome with the least cost whose predicted probability is below the outcome's threshold
def get_policy(row, thresholds, outcomes):

    '''
        Inputs:
            - outcomes (list): List of actions to choose from, in order of 
            ascending cost incurred (i.e., first action in list incurs lowest cost)
    '''

    for outcome in outcomes:
        if row[f'predicted_prob_{outcome}'] < thresholds[outcome]:
            return outcome
    
    return outcomes[0]

# generates a list of all possible combinations of threshold values for each outcome
def create_threshold_combos(threshold_vals):

    '''
        Inputs:
            - threshold_vals (dict): Dictionary mapping each outcome name to 
            list of thresholds to search over for that outcome

        Outputs:
            - List of dictionaries, where each dictionary contains a particular 
            outcome to threshold mapping combination to used when selecting optimal
            thresholds, e.g.:

            [{'outcome 1': 0.1, 'outcome 2': 0.5}, {'outcome 1': 0.3, 'outcome 2': 0.5}, ...]
    '''

    outcomes = list(threshold_vals.keys())
    thresholds_list = [threshold_vals[outcome] for outcome in outcomes]

    threshold_combos_list = list(itertools.product(*thresholds_list))
    threshold_combos_dicts = [
        dict(zip(outcomes, combo)) for combo in threshold_combos_list
    ]

    return threshold_combos_dicts
    

# returns the proportions of iat and broad outcomes for the training data and the validation data
def get_stats_for_train_val_preds(train_preds_outcomes, val_preds_outcomes, threshold_setting):

    outcome_order = ['NIT', 'SXT', 'CIP', 'LVX']
        
    train_preds_outcomes['action'] = train_preds_outcomes.apply(
        lambda x: get_policy(x, threshold_setting, 
            outcomes=outcome_order), axis=1
    )

    val_preds_outcomes['action'] = val_preds_outcomes.apply(
        lambda x: get_policy(x, threshold_setting,
            outcomes=outcome_order), axis=1
    )
    
    train_metrics = get_metrics(train_preds_outcomes)
    val_metrics = get_metrics(val_preds_outcomes)

    return train_metrics, val_metrics
    


def get_stats_for_threshold_combos(preds_df, outcomes_df,
                                 validate=True,
                                 threshold_combos=None,
                                 test_outcomes_df=None,
                                 num_splits=20):
    
   

    outcomes = list(outcomes_df.drop(columns=['example_id']).columns)

    if validate:
        # merge = merge DataFrame or named Series objects with a database-style join
        # inner merge type: use intersection of keys from both frames, 
        #   similar to a SQL inner join; preserve the order of the left keys.
        preds_outcomes_df = preds_df.merge(outcomes_df, on='example_id', how='inner')

    else:
        train_preds_df = preds_df[preds_df['is_train'] == 1]
        test_preds_df  = preds_df[preds_df['is_train'] == 0]

        preds_outcomes_df = train_preds_df.merge(outcomes_df, 
            on='example_id', how='inner')
        test_preds_outcomes_df = test_preds_df.merge(test_outcomes_df,
            on='example_id', how='inner')

    
    train_stats_by_combo, eval_stats_by_combo = [], []

    stat_columns = [
        'iat', 'broad'
    ]

    # i = index, curr_setting = outcomes, settings
    for i, curr_setting in enumerate(threshold_combos): 

        # CIP and LVX are broad-spectrum antibiotics
        if curr_setting['CIP'] != curr_setting['LVX']: continue

        logging.info(f'Working on combination {i} / {len(threshold_combos)}')
        # defaultdict(list) creates a dictionary where the values are lists
        train_stats_for_curr_setting = defaultdict(list)
        eval_stats_for_curr_setting = defaultdict(list)

        for split in range(num_splits):
            
            if validate:
                # filters preds_outcomes_df to only keep rows belonging to the current split
                preds_for_split = preds_outcomes_df[preds_outcomes_df['split_ct'] == split]
                # splits filtered data into training data and validation data
                train_preds_outcomes = preds_for_split[preds_for_split['is_train'] == 1].copy()
                eval_preds_outcomes = preds_for_split[preds_for_split['is_train'] == 0].copy()

            else:
                # filters preds_outcomes_df to only keep rows belonging to the current split and
                # splits filtered data into training data and validation data
                train_preds_outcomes = preds_outcomes_df[preds_outcomes_df['split_ct'] == split].copy()
                eval_preds_outcomes = test_preds_outcomes_df[test_preds_outcomes_df['split_ct'] == split].copy()

            
            ##########################################################################################

            # This code is only used if threshold space is initially defined in terms of FNRs.
            # Comment out this section if threshold space is already defined in terms of probabilities

            prob_threshold_setting = {
                outcome: convert_fnr_to_prob(train_preds_outcomes[outcome], 
                                            train_preds_outcomes[f'predicted_prob_{outcome}'],
                                            setting) 
                for outcome, setting in curr_setting.items()
            }

            ##########################################################################################
            
            # get the proportions of iat and broad outcomes for the training data and the validation data
            train_stats, eval_stats =  get_stats_for_train_val_preds(
                train_preds_outcomes, eval_preds_outcomes, prob_threshold_setting
            )

            # add those proportions to the dictionaries
            for stat in eval_stats.keys():
                train_stats_for_curr_setting[stat].append(train_stats[stat])     
                eval_stats_for_curr_setting[stat].append(eval_stats[stat])       
        
        # list of threshold combinations for outcomes, with stats relating to iat/broad, like the mean, appended
        compiled_train_stats_for_setting = compile_stats_for_setting(train_stats_for_curr_setting,
                                                                    curr_setting,
                                                                    outcomes, stat_columns)

        compiled_eval_stats_for_setting = compile_stats_for_setting(eval_stats_for_curr_setting,
                                                                    curr_setting,
                                                                    outcomes, stat_columns)

        train_stats_by_combo.append(compiled_train_stats_for_setting)
        eval_stats_by_combo.append(compiled_eval_stats_for_setting)
    
    # return dataframes for training and validation data, where each dataframe has 
    # lists of threshold combinations for outcomes, with stats relating to iat/broad, like the mean, appended
    return pd.DataFrame(train_stats_by_combo, columns=outcomes + stat_columns), pd.DataFrame(eval_stats_by_combo, columns=outcomes + stat_columns)


# returns list of threshold combinations for outcomes with stats relating to iat/broad, like the mean, appended
def compile_stats_for_setting(stats_dict, curr_setting, outcomes, stat_columns):
    # curr_setting = threshold combos
    compiled_stats = [curr_setting[outcome] for outcome in outcomes]

    # when calling from get_stats_for_threshold_combos:
    # stat_columns = ['iat', 'broad']
    for stat in stat_columns:
        if stat.endswith('_mean'):
            stat_name = stat[:stat.index('_mean')]
            compiled_stats.append(np.mean(stats_dict[stat_name]))

        elif stat.endswith('_std'):
            stat_name = stat[:stat.index('_std')]
            compiled_stats.append(np.std(stats_dict[stat_name]))

        else:
            compiled_stats.append(np.mean(stats_dict[stat]))

    return compiled_stats


# select the best-performing threshold combinations that satisfy outcome_constraints, while optimizing by optimize_col
def get_best_combos_for_constraints(stats_by_setting, outcome_constraints,
                                    constraint_col, optimize_col):
    
    best_val_stats = []
    
    for constraint in outcome_constraints:
        # stats_by_setting[stats_by_setting[constraint_col] < constraint]
            # = filters stats_by_setting to only include rows (settings) 
            # where the value in constraint_col is less than the current constraint
        # .sort_values(by=optimize_col)
            # = sorts the filtered settings by optimize_col, smallest first
        # .iloc[0:1]
            # = selects the best setting (the first row after sorting)
        best_setting = stats_by_setting[
            stats_by_setting[constraint_col] < constraint
        ].sort_values(by=optimize_col).iloc[0:1]
        
        best_val_stats.append(best_setting)
        
    return pd.concat(best_val_stats, axis=0)


#  build the policy frontier: the best-performing threshold combinations under varying constraints
def construct_policy_frontier(preds_df, outcomes_df, 
                              validate=True,
                              thresholds=None,
                              best_settings_df=None,
                              threshold_selection_config=None,
                              test_outcomes_df=None):

    # if validate, uses thresholds and threshold_selection_config and not best_settings_df
    if validate:
        threshold_combos = create_threshold_combos(thresholds)

        # apply policy and get threshold combinations for outcomes and stats relating to iat/broad
        train_stats_by_setting_df, eval_stats_by_setting_df = get_stats_for_threshold_combos(
            preds_df, outcomes_df,
            threshold_combos=threshold_combos
        )

        constraint_col = threshold_selection_config['constraint_col']
        optimize_col = threshold_selection_config['optimize_col']

        # constraint_start, constraint_stop, constraint_step = threshold_selection_config['constraints']
        # outcome_constraints = np.linspace(
        #         constraint_start, constraint_stop, constraint_step
        #     )

        outcome_constraints = threshold_selection_config['constraints'] 

        # select optimal settings under constraints
        train_policy_frontier_df = get_best_combos_for_constraints(
            train_stats_by_setting_df, 
            outcome_constraints,
            constraint_col=constraint_col,
            optimize_col=optimize_col
        )

        eval_policy_frontier_df = get_best_combos_for_constraints(
            eval_stats_by_setting_df, 
            outcome_constraints,
            constraint_col=constraint_col,
            optimize_col=optimize_col
        )

        policy_frontier_dict = {
            'train': train_policy_frontier_df,
            'val': eval_policy_frontier_df
        }

    # else, uses best_settings_df and not thresholds and threshold_selection_config
    else:
        outcome_cols = outcomes_df.drop(columns=['example_id']).columns
        best_threshold_combos_df = best_settings_df[outcome_cols]

        # iterates over each row in best_threshold_combos_df
        # converts the row (a pandas Series) to a Python dictionary mapping {column_name: value}
        # collects all these dictionaries into a list
        best_threshold_combos = [
            row.to_dict() for _, row in best_threshold_combos_df.iterrows()
        ]

        train_policy_frontier_df, eval_policy_frontier_df = get_stats_for_threshold_combos(
            preds_df, outcomes_df,
            threshold_combos=best_threshold_combos,
            validate=False,
            test_outcomes_df=test_outcomes_df,
            num_splits=1
        )

        policy_frontier_dict = {
            'train': train_policy_frontier_df,
            'test': eval_policy_frontier_df
        }

    # returns the best-performing threshold combinations for the training data and the validation or test data,
    # along with their stats (like iat and broad spectrum rate)
    return policy_frontier_dict

    
