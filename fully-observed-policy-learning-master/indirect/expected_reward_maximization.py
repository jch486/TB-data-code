import sys
import itertools
import operator
import pandas as pd
import logging 

sys.path.append('../')

from utils.metrics import get_metrics
from utils.rewards import get_reward_for_example

# row = outcomes for each action
def get_policy_for_row(row, reward_param):
    rewards = get_reward_for_example(row, reward_param)
    # idxmax() = returns index of first occurrence of maximum over requested axis (rows in this case)
    return rewards.idxmax()


def construct_policy_frontier(preds_df, outcomes_df, reward_params,
                              validate=True,
                              test_outcomes_df=None,
                              num_trials=20):

    # Get names of outcome columns
    outcome_cols = outcomes_df.drop(columns=['example_id']).columns

    # Get all reward parameter combinations to be tested
    param_names = list(reward_params[0].keys())

    metric_names = None

    # Cohorts for which we are going to compute policy performance frontiers
    cohorts_to_evaluate = ['train', 'val'] if validate else ['train', 'test']

    frontiers_dict = {}

    for cohort in cohorts_to_evaluate:

        logging.info(f"Evaluating models on {cohort} cohort...")

        all_stats = []
        for trial in range(num_trials):
            logging.info(f"Calculating metrics for split {trial}")

            # filter preds_df to get a subset of rows corresponding to the current trial and current cohort
            is_train = (cohort == 'train')
            preds_for_split_df = preds_df[(preds_df['split_ct'] == trial) &
                                        (preds_df['is_train'] == is_train)].reset_index(drop=True)

            # loop though each reward parameter combination
            for i, combo in enumerate(reward_params):

                logging.info(f"Evaluating at parameter setting {i} / {len(reward_params)}")

                # renames columns like 'predicted_prob_LVX' to just 'LVX' for easier access.
                # then subsets the DataFrame to include only 'example_id' and those outcome columns
                preds_for_split_df = preds_for_split_df.rename(columns={
                    f'predicted_prob_{outcome}': outcome for outcome in outcome_cols                     
                })[['example_id'] + list(outcome_cols)]

                # for each row, add a new column 'action' with the action the policy chooses
                preds_for_split_df['action'] = preds_for_split_df.apply(
                    lambda x: get_policy_for_row(x, combo), axis=1
                )
                outcomes_to_merge_df = outcomes_df if cohort != 'test' else test_outcomes_df
                # merge preds_for_split_df with outcomes_to_merge_df using 'example_id'
                policy_outcomes_df = preds_for_split_df[['example_id', 'action']].merge(outcomes_to_merge_df, on='example_id', how='inner')

                metrics = get_metrics(policy_outcomes_df)
                if metric_names is None:
                    metric_names = list(metrics.keys())

                # extract current reward parameter values and corresponding metric values
                curr_reward_param_list = [combo[param_name] for param_name in param_names]
                stats_for_trial_combo = [metrics[metric_name] for metric_name in metric_names]

                all_stats.append(curr_reward_param_list + stats_for_trial_combo)
    
        all_stats = pd.DataFrame(all_stats, columns=param_names + metric_names)
        all_stats_means = all_stats.groupby(param_names).mean().reset_index()

        frontiers_dict[cohort] = all_stats_means

    return frontiers_dict

