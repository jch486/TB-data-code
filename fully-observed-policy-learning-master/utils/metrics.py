import pandas as pd

# returns metrics in policy_outcomes_df
def get_metrics(policy_outcomes_df, policy_colname='action', include_defer=False):

    '''
    example of what policy_outcomes_df would look like
    | example_id | action       | ban_ads | increase_tax | run_campaign |
    | ---------- | ------------ | ------- | ------------ | ------------ |
    | 1          | ban_ads      | 1       | 0            | 0            |
    | 2          | run_campaign | 0       | 0            | 1            |
    | 3          | increase_tax | 0       | 0            | 1            |
    | 4          | ban_ads      | 0       | 1            | 0            |
    '''

    # accuracy = policy_outcomes_df.apply(
    #     lambda x: x[f'{x[policy_colname]}']==1, axis=1).mean()
    
    # number of correct predictions of TB
    correct_TB = ((policy_outcomes_df['has_tb'] == 1) & (policy_outcomes_df['action'] == 'has_tb')).sum()
    # total patients with outcomes of TB
    total_TB = (policy_outcomes_df['has_tb'] == 1).sum()
    # number of correct predictions of no TB
    correct_no_TB = ((policy_outcomes_df['has_no_tb'] == 1) & (policy_outcomes_df['action'] == 'has_no_tb')).sum()
    # total patients with outcomes of no TB
    total_no_TB = (policy_outcomes_df['has_no_tb'] == 1).sum()
    
    return {
        # 'accuracy': accuracy, 
        # 'correct TB': correct_TB, 
        # 'total TB': total_TB, 
        # 'correct no TB': correct_no_TB, 
        # 'total no TB': total_no_TB, 
        'TPR': correct_TB / total_TB, 
        'FPR': 1 - correct_no_TB / total_no_TB
    }

    '''
    # iat = ineffective antibiotic therapy rate
    # axis = 1 = columns so the function is applied to each row
    # x[policy_colname] = one of the actions in ['NIT', 'SXT', 'CIP', 'LVX']
    # the comparison x[f'{x[policy_colname]}']==1 checks whether the treatment was ineffective
    iat = policy_outcomes_df.apply(
        lambda x: x[f'{x[policy_colname]}']==1, axis=1).mean()

    broad = policy_outcomes_df.apply(
        lambda x: x[policy_colname] in ['CIP', 'LVX'], axis=1).mean() 
    
    return {
        'iat': iat, 'broad': broad
    }
    '''

def get_metrics_with_deferral(policy_outcomes_df, policy_colname='action'):

    assert 'prescription' in policy_outcomes_df.columns
    policy_outcomes_df['action_final'] =  policy_outcomes_df.apply(
            lambda x: x[policy_colname] if x[policy_colname] != 'defer' else x['prescription'],
            axis=1
        )
    
    iat = policy_outcomes_df.apply(
        lambda x: x[f"{x['action_final']}"]==1, axis=1).mean()

    broad = policy_outcomes_df.apply(
        lambda x: x['action_final'] in ['CIP', 'LVX'], axis=1).mean() 

    decision_cohort = policy_outcomes_df[policy_outcomes_df[policy_colname] != 'defer']

    iat_alg = decision_cohort.apply(
        lambda x: x[f"{x['action_final']}"]==1, axis=1).mean()

    broad_alg = decision_cohort.apply(
        lambda x: x['action_final'] in ['CIP', 'LVX'], axis=1).mean() 

    iat_doc = decision_cohort.apply(
        lambda x: x[f"{x['prescription']}"]==1, axis=1).mean()

    broad_doc = decision_cohort.apply(
        lambda x: x['prescription'] in ['CIP', 'LVX'], axis=1).mean() 
    
    return {
        'iat': iat, 'broad': broad,
        'iat_alg': iat_alg, 'broad_alg': broad_alg,
        'iat_doc': iat_doc, 'broad_doc': broad_doc,
        'defer_rate': 1 - len(decision_cohort)/len(policy_outcomes_df)
    }
