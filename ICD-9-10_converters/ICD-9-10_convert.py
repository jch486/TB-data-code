import pandas as pd
import os
import numpy as np
import math

def icd9_to_icd10(all_dx_visits_df, icd9_to_icd10_df):
    all_dx_visits_df = all_dx_visits_df.drop_duplicates()
    changed_df = pd.DataFrame()
    icd9_half = all_dx_visits_df[all_dx_visits_df['dx_ver'] == 9]
    icd10_half = all_dx_visits_df[all_dx_visits_df['dx_ver'] == 10]
    for row in icd9_half.itertuples(index=False):
        patient_id = row[0]
        inpatient = row[3]
        date = row[4]
        conversions = icd9_to_icd10_lookup(icd9_to_icd10_df, row[1])
        for new_dx in conversions:
            new_row = pd.DataFrame({'patient_id': [patient_id], 'dx': [new_dx], 'dx_ver': [10], 'inpatient': [inpatient], 'date': [date]})
            changed_df = pd.concat([changed_df, new_row], ignore_index=True)

    return pd.concat([changed_df, icd10_half], ignore_index=True)

def icd9_to_icd10_lookup(icd9_to_icd10_df, dx):
    return icd9_to_icd10_df.loc[(icd9_to_icd10_df['icd9cm'] == dx), 'icd10cm']

def main():
    icd9_to_icd10_fn = os.path.join('ICD-9-10_converters', 'icd9toicd10cmgem.csv')
    icd9_to_icd10_df = pd.read_csv(icd9_to_icd10_fn)
    all_dx_visits_fn = os.path.join('subsample', 'all_dx_visits.csv')
    all_dx_visits_df = pd.read_csv(all_dx_visits_fn)

    all_icd_10 = icd9_to_icd10(all_dx_visits_df, icd9_to_icd10_df)
    all_icd_10.to_csv(os.path.join('ICD-9-10_converters', 'all_icd_10.csv'), index=False)


if __name__ == '__main__':
    main()