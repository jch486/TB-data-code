import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import math

# plot usage of ICD-9 vs ICD-10 across years
def plot_dx_ver_year(all_dx_visits_df):
    # keep only dx_ver and date from table
    dx_ver_date = all_dx_visits_df[['dx_ver', 'date']]
    # split into ICD-9 and ICD-10 tables, and convert dates to year
    ICD9_year = np.floor(dx_ver_date[dx_ver_date['dx_ver'] == 9]['date'] / 365 + 1970).astype(int)
    ICD10_year = np.floor(dx_ver_date[dx_ver_date['dx_ver'] == 10]['date'] / 365 + 1970).astype(int)

    # compute range of years to plot
    first_year = math.floor(dx_ver_date['date'].to_numpy().min() / 365 + 1970)
    last_year = math.ceil(dx_ver_date['date'].to_numpy().max() / 365 + 1970)
    year_range = range(first_year, last_year)

    # convert series of years to counts of the times those years appeared, for both ICD-9 and ICD-10
    ICD9_year_counts = ICD9_year.value_counts().sort_index()
    ICD9_full_year_range = pd.Series(0, index=year_range)
    ICD9_full_year_range.update(ICD9_year_counts)
    ICD9_result = ICD9_full_year_range.astype(int)

    ICD10_year_counts = ICD10_year.value_counts().sort_index()
    ICD10_full_year_range = pd.Series(0, index=year_range)
    ICD10_full_year_range.update(ICD10_year_counts)
    ICD10_result = ICD10_full_year_range.astype(int)

    plt.figure()
    plt.grid()
    plt.title("Usage of ICD-9 vs ICD-10 across years")
    plt.plot(list(year_range), ICD9_result, 'blue', label='ICD-9')
    plt.plot(list(year_range), ICD10_result, 'red', label='ICD-10')
    plt.xlabel('Year')
    plt.ylabel('Number of diagnoses')
    plt.legend()
    plt.show()

# plot a histogram of index TB diagnoses by year
def plot_index_TB_diagnoses_year(index_tb_date_df):
    diagnosis_dates = index_tb_date_df['index_date']
    # print(diagnosis_dates.head())

    # convert dates to years
    diagnosis_year = diagnosis_dates / 365 + 1970

    # find the first and last year that TB diagnoses were tracked
    first_year = math.floor(diagnosis_year.to_numpy().min())
    last_year = math.ceil(diagnosis_year.to_numpy().max())

    # print("first year of tb diagnoses:", first_year)
    # print("last year of tb diagnoses:", last_year)

    # plotting
    plt.hist(diagnosis_year, np.arange(first_year, last_year))
    plt.title("Index TB diagnoses by year")
    plt.xlabel('Year')
    plt.ylabel('Number of TB diagnoses')
    plt.show()

# plot a histogram of TB diagnoses by year
def plot_TB_diagnoses_year(tb_dx_visits_df):
    # get TB diagnoses dates, removing duplicate patient-date combos
    id_date = tb_dx_visits_df[['patient_id', 'date']]
    unique_id_date = id_date.drop_duplicates()
    diagnosis_dates = unique_id_date['date']

    # find the first and last year that TB diagnoses were tracked
    first_year = math.floor(diagnosis_dates.to_numpy().min() / 365 + 1970)
    last_year = math.ceil(diagnosis_dates.to_numpy().max() / 365 + 1970)

    plt.hist(diagnosis_dates / 365 + 1970, np.arange(first_year, last_year))
    plt.title("TB diagnoses by year")
    plt.xlabel('Year')
    plt.ylabel('Number of TB diagnoses')
    plt.show()

# plot a histogram of TB diagnoses by month
def plot_TB_diagnoses_month(tb_dx_visits_df):
    # get TB diagnoses dates, removing duplicate patient-date combos
    id_date = tb_dx_visits_df[['patient_id', 'date']]
    unique_id_date = id_date.drop_duplicates()
    diagnosis_dates = unique_id_date['date']

    # convert to month
    diagnosis_month = (diagnosis_dates / 30.4167) % 12

    # plotting
    plt.hist(diagnosis_month, np.arange(0, 13))
    plt.title("TB diagnoses by month")
    plt.xlabel('Month')
    plt.ylabel('Number of TB diagnoses')
    plt.show()

# plot a histogram of TB diagnoses by egeoloc code location
def plot_TB_diagnoses_location_egeoloc(patient_location_df):
    # remove duplicate patients
    unique_patient_location_df = patient_location_df.drop_duplicates(subset=['patient_id'])
    # pandas Series with only location codes
    diagnosis_locations = unique_patient_location_df['egeoloc']

    # find the smallest and largest location code numbers
    first_location = diagnosis_locations.to_numpy().min()
    last_location = diagnosis_locations.to_numpy().max()

    # plotting
    plt.hist(diagnosis_locations, np.arange(first_location, last_location))
    plt.title("TB diagnoses by location")
    plt.xlabel('egeoloc code')
    plt.ylabel('Number of TB diagnoses')
    plt.show()

# plot a histogram of TB diagnoses by msa code location
def plot_TB_diagnoses_location_msa(patient_location_df):
    # remove duplicate patients
    unique_patient_location_df = patient_location_df.drop_duplicates(subset=['patient_id'])
    # pandas Series with only metropolitan statistical area codes
    diagnosis_locations = unique_patient_location_df['msa']

    # find the smallest and largest location code numbers
    first_location = diagnosis_locations.to_numpy().min()
    last_location = diagnosis_locations.to_numpy().max()

    # plotting histogram results in bars too thin to see
    '''
    plt.hist(diagnosis_locations, np.arange(first_location, last_location))
    plt.title("TB diagnoses by location")
    plt.xlabel('msa (metropolitan statistical area) code')
    plt.ylabel('Number of TB diagnoses')
    plt.show()
    '''

    # printing works better than plotting
    print("5 most common msa codes:", diagnosis_locations.value_counts().head(5))

# plot TB diagnoses by gender across years
def plot_TB_diagnoses_gender_year(index_tb_date_df, all_enroll_ccae_mdcr_df, all_enroll_medicaid_df):
    # keep only patient_id and sex info from ccae/mdcr and medicaid tables
    ccae_mdcr_gender = all_enroll_ccae_mdcr_df[['patient_id', 'sex']]
    medicaid_gender = all_enroll_medicaid_df[['patient_id', 'sex']]
    # DataFrame with patient_id and index_date
    id_dates = index_tb_date_df[['patient_id', 'index_date']]
    # combine ccae/mdcr and medicaid tables, and remove duplicate patient entries
    all_gender = pd.concat([ccae_mdcr_gender, medicaid_gender], ignore_index=True).drop_duplicates(subset=['patient_id'])
    # DataFrame with patient_id, gender, and index_date info
    gender_year = id_dates.merge(all_gender, on='patient_id')
    # split into male and female patient tables, and convert dates to years
    male_year = np.floor(gender_year[gender_year['sex'] == 1]['index_date'] / 365 + 1970).astype(int)
    female_year = np.floor(gender_year[gender_year['sex'] == 2]['index_date'] / 365 + 1970).astype(int)

    # compute range of years to plot
    first_year = math.floor(id_dates['index_date'].to_numpy().min() / 365 + 1970)
    last_year = math.ceil(id_dates['index_date'].to_numpy().max() / 365 + 1970)
    year_range = range(first_year, last_year)

    # convert series of years to counts of the times those years appeared, for both male and female data
    male_year_counts = male_year.value_counts().sort_index()
    male_full_year_range = pd.Series(0, index=year_range)
    male_full_year_range.update(male_year_counts)
    male_result = male_full_year_range.astype(int)

    female_year_counts = female_year.value_counts().sort_index()
    female_full_year_range = pd.Series(0, index=year_range)
    female_full_year_range.update(female_year_counts)
    female_result = female_full_year_range.astype(int)

    # plotting
    plt.figure()
    plt.grid()
    plt.title("Index TB diagnosis by gender across years")
    plt.plot(list(year_range), male_result, 'blue', label='Male')
    plt.plot(list(year_range), female_result, 'red', label='Female')
    plt.xlabel('Year')
    plt.ylabel('Number of TB diagnoses')
    plt.legend()
    plt.show()

# determine the most common ICD codes (separately for ICD-9 and ICD-10) 
# recorded during the five visits preceding a TB diagnosis
def find_most_common_ICD_codes(index_tb_date_df, all_dx_visits_df, icd_labels_df):
    # DataFrame with patient_id and index_date
    id_dates = index_tb_date_df[['patient_id', 'index_date']]
    # DataFrame with patient_id, index_date, and visit info
    combined_data = id_dates.merge(all_dx_visits_df, on='patient_id')
    # keep only visits that occured before each patient's TB diagnosis
    combined_data_before_diagnosis_only = combined_data[combined_data['date'] < combined_data['index_date']]

    most_recent_five_dates = (
        combined_data_before_diagnosis_only[['patient_id', 'date']]
        # remove duplicates
        .drop_duplicates()
        # sort ascending for patient_id, descending for date
        .sort_values(['patient_id', 'date'], ascending=[True, False])
        # get the most recent (largest number) 5 dates for each patient
        .groupby('patient_id')
        .head(5)
    )

    filtered_df = combined_data_before_diagnosis_only.merge(most_recent_five_dates, on=['patient_id', 'date'], how='inner')

    # split into ICD-9 and ICD-10 tables and sort by counts of dx
    only_icd9 = filtered_df[filtered_df['dx_ver'] == 9]['dx'].value_counts()
    only_icd10 = filtered_df[filtered_df['dx_ver'] == 10]['dx'].value_counts()

    print("Most common ICD-9 codes:")
    for i in range(0,5):
        print(only_icd9.iloc[i:i+1])
        print("description:", icd_lookup(icd_labels_df, only_icd9.index[i], 9), "\n")

    print("\nMost common ICD-10 codes:")
    for i in range(0,5):
        print(only_icd10.iloc[i:i+1])
        print("description:", icd_lookup(icd_labels_df, only_icd10.index[i], 10), "\n")

# determine the most common diagnoses recorded before a visit with/out a TB diagnosis
def find_most_common_dx(dx_features_outcomes, icd_labels_df):
    # split into with TB and without TB tables and sort by counts of dx
    with_TB = dx_features_outcomes[dx_features_outcomes['has_tb'] == 1]['dx'].explode().value_counts()
    without_TB = dx_features_outcomes[dx_features_outcomes['has_no_tb'] == 1]['dx'].explode().value_counts()
    total_with_TB = with_TB.sum()
    total_without_TB = without_TB.sum()
    
    print("Most common dx in the timespan before a visit, with TB:")
    for i in range(0,30):
        print("Code:", with_TB.index[i],", Description:", icd_lookup(icd_labels_df, with_TB.index[i], 10))
        print("Proportion:", round(100*with_TB.iloc[i]/total_with_TB, 3),"%\n")
    
    print("\nMost common dx in the timespan before a visit, without TB:")
    for i in range(0,10):
        print("Code:", without_TB.index[i],", Description:", icd_lookup(icd_labels_df, without_TB.index[i], 10))
        print("Proportion:",round(100*without_TB.iloc[i]/total_without_TB, 3),"%\n")

# plot number of visits per patient for any one-month span
# and find the average
def avg_month_visits(all_dx_visits_df, all_proc_visits_df):
    # keep only patient_id and date (of visit)
    proc_visits_stripped = all_proc_visits_df[['patient_id', 'date']]
    dx_visits_stripped = all_dx_visits_df[['patient_id', 'date']]
    # combine both types of visits and remove duplicate visits
    combined_visits = pd.concat([dx_visits_stripped, proc_visits_stripped], ignore_index=True).drop_duplicates()

    # each month is [month_start, month_end)
    starting_start = combined_visits['date'].to_numpy().min()
    month_length = 30
    starting_end = starting_start + month_length
    end_date = combined_visits['date'].to_numpy().max()

    unique_patients = combined_visits.drop_duplicates(subset=['patient_id'])['patient_id']
    one_month_visits = []

    # for each patient
    for curr_patient in unique_patients:
        # intialize the start and end of the month we are currently looking at
        curr_start = starting_start
        curr_end = starting_end

        # filter for only the visits for the current patient
        curr_patient_visits = combined_visits[combined_visits['patient_id'] == curr_patient]
        
        while curr_start <= end_date + 1:
            # filter for visits in the current month span
            curr_dates_visits = curr_patient_visits[curr_patient_visits['date'].isin(range(curr_start, curr_end))]
            one_month_visits.append(curr_dates_visits.shape[0])

            curr_start += 1
            curr_end += 1
    
    # plotting
    plt.hist(one_month_visits, np.arange(0, np.max(one_month_visits)))
    plt.title("Visits in any one-month period")
    plt.xlabel('Number of visits')
    plt.ylabel('Number of one-month periods')
    plt.show()
    
    # print average visits
    print(np.mean(np.array(one_month_visits)))

# plot number of visits per patient for the month before index TB diagnosis
# and find the average
def avg_month_visits_before_diagnosis(index_tb_date_df, all_dx_visits_df, all_proc_visits_df):
    # keep only patient_id and date (of visit)
    proc_visits_stripped = all_proc_visits_df[['patient_id', 'date']]
    dx_visits_stripped = all_dx_visits_df[['patient_id', 'date']]
    # keep only patient_id and index_date
    id_index_dates = index_tb_date_df[['patient_id', 'index_date']]
    # combine both types of visits and index_date, and remove duplicate visits
    combined_visits = pd.concat([id_index_dates, dx_visits_stripped, proc_visits_stripped], ignore_index=True).drop_duplicates()

    unique_patients = combined_visits.drop_duplicates(subset=['patient_id'])['patient_id']
    one_month_visits = []

    # for each patient
    for curr_patient in unique_patients:
        # calculate start and end of month before TB diagnosis
        # each month is [month_start, month_end)
        month_end = combined_visits.loc[combined_visits['patient_id'] == curr_patient, 'index_date'].iloc[0].astype(int)
        month_length = 30
        month_start = month_end - month_length

        # filter for only the visits for the current patient
        curr_patient_visits = combined_visits[combined_visits['patient_id'] == curr_patient]

        # count number of visits and append to list
        curr_dates_visits = curr_patient_visits[curr_patient_visits['date'].isin(range(month_start, month_end))]
        one_month_visits.append(curr_dates_visits.shape[0])

    # plotting
    plt.hist(one_month_visits, np.arange(0, np.max(one_month_visits)))
    plt.title("Visits in the one-month period before index TB diagnosis")
    plt.xlabel('Number of visits')
    plt.ylabel('Number of one-month periods')
    plt.show()

    # print average visits
    print(np.mean(np.array(one_month_visits)))

# determine the most common diagnoses recorded during a visit with a TB diagnosis
def find_most_common_comorbidities(tb_dx_visits_df, all_dx_visits_df, icd_labels_df):
    # DataFrame with patient_id and TB diagnosis dates
    id_dates = tb_dx_visits_df[['patient_id', 'date']]
    unique_id_date = id_dates.drop_duplicates()
    # DataFrame with patient_id, date, and visit info
    combined_data = unique_id_date.merge(all_dx_visits_df, on=['patient_id', 'date'])

    # split into ICD-9 and ICD-10 tables and sort by counts of dx
    only_icd9 = combined_data[combined_data['dx_ver'] == 9]['dx'].value_counts()
    only_icd10 = combined_data[combined_data['dx_ver'] == 10]['dx'].value_counts()
    
    print("Most common comorbidities, ICD-9:")
    for i in range(0,10):
        print(only_icd9.iloc[i:i+1])
        print("description:", icd_lookup(icd_labels_df, only_icd9.index[i], 9), "\n")
    
    print("\nMost common comorbidities, ICD-10:")
    for i in range(0,10):
        print(only_icd10.iloc[i:i+1])
        print("description:", icd_lookup(icd_labels_df, only_icd10.index[i], 10), "\n")

# return description given diagnosis code and ICD version
def icd_lookup(icd_labels_df, dx, dx_ver):
    return icd_labels_df.loc[(icd_labels_df['dx'] == dx) & (icd_labels_df['dx_ver'] == dx_ver), 'desc'].iloc[0]

# plot TB diagnoses by age across years
def plot_TB_diagnoses_age_year(index_tb_date_df, all_enroll_ccae_mdcr_df, all_enroll_medicaid_df):
    # keep only patient_id and dobyr from tables
    ccae_mdcr_dobyr = all_enroll_ccae_mdcr_df[['patient_id', 'dobyr']]
    medicaid_dobyr = all_enroll_medicaid_df[['patient_id', 'dobyr']]
    # DataFrame with patient_id and index_date
    id_dates = index_tb_date_df[['patient_id', 'index_date']]
    # combine ccae/mdcr and medicaid tables, and remove duplicate patient entries
    all_info = pd.concat([ccae_mdcr_dobyr, medicaid_dobyr], ignore_index=True).drop_duplicates(subset=['patient_id'])
    # DataFrame with patient_id, dobyr, and index_date info
    age_date = id_dates.merge(all_info, on='patient_id')
    # convert dates to years
    age_date['index_date'] = np.floor(age_date['index_date'] / 365 + 1970).astype(int)
    ages = age_date['index_date'] - age_date['dobyr']

    # compute range of ages to plot
    age_range = range(ages.to_numpy().min(), ages.to_numpy().max()+1)

    # convert series of ages to counts of the times those ages appeared
    # and initialize counts of ages in age_range that did not appear to 0
    age_counts = ages.value_counts().sort_index()
    age_counts_with_zeros = age_counts.reindex(age_range).fillna(0)
    
    # plotting
    plt.figure()
    plt.grid()
    plt.title("TB diagnoses by age")
    plt.plot(list(age_range), age_counts_with_zeros)
    plt.xlabel('Age')
    plt.ylabel('Number of TB diagnoses')
    plt.show()

def main():
    # load data from csv files
    index_tb_date_fn = os.path.join('subsample', 'index_tb_date.csv')
    index_tb_date_df = pd.read_csv(index_tb_date_fn)
    tb_dx_visits_fn = os.path.join('subsample', 'tb_dx_visits.csv')
    tb_dx_visits_df = pd.read_csv(tb_dx_visits_fn)
    patient_location_fn = os.path.join('subsample', 'patient_location.csv')
    patient_location_df = pd.read_csv(patient_location_fn)
    all_enroll_ccae_mdcr_fn = os.path.join('subsample', 'all_enroll_ccae_mdcr.csv')
    all_enroll_ccae_mdcr_df = pd.read_csv(all_enroll_ccae_mdcr_fn)
    all_enroll_medicaid_fn = os.path.join('subsample', 'all_enroll_medicaid.csv')
    all_enroll_medicaid_df = pd.read_csv(all_enroll_medicaid_fn)
    all_dx_visits_fn = os.path.join('subsample', 'all_dx_visits.csv')
    all_dx_visits_df = pd.read_csv(all_dx_visits_fn)
    all_proc_visits_fn = os.path.join('subsample', 'all_proc_visits.csv')
    all_proc_visits_df = pd.read_csv(all_proc_visits_fn)
    icd_labels_fn = os.path.join('subsample', 'icd_labels.csv')
    icd_labels_df = pd.read_csv(icd_labels_fn)
    dx_features_outcomes_fn = os.path.join('other_data', 'dx_features_outcomes.pkl')
    dx_features_outcomes = pd.read_pickle(dx_features_outcomes_fn)

    # plot usage of ICD-9 vs ICD-10 across years
    # plot_dx_ver_year(all_dx_visits_df)

    # plot a histogram of index TB diagnoses by year
    # plot_index_TB_diagnoses_year(index_tb_date_df)

    # plot a histogram of TB diagnoses by year
    # plot_TB_diagnoses_year(tb_dx_visits_df)

    # plot a histogram of TB diagnoses by egeoloc code location
    # plot_TB_diagnoses_location_egeoloc(patient_location_df)

    # plot a histogram of TB diagnoses by msa (metropolitan statistical area)
    # plot_TB_diagnoses_location_msa(patient_location_df)

    # plot TB diagnoses by gender across years
    # plot_TB_diagnoses_gender_year(index_tb_date_df, all_enroll_ccae_mdcr_df, all_enroll_medicaid_df)

    # determine the most common ICD codes (separately for ICD-9 and ICD-10) 
    # recorded during the five visits preceding a TB diagnosis.
    # find_most_common_ICD_codes(index_tb_date_df, all_dx_visits_df, icd_labels_df)

    # find most common diagnoses before visit with/without TB diagnosis
    find_most_common_dx(dx_features_outcomes, icd_labels_df)

    # plot number of diagnosis + procedure visits per patient for any one-month span
    # and find the average
    # avg_month_visits(all_dx_visits_df, all_proc_visits_df)

    # plot number of diagnosis + procedure visits per patient for the month before index TB diagnosis
    # and find the average
    # avg_month_visits_before_diagnosis(index_tb_date_df, all_dx_visits_df, all_proc_visits_df)

    # determine the most common diagnoses recorded during a visit with a TB diagnosis
    # find_most_common_comorbidities(tb_dx_visits_df, all_dx_visits_df, icd_labels_df)

    # plot TB diagnoses by age across years
    # plot_TB_diagnoses_age_year(index_tb_date_df, all_enroll_ccae_mdcr_df, all_enroll_medicaid_df)

if __name__ == '__main__':
    main()