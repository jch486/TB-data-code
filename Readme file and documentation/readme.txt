The following is a summary of the csv files and corresponding variables contained in this folder.

timemap.csv - longitudinal timemap that contains all of the distinct dates on which any patient in the cohort had a healthcare claim
 -  patient_id - patient identifier 
 -  date - date of service (default: the Unix epoch of "1970-01-01”)
 -  mdcr - indicator if claims appear in the MDCR database
 -  ccae - indicator if claims appear in the CCAE database
 -  medicaid - indicator if claims appear in the medicaid database
 -  outpatient - indicator if claims occurred in an outpatient setting 
    (Care provided without being admitted to a hospital (e.g., clinic visits, same-day surgeries, ER visits without admission).
 -  ed - indicator if claims occurred in an ED setting 
    (Emergency Department setting, i.e., care provided in a hospital's emergency room.)
 -  obs_stay - indicator if claims occurred in an observational stay 
    (A hospital stay where a patient is not formally admitted as an inpatient, but is kept under observation)
 -  inpatient - indicator if claims occurred in an inpatient stay 
    (A patient is formally admitted to a hospital with a physician’s order and stays at least overnight)
 -  rx - indicator if claims generated from outpatient prescription drugs
    (Prescribed by a healthcare provider. Filled at a outpatient pharmacy. Taken by the patient at home)

index_tb_date.csv - contains the index diagnosis date for tuberculosis
 -  patient_id - patient identifier 
 -  index_date - date of index TB diagnosis
 -  time_before_index - number of days of continuous enrollment before the index date 
 -  max_time_before_index - total number of days of enrollment between index date and the earliest enrollment date (NA if family member of case)
 
tb_dx_visits.csv - contains all visit dates where a diagnosis of tuberculosis was received
 -  patient_id - patient identifier 
 -  dx - ICD diagnosis code
    (standardized code used to classify and record diseases, conditions, and other health-related issues)
 -  dx_ver - ICD code version (ICD-9 or ICD-10)
 -  inpatient - indicator if diagnosis occurred in inpatient setting
 -  date - date of service (default: the Unix epoch of "1970-01-01”)
 
 enrollment_periods.csv - describes periods of continuous enrollment and medication claims coverage
 -  patient_id - patient identifier 
 -  period - integer marker for period of continuous enrollment (new values generated each time there is a break in enrollment)
 -  dtstart - enrollment period start date (default: the Unix epoch of "1970-01-01”)
 -  dtend - enrollment period end date (default: the Unix epoch of "1970-01-01”)
 
all_enroll_ccae_mdcr.csv - contains summary enrollment information on the extracted patient cohort from the CCAE or MDCR databases (i.e., not medicaid)
 ⁃  patient_id - patient identifier 
 ⁃  efamid - enrollee family identifier
 ⁃  dobyr - patient date of birth year
 ⁃  sex - patient sex (1=male, 2=female)
 ⁃  emprel - relation to employee (primary on plan)
 ⁃  first_year - first year the enrollee appears in the database
 ⁃  last_year - last year the enrollee appears in the database
 ⁃  enrmon - total number of months the patient was enrolled
 ⁃  first_date - first date of enrollment (default: the Unix epoch of "1970-01-01”) 
 ⁃  last_date - last date of enrollment (default: the Unix epoch of "1970-01-01”) 
 ⁃  ccae_first_year - first year the enrollee appears in the CCAE database
 ⁃  ccae_last_year - last year the enrollee appears in the CCAE database
 ⁃  ccae_enrmon - total number of months the patient was enrolled in the CCAE database
 ⁃  mdcr_first_year - first year the enrollee appears in the MDCR database
 ⁃  mdcr_last_year - last year the enrollee appears in the MDCR database
 ⁃  mdcr_enrmon - total number of months the patient was enrolled in the MDCR database
 
 
all_enroll_medicaid.csv - contains summary enrollment information on the extracted patient cohort from the Medicaid database
 ⁃  patient_id - patient identifier 
 ⁃  dobyr - patient date of birth year
 ⁃  sex - patient sex (1=male, 2=female)
 -  stdrace - patient race 
 ⁃  enrmon - total number of months the patient was enrolled
 ⁃  first_year - first year the enrollee appears in the database
 ⁃  last_year - last year the enrollee appears in the database
 ⁃  first_date - first date of enrollment (default: the Unix epoch of "1970-01-01”) 
 ⁃  last_date - last date of enrollment (default: the Unix epoch of "1970-01-01”)
 
 all_dx_visits.csv - contains all diagnosis codes that were recorded
 -  patient_id - patient identifier 
 -  dx - diagnosis code 
 -  dx_ver - diagnosis code version (ICD-9 or ICD-10)
 -  inpatient - indicator if the diagnosis code occured in an inpatient setting
 -  date - diagnosis date (default: the Unix epoch of "1970-01-01”)
 
icd_labels - contains a list of icd codes and descriptions 
    (Describe why the patient was seen (the disease or condition))
 -  dx - diagnosis code 
 -  dx_ver - diagnosis code version (ICD-9 or ICD-10)
 -  desc - description
 
all_proc_visits.csv - contains all procedure codes that were recorded
    (Describe what was done (the treatment or service))
 -  patient_id - patient identifier 
 -  proc - procedure code (Usually a CPT4 code. ICD codes and HCPC codes appear occasionally)
 -  inpatient - indicator if the procedure code occurred in an inpatient setting
 -  date - procedure date (default: the Unix epoch of "1970-01-01”)
 
all_rx_visits.csv - contains all prescription drug codes that were recorded
 -  patient_id - patient identifier 
 -  date - prescription fill date (default: the Unix epoch of "1970-01-01”)
 -  ndcnum - national drug code (see redbook file for details)
 -  daysupp - days supplied
 
redbook.csv - contains ndc codes and medication information for all medications. See marketscan data dictionary for summary of variables.

stdprov_dates - visit dates to different provider types
 -  patient_id - patient identifier 
 -  date - prescription fill date (default: the Unix epoch of "1970-01-01”)
 -  stdprov - provider type (see MarketScan data dictionary for list of stdprov values)
 
stdplac_dates - visit dates to different places of care
 -  patient_id - patient identifier 
 -  date - prescription fill date (default: the Unix epoch of "1970-01-01”)
 -  stdprov - place of care type (see MarketScan data dictionary for list of stdplac values)
 
patient_location - contains patient location and corresponding enrollment periods
 -  patient_id - patient identifier 
 -  year - enrollment year
 -  month - enrollment month
 -  dtend - enrollment end date
 -  dtstart - enrollment start date
 -  msa - metropolitan statistical area of employee (See MarketScan data dictionary for MSA values)
 -  egeoloc - geographic location of employee (See MarketScan data dictionary for egeoloc values)
 
 
 
 