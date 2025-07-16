## Usage:
1. Add .csv files of the data to a folder called subsample.
2. Run construct_other_data.py to convert ICD-9 codes to ICD-10 and create feature vectors, outcomes, and metadata files.

   info: feature vectors are constructed using [2022 ICD-10 CM 10D embedding](https://github.com/kaneplusplus/icd-10-cm-embedding)
3. Run run_policy_learning.zsh to train models.
4. Results are stored in experiment_results/my_experiment/results
