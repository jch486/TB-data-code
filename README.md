## Usage:
1. Add .csv files of the data to a folder called subsample.
2. Run construct_other_data.py to convert ICD-9 codes to ICD-10 and create feature vectors, outcomes, and metadata files.

   info: feature vectors are constructed using [Pat2Vec](https://ai.jmir.org/2023/1/e40755) ([model](https://huggingface.co/zidatasciencelab/Pat2Vec))
3. Run run_policy_learning.zsh to train models.
4. Results are stored in experiment_results/my_experiment/results
