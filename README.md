## Usage:
1. Add .csv files of the data to a folder called subsample.
2. Run ICD-9-10_convert.py in the folder ICD-9-10_converters to convert ICD-9 codes to ICD-10.
3. Run construct_other_data.py to create feature vectors and outcomes and metadata files.

   info: feature vectors are constructed using [Pat2Vec](https://ai.jmir.org/2023/1/e40755) ([model](https://huggingface.co/zidatasciencelab/Pat2Vec))
4. Run run_policy_learning.zsh to train models.
5. Results are stored in experiment_results/my_experiment/results
