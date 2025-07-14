## Usage:
1. Add .csv files of the data to a folder called subsample.
2. Run ICD-9-10_convert.py in the folder ICD-9-10_converters to convert ICD-9 codes to ICD-10.
3. Run construct_features.py and construct_outcomes_and_metadata.py to create feature vectors and outcomes and metadata files.

   info: feature vectors are constructed using [Pat2Vec](https://ai.jmir.org/2023/1/e40755) ([model](https://huggingface.co/zidatasciencelab/Pat2Vec))
4. Run run_policy_learning.zsh to produce results.
