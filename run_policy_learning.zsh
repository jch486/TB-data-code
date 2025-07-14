#!/bin/zsh

# Make sure the script stops on first error
set -e

# Set experiment name
EXP_NAME="my_experiment"

# Set paths to required inputs
FEATURES_PATH="other_data/features_vectors_formatted.csv"
OUTCOMES_PATH="other_data/outcomes.csv"
METADATA_PATH="other_data/metadata.csv"

REWARD_PARAMS_PATH="reward_space.json"                     # For direct or exp_reward_max mode

# Set mode to one of: direct, thresholding, exp_reward_max
MODE="direct"

# Run the script
python fully-observed-policy-learning-master/policy_learning.py \
  --exp_name "$EXP_NAME" \
  --mode "$MODE" \
  --num_trials 20 \
  --features_path "$FEATURES_PATH" \
  --outcomes_path "$OUTCOMES_PATH" \
  --metadata_path "$METADATA_PATH" \
  --validate \
  --reward_params_path "$REWARD_PARAMS_PATH"
