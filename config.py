# config.py
"""
CTRFT Central Configuration File

Place this in your CTRFT-main or main working directory.

Usage in any module:
    from config import Terms, trees, n_class
"""

# Number of terms per tree (default for CTRFT)
Terms = 40
# Number of trees in the forest
trees = 400
# Number of target classes in your dataset
n_class = 10

# === Optional future additions ===
# max_depth = 10
# min_samples_split = 2
# random_state = 42
