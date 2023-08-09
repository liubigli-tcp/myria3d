# main

# 3.4.8
- Raise an informative error in case of unexpected task_name.

# 3.4.7
- Remove tqdm when splitting a lidar tile to avoid cluttered logs during data preparation.

# 3.4.6
- Document the possible use of ign-pdal-tools for colorization.

# 3.4.5
- Set a default task_name (fit) to avoid common error at lauch time.

# 3.4.4
- Remove duplicated experiment configuration.

# 3.4.3
- Remove outdated and incorrect hydra parameter in config.yaml.

# 3.4.2
- Reconstruct absolute path of input LAS files explicitely, removing a costly glob operation.

# 3.4.1
- Fix dataset description for pacasam: there was an unwanted int-to-int mapping in classification_dict.

# 3.4.0
- Allow inference for the smallest possible patches (num_nodes=1) to have consistent inference behavior 