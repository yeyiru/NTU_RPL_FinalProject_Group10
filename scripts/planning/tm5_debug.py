import numpy as np
file_path="scripts/neural_rendering/data/dataset/dtu_dataset/rs_dtu_4/DTU/scan1/cameras.npz"
poem=np.load(file_path,allow_pickle=True)
poem.files