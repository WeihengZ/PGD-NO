import os
import pickle
from scipy.sparse import csr_matrix
import numpy as np

# put all the data into one pickle file
RAW_JEB_DATA_PATH = "./"
ALL_JEB_DATA_PATH = "./all_jeb.pkl"

# find all pickle files in the RAW_JEB_DATA_PATH
pickle_files = [f for f in os.listdir(RAW_JEB_DATA_PATH) if f.endswith('.pkl')]

print(f"Found {len(pickle_files)} pickle files")
# load all the data from the pickle files
all_data = {}
num_files = 0
for file in pickle_files:
    if file == "all_jeb.pkl":
        continue
    num_files += 1
    print(f"Loading file {num_files} of {len(pickle_files)}")
    sim_id = file.split(".")[0]
    with open(os.path.join(RAW_JEB_DATA_PATH, file), 'rb') as f:
        data = pickle.load(f)
    
    # convert the seg_matrix to a dense matrix
    seg_matrix = data["seg_matrix"]
    if isinstance(seg_matrix, csr_matrix):
        seg_matrix = seg_matrix.toarray()
    data["seg_matrix"] = seg_matrix   
    assert isinstance(data["seg_matrix"], np.ndarray), "seg_matrix must be a dense numpy array"

    all_data[sim_id] = data

# save the data to the ALL_JEB_DATA_PATH
with open(ALL_JEB_DATA_PATH, 'wb') as f:
    pickle.dump(all_data, f)