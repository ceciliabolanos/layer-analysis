from sklearn.cross_decomposition import CCA
from tqdm import tqdm
import os
import json
import numpy as np


def process_embedding_files_and_create_matrices(directory, layer, model1, model2):
    # Initialize matrices for wav2vec2 and glove embeddings
    model1_matrix = None
    model2_matrix = None

    for filename in tqdm(os.listdir(directory)):
        if filename.startswith(f'embeddings_{model1}_words_'):
            # Extract the identifier (anumber)
            identifier = filename.split('_')[-1]
            model2_path = f'embeddings_{model2}_words_{identifier}'

            model1_path = os.path.join(directory, filename)
            model2_path = os.path.join(directory, model2_path)

            if os.path.isfile(model2_path):
                with open(model1_path, 'r') as model1_file, open(model2_path, 'r') as model2_file:
                    model1_data = json.load(model1_file)
                    model2_data = json.load(model2_file)

                    # Filter keys that appear in both files
                    audio, _ = os.path.splitext(identifier)
                    common_keys = set(model1_data[audio].keys()) & set(model2_data[audio].keys())
                   
                    for key in common_keys:
                        model1_vector = np.array(model1_data[audio][key][0][layer])
                        model2_vector = np.array(model2_data[audio][key][layer])

                        # Append the vectors to their respective matrices
                        if model1_matrix is None:
                            model1_matrix = model1_vector
                        else:
                            model1_matrix = np.vstack((model1_matrix, model1_vector))

                        if model2_matrix is None:
                            model2_matrix = model2_vector
                        else:
                            model2_matrix = np.vstack((model2_matrix, model2_vector))
   
    return model1_matrix, model2_matrix

directory_path = '../experiments'  # Replace with the path to your directory
cca_results = []

for i in range(12):
    wav2vec2_matrix, glove_matrix = process_embedding_files_and_create_matrices(directory_path, layer=i, model1='wav2vec2', model2='glove')
    cca = CCA(n_components=300)
    cca.fit(wav2vec2_matrix, glove_matrix)
    X_c, Y_c = cca.transform(wav2vec2_matrix, glove_matrix)
    correlation = np.corrcoef(X_c.T, Y_c.T).diagonal(offset=300)
    cca_results.append(correlation)

# Specify the filename
filename = 'correlations.json'

# Open a file in write mode ('w') and save the list to it
with open(filename, 'w') as f:
    json.dump(cca_results, f)