from tqdm import tqdm
import os
import json
import numpy as np


def main():

    def process_embedding_files_and_create_matrices(directory, layer, model1, model2, model3):
        model1_vectors = []
        model2_vectors = []
        model3_vectors = []
        files = sorted(os.listdir(os.path.join(directory, model1)))
       
        for filename in tqdm(files):
            identifier = filename.split('_')[-1]
            model2_path = f'{model2}/embeddings_words_{identifier}'
            model3_path = f'{model3}/embeddings_words_{identifier}'

            model1_path = os.path.join(directory, f'{model1}/{filename}')
            model2_path = os.path.join(directory, model2_path)
            model3_path = os.path.join(directory, model3_path)

            if os.path.isfile(model2_path):
                with open(model1_path, 'r') as model1_file, open(model2_path, 'r') as model2_file, open(model3_path, 'r') as model3_file:
                    model1_data = json.load(model1_file)
                    model2_data = json.load(model2_file)
                    model3_data = json.load(model3_file)

                    # Filter keys that appear in all files
                    audio, _ = os.path.splitext(identifier)
                    common_keys = set(model1_data[audio].keys()) & set(model2_data[audio].keys()) & set(model3_data[audio].keys())
                    common_keys = sorted(common_keys) 
                    for key in common_keys:
                        model1_vector = model1_data[audio][key][0][layer]
                        model2_vector = model2_data[audio][key][layer]
                        model3_vector = model3_data[audio][key][0][layer]
                       
                        model1_vectors.append(model1_vector)
                        model2_vectors.append(model2_vector)
                        model3_vectors.append(model3_vector)
                    

        # Save the lists to JSON files
        
        with open(os.path.join('..', 'experiments', 'layers', f'embeddings_layer{layer}_{model1}.json'), 'w') as f:
            json.dump(model1_vectors, f) 
            
        with open(os.path.join('..', 'experiments', 'layers', f'embeddings_layer{layer}_{model2}.json'), 'w') as f:
            json.dump(model2_vectors, f)

        with open(os.path.join('..', 'experiments', 'layers', f'embeddings_layer{layer}_{model3}.json'), 'w') as f:
            json.dump(model3_vectors, f)
                
        return model1_vectors, model2_vectors, model3_vectors                    

    for i in tqdm(range(12)):
        process_embedding_files_and_create_matrices('../experiments', layer=i, model1='wav2vec2', model2='glove', model3='bert-base-uncased')


if __name__ == '__main__':
    main()






