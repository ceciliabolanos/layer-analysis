from tqdm import tqdm
import os
import json
import numpy as np
import argparse

def main():
    parser = argparse.ArgumentParser(description='')

    parser.add_argument('--layer', type=int, default=1) 
    parser.add_argument('--model', type=str, default='wav2vec2') 
    parser.add_argument('--words_path', type=str, default='words_in_order1') 
    args = parser.parse_args()

    def process_embedding_files_and_create_matrices(directory, layer, model1, words_path):
        model1_vectors = []
        with open(words_path, 'r'):
           words = json.load(words_path)

        files = sorted(os.listdir(os.path.join(directory, model1)))
       
        for filename in tqdm(files):
            identifier = filename.split('_')[-1]
            model1_path = os.path.join(directory, f'{model1}/{filename}')
        
            with open(model1_path, 'r') as model1_file:
                model1_data = json.load(model1_file)
                audio, _ = os.path.splitext(identifier)
                common_keys = words[identifier]
                
                for key in common_keys:
                    if model1 == 'glove':
                        model1_vector = model1_data[audio][key][layer]
                    else:
                        model1_vector = model1_data[audio][key][0][layer]    

                    model1_vectors.append(model1_vector)
          
        with open(os.path.join('..', 'experiments', 'layers', f'embeddings_layer{layer}_{model1}.json'), 'w') as f:
            json.dump(model1_vectors, f) 

    
    process_embedding_files_and_create_matrices('../experiments', layer=args.layer, model1=args.model, words_path=args.words_paths)