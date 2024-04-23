import os
import json
import torch
import numpy as np
from torchmetrics.functional.pairwise import pairwise_cosine_similarity
import gc
import argparse
from tqdm import tqdm

def main():

    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--keys', type=str, default='words_in_order.json') 
    args = parser.parse_args()
    
    with open(args.keys, 'r') as f:
        keys = json.load(f)

    for h in tqdm(range(1,12)):
        retrieval = []
        path_layer1 = f'../experiments/layers/embeddings_layer{h}_wav2vec2.json'
        with open(path_layer1, 'r') as f:
            audio = np.array(json.load(f))
            n = audio.shape[0]    
            
        path_layer2 = f'../experiments/layers/embeddings_layer{h}_bert-base-uncased.json'
        with open(path_layer2, 'r') as f:
            nlp = np.array(json.load(f))
        # Delete 10% of the observations for testing
        np.random.seed(2211) 
        rows_to_delete = np.random.choice(n, int(n*0.1), replace=False)
        deleted_rows = audio[rows_to_delete]  # Save the rows we're about to delete
        audio_new = np.delete(audio, rows_to_delete, axis=0)
        nlp_new = np.delete(nlp, rows_to_delete, axis=0)
        keys_new = np.delete(keys, rows_to_delete)
        
        p = 1
        size = n - int(n*0.1)
        k = 4221
        print(int(n*0.1), k, size) 

        # Process nlp similatiry matrix
        large_matrix = torch.zeros(size, size, dtype=torch.float32, device='cpu')
        audio_new = torch.from_numpy(audio_new)
        nlp_new = torch.from_numpy(nlp_new)
        batch_size = 5000
        for i in range(0,  size, batch_size):
            end = min(i + batch_size,  size)
            batch_embeddings = nlp_new[i:end]
            output = pairwise_cosine_similarity(batch_embeddings, nlp_new)
            values, indices = torch.topk(output, k=k, dim=1)
            zero_matrix = torch.zeros_like(output)

            for row_idx, col_indices in enumerate(indices):
                zero_matrix[row_idx, col_indices] = values[row_idx] ** p

            large_matrix[i:end,:] = zero_matrix

            del output, zero_matrix
            gc.collect()

        batch_size = 10000  
        for j in tqdm(range(int(n*0.1))):
            audio_rep = torch.from_numpy(deleted_rows[j])
            relative_rep = pairwise_cosine_similarity(audio_rep.unsqueeze(0).double(), audio_new)
            similarity_results = []
            for i in range(0, size, batch_size):
                end = min(i + batch_size, size)
                batch = large_matrix[i:end, :] 
                similarity = pairwise_cosine_similarity(relative_rep.float(), batch)
                similarity_results.extend(similarity.cpu())
            full_similarity = torch.cat(similarity_results, dim=0)    
            values, indices = torch.topk(full_similarity, k=6)
            retrieval.append([keys[rows_to_delete[j]], [keys_new[index.item()] for index in indices]])
        
        info = {'words_retrieval': retrieval,
                'rows_deleted': rows_to_delete.tolist()}
        
        with open(os.path.join('results',f'retrieval_w2v_{h}_bert_{h}_p{p}_k{k}.json'), 'w') as f:
            json.dump(info, f)     

if __name__ == '__main__':
    main()




   