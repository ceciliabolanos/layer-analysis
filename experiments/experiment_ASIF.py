import os
import json
import torch
import numpy as np
import argparse
from tqdm import tqdm
from ASIF import zero_shot_classification

def main():

    parser = argparse.ArgumentParser(description='')

    parser.add_argument('--path_layer1', type=str, default='../experiments/layers/embeddings_layer6_wav2vec2.json') 
    parser.add_argument('--path_layer2', type=str, default='../experiments/layers/embeddings_layer3_bert-base-uncased.json') 
    parser.add_argument('--k', type=int, default=4221) 
    parser.add_argument('--p', type=int, default=1) 
    parser.add_argument('--keys', type=str, default='words_in_order1.json') 
    args = parser.parse_args()
    
    # Upload the matrices
    with open(args.path_layer1, 'r') as f:
        audio = np.array(json.load(f))

    with open(args.path_layer2, 'r') as f:
        nlp = np.array(json.load(f))

    with open(args.keys, 'r') as f:
        keys = json.load(f)


    all_values = [value for values_list in keys.values() for value in values_list]
    n = audio.shape[0]
    np.random.seed(2211) 
    rows_to_delete = np.random.choice(n, int(n*0.15), replace=False) # Eliminamos el 10% de las filas para retrieval
    deleted_rows = audio[rows_to_delete]  # Guardamos los indices de las filas eliminadas
    audio_new = np.delete(audio, rows_to_delete, axis=0)
    nlp_new = np.delete(nlp, rows_to_delete, axis=0)
    keys_new = np.delete(all_values, rows_to_delete)
    
    audio_new = torch.from_numpy(audio_new)
    nlp_new = torch.from_numpy(nlp_new)
    to_predict = torch.from_numpy(deleted_rows)

    total_anchors = len(audio_new) 
    range_anch = range(total_anchors, total_anchors + 1)

    n_anchors, sims = zero_shot_classification(to_predict, nlp_new, audio_new, nlp_new, non_zeros=args.k, 
                                               range_anch = range_anch, val_exps=[args.p], dic_size = 100_000, max_gpu_mem_gb = 8.)
    
    info = {'retrieval': sims.tolist(),
            'rows_deleted': rows_to_delete.tolist(),
            'keys_anchors': keys_new.tolist()}
    
    with open(os.path.join('results',f'retrieval_a_p{args.p}_k{args.k}.json'), 'w') as f:
        json.dump(info, f)     

if __name__ == '__main__':
    main()




   