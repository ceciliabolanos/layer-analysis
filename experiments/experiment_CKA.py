import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import argparse
import json
import os
from tqdm import tqdm
from embeddings.utils_embeddings import linear_CKA

def main():

    parser = argparse.ArgumentParser(description='')

    parser.add_argument('--model1', type=str, default="wav2vec2") 
    parser.add_argument('--layer1', type=int, default=24) 
    parser.add_argument('--model2', type=str, default="bert-base-uncased") 
    parser.add_argument('--layer2', type=int, default=12)  
    args = parser.parse_args()


    # Initialize the CKA similarity matrix.
    cka_similarity = np.zeros((args.layer1, args.layer2))

    # Compute the CKA for each pair of layers.
    for i in tqdm(range(args.layer1)):
        with open(os.path.join('..', 'experiments', 'layers', f'embeddings_layer{i}_{args.model1}.json'), 'r') as f:
            model1_list = json.load(f)
        model1_matrix = np.array(model1_list)
        for j in range(args.layer2):
            with open(os.path.join('..', 'experiments', 'layers', f'embeddings_layer{j}_{args.model2}.json'), 'r') as f:
                model2_list = json.load(f)
            model2_matrix = np.array(model2_list)
            cka_similarity[i, j] = linear_CKA(model1_matrix, model2_matrix)
      
    with open(f'cka_{args.model2}_{args.model1}.json', 'w') as f:
        json.dump(cka_similarity.tolist(), f)

    plt.figure(figsize=(8, 6))
    sns.heatmap(cka_similarity, annot=True, fmt=".1f", cmap='viridis', cbar_kws={'label': 'CKA (Linear)'})
    plt.xlabel(f"Layer {args.model2}")
    plt.ylabel(f"Layer {args.model1}")
    plt.ylim(0, len(cka_similarity))
    plt.savefig(f'cka_{args.model2}_{args.model1}.png', bbox_inches='tight') 
    plt.show()


if __name__ == '__main__':
    main()
