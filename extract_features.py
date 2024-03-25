import os
import json
from utils_features import get_embedding_across_layers, process_embeddings_for_word, save_json_with_new_embedding
import s3prl.hub as hub
import argparse

"""
Info that I've learnt from sp3prl:
  reps = model(waveform)["hidden_states"]
  reps[0] output cnns
  reps[i] for i in 1 to len-1 -> hidden state for layer i
  reps[i][j] hidden state for layer i and audio j
  reps[i][j][k] hidden state for layer i and audio j and frame k (waveform can be more that such 1 audio)
"""

def main():
    
    parser = argparse.ArgumentParser(description='')

    parser.add_argument('--model', type=str, required=False, help='', default="wav2vec") 
    parser.add_argument('--path', type=str, required=False, help='path to words or phones file', default='words.json') 
    parser.add_argument('--path_counts', type=str, required=False, help='path to words or phones counts file', default='phones_counts.json') 
    parser.add_argument('--select', type=str, required=False, help='words or phones', default='phones')
    parser.add_argument('--audio_dir', type=str, required=False, help='Directory containing the audio files', default='./librispeech_data')  
    parser.add_argument('--device', type=str, required=False, default='cpu')      
    args = parser.parse_args()

    audio_dir = args.audio_dir

    with open(args.path, 'r') as f:
        data = json.load(f)

    with open(args.path_counts, 'r') as f:
        data_counts = json.load(f)    

    model = getattr(hub, args.model)()  # build the Wav2Vec 2.0 model with pre-trained weights
    device = args.device # or gpu
    model = model.to(device)

    output_file = f'embedding_{args.select}.json'

    # Process each word and save its averaged embeddings
    i = 0
    for key, items in data.items():
        if data_counts[key] > 30:
            embeddings_per_layer = process_embeddings_for_word(items, audio_dir, model, device)
            embeddings_as_lists = [emb.tolist() for emb in embeddings_per_layer]
            i += 1
            save_json_with_new_embedding(output_file, key, embeddings_as_lists)
            print(i)

if __name__ == '__main__':
    main()

