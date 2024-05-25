import os
import numpy as np
import torch
import json
from utils_embeddings import get_embedding_across_layers_audio, get_embeddings_audio, save_json_with_embedding
import argparse
from tqdm import tqdm
import sys
sys.path.insert(0, '../layer-analysis')
from easyaudio.easyaudio.hub import get_model

"""
Info that I've learned from easyaudio:
  reps = model(waveform)["hidden_states"]
  reps[i] for i in 0 to len-1 -> hidden state for layer i
  reps[i][j] hidden state for layer i and frame j
"""

def main():
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    parser = argparse.ArgumentParser(description='')

    parser.add_argument('--model', type=str, required=False, help='', default="encodecmae_base") 
    parser.add_argument('--path', type=str, required=False, help='path to words file', default='alignments/audio_alignments_13.33_24.json') 
    parser.add_argument('--device', type=str, required=False, default='cuda')      
    args = parser.parse_args()

    with open(args.path, 'r') as f:
        data = json.load(f)

    model = get_model(args.model)

    # For every audio, process each word and save its averaged embeddings   
    for audio, items in tqdm(data.items(), desc="Processing audio files"):
        embeddings_audio = {}
        audio_path = os.path.join(audio + '.flac')
        reps = get_embeddings_audio(audio_path, model, args.device)
        for word, intervals in items.items():
            if word != 'transcript_audio':
                embeddings_list = []
                for start_frame, end_frame in intervals:
                    # Get averaged embedding from start frame to end frame
                    embeddings_per_layer = get_embedding_across_layers_audio(start_frame, end_frame, reps, args.model)
                    embeddings_as_lists = [emb.tolist() for emb in embeddings_per_layer]
                    embeddings_list.append(embeddings_as_lists)
                embeddings_audio[word] = embeddings_list  
        audio_name = audio.split('/')[-1]     
        output_dir = os.path.join(f'../experiments/{args.model}')
        output_file = os.path.join(output_dir, f'embeddings_words_{audio_name}.json')
        os.makedirs(output_dir, exist_ok=True)             
        save_json_with_embedding(output_file, audio_name, embeddings_audio)

if __name__ == '__main__':
    main()
