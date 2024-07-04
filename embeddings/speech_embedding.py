import os
import json
from utils_embeddings import get_embedding_across_layers, get_embeddings_speech, save_json_with_embedding
import s3prl.hub as hub
import argparse
from tqdm import tqdm
import numpy as np
import torch


def main():

    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
        
    parser = argparse.ArgumentParser(description='')

    parser.add_argument('--model', type=str, required=False, help='', default="wav2vec2") 
    parser.add_argument('--path', type=str, required=False, help='path to words file', default='alignments/audio_alignments.json') 
    parser.add_argument('--device', type=str, required=False, default='cuda')      
    args = parser.parse_args()

    with open(args.path, 'r') as f:
        data = json.load(f)

    model = getattr(hub, args.model)() 
    model = model.to(args.device)

    # For every audio, process each word and save its averaged embeddings   
    for audio, items in tqdm(data.items(), desc="Processing audio files"):
        embeddings_audio = {}
        audio_path = os.path.join(audio + '.flac')
        reps = get_embeddings_speech(audio_path, model, args.device)
        for word, intervals in items.items():
            if word != 'transcript_audio':
                embeddings_list = []
                for start_frame, end_frame in intervals:
                    # Get averaged embedding from start frame to end frame
                    embeddings_per_layer = get_embedding_across_layers(start_frame, end_frame, reps)
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

