import os
import json
from utils_embeddings import get_embedding_across_layers, get_embeddings_speech, save_json_with_new_embedding
import s3prl.hub as hub
import argparse

"""
Info that I've learned from sp3prl:
  reps = model(waveform)["hidden_states"]
  reps[0] output cnns
  reps[i] for i in 1 to len-1 -> hidden state for layer i
  reps[i][j] hidden state for layer i and audio j
  reps[i][j][k] hidden state for layer i and audio j and frame k (waveform can be more that such 1 audio)
"""

def main():
    
    parser = argparse.ArgumentParser(description='')

    parser.add_argument('--model', type=str, required=False, help='', default="wav2vec") 
    parser.add_argument('--path', type=str, required=False, help='path to words file', default='words_and_phrase.json') 
    parser.add_argument('--device', type=str, required=False, default='cpu')      
    args = parser.parse_args()

    with open(args.path, 'r') as f:
        data = json.load(f)

    model = getattr(hub, args.model)()  # build the Wav2Vec 2.0 model with pre-trained weights
    device = args.device # or gpu
    model = model.to(device)

    output_file = f'embedding_{args.model}_words.json'

    # Process each word and save its averaged embeddings
    for audio, items in data.items():
        embeddings_audio = {}
        audio_path = os.path.join(audio + '.flac')
        reps = get_embeddings_speech(audio_path, model, device)
        for word, intervals in items.items():
            if word != 'transcript':
                embeddings_list = []
                for start_frame, end_frame in intervals:
                    # Assume get_embedding_across_layers is a function that takes start, end frame, and reps to calculate embeddings
                    embeddings_per_layer = get_embedding_across_layers(start_frame, end_frame, reps)
                    embeddings_as_lists = [emb.tolist() for emb in embeddings_per_layer]
                    embeddings_list.append(embeddings_as_lists)
                embeddings_audio[word] = embeddings_list    
        save_json_with_new_embedding(output_file, audio, embeddings_audio)
        print(f'termine{audio}')


if __name__ == '__main__':
    main()

