import os
import json
from utils_embeddings import get_embeddings_glove, save_json_with_embedding, load_glove_embeddings
import argparse
from tqdm import tqdm


def main():
    
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--path', type=str, required=False, help='path to words file', default='alignments/audio_alignments.json') 
    args = parser.parse_args()

    with open(args.path, 'r') as f:
        data = json.load(f)
   
    glove_file_path = '../datasets/glove.42B.300d.txt'

    glove_embeddings = load_glove_embeddings(glove_file_path)

    for audio, items in tqdm(data.items(), desc="Processing audio files"):
        for key, input_text in items.items():
            if key == 'transcript_audio':
                embeddings_audio = get_embeddings_glove(input_text.lower(), glove_embeddings)
                audio_name = audio.split('/')[-1]          
                output_dir = os.path.join('../experiments/glove')
                output_file = os.path.join(output_dir, f'embeddings_words_{audio_name}.json')
                os.makedirs(output_dir, exist_ok=True)         
                save_json_with_embedding(output_file, audio_name, embeddings_audio)

if __name__ == '__main__':
    main()

