import os
import json
from utils_embeddings import get_embedding_across_layers, get_embeddings_nlp, save_json_with_embedding, match_word_to_tokens, unite_subword_tokens
import argparse
from transformers import BertTokenizerFast, BertModel
from tqdm import tqdm
import numpy as np
import torch


def main():
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    parser = argparse.ArgumentParser(description='')

    parser.add_argument('--model', type=str, required=False, help='', default="bert-base-uncased") 
    parser.add_argument('--path', type=str, required=False, help='path to words file', default='alignments/audio_alignments.json') 
    parser.add_argument('--device', type=str, required=False, default='cuda')      
    args = parser.parse_args()

    with open(args.path, 'r') as f:
        data = json.load(f)

    # Load the fast tokenizer
    tokenizer = BertTokenizerFast.from_pretrained(args.model)
    model = BertModel.from_pretrained(args.model, output_hidden_states=True, output_attentions=True)
    model = model.to(args.device)
    
    
    for audio, items in tqdm(data.items(), desc="Processing audio files"):
        embeddings_audio = {}
        for key, input_text in items.items():
            if key == 'transcript_audio':
                encoded_input = tokenizer(input_text, return_tensors='pt')
                tokens = tokenizer.tokenize(input_text)
                word_ids = encoded_input.word_ids(batch_index=0)  
                word_list = unite_subword_tokens(tokens)   
                dict_match = match_word_to_tokens(word_ids, word_list) # match start token to end token for each word
                reps = get_embeddings_nlp(encoded_input, model, args.device)
                for word, intervals in dict_match.items():
                    embeddings_list = []
                    for sequence in intervals:
                        embeddings_per_layer = get_embedding_across_layers(sequence[0], sequence[-1], reps)
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

