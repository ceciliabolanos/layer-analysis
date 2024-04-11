import os
import json
from utils_embeddings import get_embedding_across_layers, get_embeddings_nlp, save_json_with_embedding, match_word_to_tokens, unite_subword_tokens
import argparse
from transformers import BertTokenizerFast, BertModel
from tqdm import tqdm

"""
Info that I've learned from huggingFace:
  all_hidden_states[i][0][j] Hidden states for layer i and token j. 
  all_attentions_states[i][0][j] Attentions weighs for layer i, multihead attention j. So this is the matrix tokens x tokens
  Chequear si la ultima capa de nlp y de audio es igual:
https://huggingface.co/docs/transformers/en/main_classes/output#transformers.modeling_outputs.BaseModelOutputWithPooling.hidden_states
When passing output_hidden_states=True you may expect the outputs.hidden_states[-1] to match outputs.last_hidden_states exactly. However, this is not always the case. Some models apply normalization or subsequent process to the last hidden state when it’s returned.
"""

def main():
    
    parser = argparse.ArgumentParser(description='')

    parser.add_argument('--model', type=str, required=False, help='', default="bert-base-uncased") 
    parser.add_argument('--path', type=str, required=False, help='path to words file', default='audio_alignments.json') 
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
        output_file = os.path.join(f'../experiments/{args.model}', f'embeddings_words_{audio_name}.json')         
        save_json_with_embedding(output_file, audio_name, embeddings_audio)
        

if __name__ == '__main__':
    main()

