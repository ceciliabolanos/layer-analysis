import os
import json
from utils_embeddings import get_embedding_across_layers, get_embeddings_nlp, save_json_with_new_embedding, match_word_to_tokens, unite_subword_tokens
import argparse
from transformers import BertTokenizerFast, BertModel

"""
Info that I've learned from huggingFace:
  all_hidden_states[i][0][j] Hidden states for layer i and token j. 
  all_attentions_states[i][0][j] Attentions weighs for layer i, multihead attention j. So this is the matrix tokens x tokens
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
    device = args.device
    model = model.to(device)
    chunk_size = 100
    data_chunks = [dict(list(data.items())[i:i + chunk_size]) for i in range(0, len(data), chunk_size)]

    for idx, chunk in enumerate(data_chunks):
        if idx >= 16:
            output_file = os.path.join('../experiments', f'embeddings_{args.model}_words_chunk{idx}.json')  # Update output file name with chunk identifier
            i = 0
            for audio, items in chunk.items():
                embeddings_audio = {}
                for key, input_text in items.items():
                    if key == 'transcript_audio':
                        encoded_input = tokenizer(input_text, return_tensors='pt')
                        tokens = tokenizer.tokenize(input_text)
                        word_ids = encoded_input.word_ids(batch_index=0)  
                        word_list = unite_subword_tokens(tokens)   
                        dict_match = match_word_to_tokens(word_ids, word_list) # match start token to end token for each word
                        reps = get_embeddings_nlp(encoded_input, model, device)
                        for word, intervals in dict_match.items():
                            embeddings_list = []
                            for sequence in intervals:
                                # Assume get_embedding_across_layers is a function that takes start, end frame, and reps to calculate embeddings
                                embeddings_per_layer = get_embedding_across_layers(sequence[0], sequence[-1], reps)
                                embeddings_as_lists = [emb.tolist() for emb in embeddings_per_layer]
                                embeddings_list.append(embeddings_as_lists)
                            embeddings_audio[word] = embeddings_list 
                print(f'terminé {i}') 
                i += 1           
                save_json_with_new_embedding(output_file, audio, embeddings_audio)
        print(f'terminé {idx}')


if __name__ == '__main__':
    main()

