import torchaudio
import torch
import json
import torch
import numpy as np

def get_embeddings_nlp(encoded_input, model, device):
    """
     Get all the hidden_states (all layers) for a nlp input
    """
    encoded_input = {key: val.to(device) for key, val in encoded_input.items()}
    with torch.no_grad():
        output = model(**encoded_input)
        reps  = output.hidden_states
        # all_attentions_states = output.attentions    
    return reps 

def get_embeddings_speech(audio_path, model, device):
    """
     Get all the hidden_states for a specific audio
    """
    waveform, sample_rate = torchaudio.load(audio_path)
    waveform = waveform.to(device) 
    with torch.no_grad():
        reps = model(waveform)["hidden_states"]
  
    return reps 

def get_embedding_across_layers(start_frame, end_frame, reps):
    """
     Get the embedding across all layers related to some word or phone for a specific model.
     This is calculated as the mean between the embeddings from start_frame to end_frame included (from alignment)
    """
    averaged_reps = []
    for i in range(1, len(reps)): 
        # Extract the relevant frames and calculate their mean
        averaged_rep = reps[i][0][start_frame:end_frame + 1].mean(dim=0)
        averaged_reps.append(averaged_rep)
    return averaged_reps    


def save_json_with_embedding(fname, audio, embeddings):
    """Save Json with the embedding of each word/phone of an audio"""
    try:
        with open(fname, "r") as f:
            data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        data = {}  

    data[audio] = embeddings

    with open(fname, "w") as f:
        json.dump(data, f, indent=4)


def unite_subword_tokens(tokens):
    # Create an empty list to store the united tokens
    united_tokens = []
    for token in tokens:
        if token.startswith("##"):
            # If it's a subword, we remove the hashes and concatenate with the previous token
            united_tokens[-1] += token[2:]
        else:
            # Otherwise, we just add the token to the list
            united_tokens.append(token)
    return united_tokens        


def match_word_to_tokens(word_ids, word_list):
    match_word = {}
    for i in range(len(word_ids)):
        if word_ids[i] != None:
            word = word_list[word_ids[i]] 
            if word not in match_word:
                match_word[word] = [[i]]
            else:  
                if match_word[word][-1][-1] == (i-1):
                    match_word[word][-1].append(i)
                else:
                    match_word[word].append([i])            
    return match_word 

def load_glove_embeddings(file_path):
    embeddings_dict = {}
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            vector = np.asarray(values[1:], "float32")
            embeddings_dict[word] = vector
    return embeddings_dict



def get_embeddings_glove(input_text, glove_embeddings):
    
    """Retrieve a dictionary with the embeddings for each word in input_text"""

    words = input_text.split()
    embeddings = {}
    for word in words:
        embedding = glove_embeddings.get(word)
        if embedding is not None:
            # Convert numpy array to list and create 12 separate copies to compare with 12 layers of transformers
            embeddings[word] = [embedding.tolist() for _ in range(12)]

    return embeddings   