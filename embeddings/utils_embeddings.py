import torchaudio
import torch
import json
import torch
import numpy as np


# Text models

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

def unite_subword_tokens(tokens):
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

# Speech models

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

def get_embeddings_speech(audio_path, model, device):
    """
     Get all the hidden_states for a specific audio
    """
    waveform, sample_rate = torchaudio.load(audio_path)
    waveform = waveform.to(device) 
    with torch.no_grad():
        reps = model(waveform)["hidden_states"]
  
    return reps 

# Audio models 

def get_embeddings_audio(audio_path, model, device):
    """
     Get all the hidden_states for a specific audio
    """
    waveform, sample_rate = torchaudio.load(audio_path)
    waveform = waveform.to(device) 
    with torch.no_grad():
        reps = model.extract_activations_from_array(waveform) 
  
    return reps  

def get_embedding_across_layers_audio(start_frame, end_frame, reps, model):
    """
     Get the embedding across all layers related to some word or phone for a specific model.
     This is calculated as the mean between the embeddings from start_frame to end_frame included (from alignment)
    """
    averaged_reps = []
    if model == 'encodecmae_base':
        for i in range(0, len(reps)): 
            # Extract the relevant frames and calculate their mean
            averaged_rep = reps[i][start_frame:end_frame + 1].mean(axis=0)
            averaged_reps.append(averaged_rep)

    if model == 'BEATs_iter3':
        patch_start = (start_frame // 16) * 8
        patch_end = (start_frame // 16) * 8
        if len(reps[0]) < patch_start:
            return averaged_reps    
        for i in range(1, len(reps)): 
            if len(reps[i]) > patch_end:
                averaged_rep = reps[i][patch_start:].mean(axis=0) 
            else:   
                averaged_rep = reps[i][patch_start:patch_end + 8].mean(axis=0)
            averaged_reps.append(averaged_rep)
    return averaged_reps   

# Others

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

def linear_CKA(X, Y):
    X_centered = X - X.mean(axis=0)
    Y_centered = Y - Y.mean(axis=0)
    hsic = np.linalg.norm(X_centered.T @ Y_centered, 'fro') ** 2
    var1 = np.linalg.norm(X_centered.T @ X_centered, 'fro')
    var2 = np.linalg.norm(Y_centered.T @ Y_centered, 'fro')

    return hsic / (var1 * var2)
