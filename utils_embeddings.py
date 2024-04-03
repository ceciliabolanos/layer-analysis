import torchaudio
import torch
import json
import torch


def get_embeddings_nlp(encoded_input, model, device):
    """
     Get all the hidden_states for a nlp input
    """
    # Get hidden states from all layers
    encoded_input = {key: val.to(device) for key, val in encoded_input.items()}
    with torch.no_grad():
        output = model(**encoded_input)
        reps  = output.hidden_states
        # all_attentions_states = output.attentions
        
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

def get_embeddings_speech(audio_path, model, device):
    """
     Get all the hidden_states for a specific audio
    """
    waveform, sample_rate = torchaudio.load(audio_path)
    waveform = waveform.to(device)  # or 'cuda' if using GPU
    with torch.no_grad():
        reps = model(waveform)["hidden_states"]
  
    return reps 


def save_json_with_new_embedding(fname, audio, embeddings):
    # Load existing data
    try:
        with open(fname, "r") as f:
            data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        data = {}  # If file doesn't exist or is empty, start with an empty dict

    # Update with embeddings for each word inside an audio
    data[audio] = embeddings

    # Save back to file
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