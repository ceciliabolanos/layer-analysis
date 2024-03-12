import torchaudio
import torch
import os
import json

def get_embedding_across_layers(start_frame, end_frame, audio_path, model, device):
    """
     Get the embedding across all layers related to some word or phone for a specific model.
     This is calculated as the mean between the embeddings from start_frame to end_frame included (from alignment)
    """
    waveform, sample_rate = torchaudio.load(audio_path)
    waveform = waveform.to(device)  # or 'cuda' if using GPU
    averaged_reps = []
    with torch.no_grad():
        reps = model(waveform)["hidden_states"]
    for i in range(1, len(reps)): 
        # Extract the relevant frames and calculate their mean
        averaged_rep = reps[i][0][start_frame:end_frame + 1].mean(dim=0)
        averaged_reps.append(averaged_rep)
    return averaged_reps    


def process_embeddings_for_word(word_items, audio_dir, model, device):
    summed_embeddings = None
    for item in word_items:
        start_frame, end_frame, audio_path = item
        embeddings = get_embedding_across_layers(start_frame, end_frame, os.path.join(audio_dir, audio_path), model, device)
        
        if summed_embeddings is None:
            summed_embeddings = [torch.zeros_like(emb) for emb in embeddings]
        
        for i, emb in enumerate(embeddings):
            summed_embeddings[i] += emb
    
    # Calculate the average embeddings across all items for the current word
    average_embeddings = [sum_emb / len(word_items) for sum_emb in summed_embeddings]

    return average_embeddings


def save_json_with_new_embedding(fname, word, embedding):
    # Load existing data
    try:
        with open(fname, "r") as f:
            data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        data = {}  # If file doesn't exist or is empty, start with an empty dict

    # Update with new embedding
    data[word] = embedding

    # Save back to file
    with open(fname, "w") as f:
        json.dump(data, f, indent=4)