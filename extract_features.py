import os
import json
import torch
import torchaudio
import pickle
import chromadb
import s3prl.hub as hub
from chroma_cumber import Chroma

"""
Info that I've learnt from sp3prl:
  reps = model(waveform)["hidden_states"]
  reps[0] output cnns
  reps[i] for i in 1 to len-1 -> hidden state for layer i
  reps[i][j] hidden state for layer i and audio j
  reps[i][j][k] hidden state for layer i and audio j and frame k (waveform can be more that such 1 audio)
"""

persistent_client = chromadb.PersistentClient()
collection = persistent_client.get_or_create_collection("audio_embeddings")

# Directory containing the audio files
audio_dir = './librispeech_data'

model = getattr(hub, 'wav2vec2')()  # build the Wav2Vec 2.0 model with pre-trained weights
device = 'cpu'  # or gpu
model = model.to(device)

def get_embedding_across_layers(start_frame, end_frame, audio_path):
    """
     Get the embedding across all layers related to some word or phone for a specific model.
     This is calculated as the mean between the embeddings from start_frame to end_frame included (from alignment)
    """
    waveform, sample_rate = torchaudio.load(audio_path)
    waveform = waveform.to('cpu')  # or 'cuda' if using GPU
    averaged_reps = []
    with torch.no_grad():
        reps = model(waveform)["hidden_states"]
    for i in range(1, len(reps)-1): 
        # Extract the relevant frames and calculate their mean
        averaged_rep = reps[i][0][start_frame:end_frame + 1].mean(dim=0)
        averaged_reps.append(averaged_rep)
    return averaged_reps    


def process_embeddings_for_word(word_items, word_key):
    summed_embeddings = None
    for item in word_items:
        start_frame, end_frame, audio_path = item
        print(os.path.join(audio_dir, audio_path))
        embeddings = get_embedding_across_layers(start_frame, end_frame, os.path.join(audio_dir, audio_path))
        
        if summed_embeddings is None:
            summed_embeddings = [torch.zeros_like(emb) for emb in embeddings]
        
        for i, emb in enumerate(embeddings):
            summed_embeddings[i] += emb
    
    # Calculate the average embeddings across all items for the current word
    average_embeddings = [sum_emb / len(word_items) for sum_emb in summed_embeddings]

    return average_embeddings

json_file_path = 'words1.json'

# Load the data from the JSON file
with open(json_file_path, 'r') as f:
    data = json.load(f)


# Process each word and save its averaged embeddings
for key, items in data.items():
    embeddings_per_layer = process_embeddings_for_word(items, key)
    embeddings_as_lists = [emb.tolist() for emb in embeddings_per_layer]
    
    for i, layer_embedding in enumerate(embeddings_as_lists):
        # Generate a unique identifier for this layer's embedding of the word
        unique_id = f"{key}_{i}"
        
        # Add the layer's embedding to the collection with the generated unique identifier
        try:
            collection.add(ids=[unique_id], embeddings=[layer_embedding])
        except Exception as e:
            print(f"Error adding layer {i} embeddings for {key} to the collection: {e}")
    print("SAVE")


with open('words_embeddings.pklz', 'wb') as file:
    # Step 2: Use Chroma.dump to save the compressed and pickled collection to the file
    Chroma.dump(collection, file)

print("Collection saved successfully.")











