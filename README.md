# Requirements
sudo apt-get update
sudo apt-get install sox libsox-fmt-all
pip install s3prl[all]
pip install torchaudio
pip install chardet
pip install PySoundFile librosa



# Embeddings

Step by step instructions:

1. Execute the script for generating alignments using the command python alignments_and_trans.py with appropriate arguments for frame length, stride, sample rate, and alignments path. This will produce a JSON file containing the word alignments for each audio file. 

 - Command: 
     ```
     python3 alignments_and_trans.py --frame_length 25 --stride 20 --sample_rate 16000 --alignments '../datasets/alignments'
     ```

1. Generate speech embeddings by running speech_embedding.py with parameters specifying the model, input path, and computation device. This will output a JSON file containing the speech embeddings for each word in each audio, averaged over all frames where the word occurs.

- Command: 
     ```
     python3 speech_embedding.py --model wav2vec2 --path 'audio_alignments.json' --device 'cuda'
     ```

1. Obtain Glove embeddings by executing glove_embedding.py with input path as argument. This script outputs a JSON file with Glove embeddings for each word in each audio.

- Command: 
     ```
     python3 glove_embedding.py --path 'audio_alignments.json'
     ```

1. Obtain NLP embeddings by executing nlp_embedding.py with the model, input path, and device as arguments. This script outputs a JSON file with NLP embeddings for each word in each audio, averaged over all tokens of the word.

- Command: 
     ```
     python3 nlp_embedding.py --model bert-base-uncased --path 'audio_alignments.json' --device 'cuda'
     ```

 Once we have the embedding for each word in each audio we need to compute the matrix of all the embeddings for a specific audio. This is because we want to compare how similare are these representations when we change the layer or the model comparison. For doing that we can run   

 Command: 
     ```
     python3 embeddings_matrix.py
     ```  

 With this ready, we can play a little with some experiments like CKA.

  Command: 
     ```
     python3 experiment_CKA.py --model1 'wav2vec2' --layer1 24 --model2 'bert-base-uncased' --layer2 12 
     ```  
  This generates a matrix with the linear CKA score for each pair of layers.
