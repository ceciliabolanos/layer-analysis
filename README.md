# Requirements
sudo apt-get update
sudo apt-get install sox libsox-fmt-all
pip install s3prl[all]
pip install torchaudio
pip install chardet
pip install PySoundFile librosa

Esto esta en pausa xq ahora lo obtenemos de otra forma
To get the alignments: 

python3 read_alignments_librispeech.py 
python3 extract_features.py


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

1. Obtain NLP embeddings by executing nlp_embedding.py with the model, input path, and device as arguments. This script outputs a JSON file with NLP embeddings for each word in each audio, averaged over all tokens of the word.

- Command: 
     ```
     python3 nlp_embedding.py --model bert-base-uncased --path 'audio_alignments.json' --device 'cuda'
     ```