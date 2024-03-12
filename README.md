sudo apt-get update
sudo apt-get install sox libsox-fmt-all
pip install s3prl
pip install torchaudio
pip install PySoundFile librosa


To get the alignments: 

python3 read_alignments_librispeech.py 
python3 extract_features.py