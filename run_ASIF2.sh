# Busqueda de hiperparametros para Wav2vec y BERT

path_layer1="../experiments/layers/embeddings_layer6_wav2vec2.json"
path_layer2="../experiments/layers/embeddings_layer3_bert-base-uncased.json"
keys="words_in_order1.json"

# python3 experiments/experiment_ASIF.py --path_layer1 $path_layer1 --path_layer2 $path_layer2 --k 250 --p 1 --keys $keys
# python3 experiments/experiment_ASIF.py --path_layer1 $path_layer1 --path_layer2 $path_layer2 --k 500 --p 1 --keys $keys
# python3 experiments/experiment_ASIF.py --path_layer1 $path_layer1 --path_layer2 $path_layer2 --k 750 --p 1 --keys $keys
# python3 experiments/experiment_ASIF.py --path_layer1 $path_layer1 --path_layer2 $path_layer2 --k 250 --p 2 --keys $keys
# python3 experiments/experiment_ASIF.py --path_layer1 $path_layer1 --path_layer2 $path_layer2 --k 500 --p 2 --keys $keys
# python3 experiments/experiment_ASIF.py --path_layer1 $path_layer1 --path_layer2 $path_layer2 --k 750 --p 2 --keys $keys
# python3 experiments/experiment_ASIF.py --path_layer1 $path_layer1 --path_layer2 $path_layer2 --k 250 --p 3 --keys $keys
# python3 experiments/experiment_ASIF.py --path_layer1 $path_layer1 --path_layer2 $path_layer2 --k 500 --p 3 --keys $keys
# python3 experiments/experiment_ASIF.py --path_layer1 $path_layer1 --path_layer2 $path_layer2 --k 750 --p 3 --keys $keys
# python3 experiments/experiment_ASIF.py --path_layer1 $path_layer1 --path_layer2 $path_layer2 --k 250 --p 4 --keys $keys
# python3 experiments/experiment_ASIF.py --path_layer1 $path_layer1 --path_layer2 $path_layer2 --k 500 --p 4 --keys $keys
# python3 experiments/experiment_ASIF.py --path_layer1 $path_layer1 --path_layer2 $path_layer2 --k 750 --p 4 --keys $keys
# python3 experiments/experiment_ASIF.py --path_layer1 $path_layer1 --path_layer2 $path_layer2 --k 250 --p 5 --keys $keys
# python3 experiments/experiment_ASIF.py --path_layer1 $path_layer1 --path_layer2 $path_layer2 --k 250 --p 6 --keys $keys
# python3 experiments/experiment_ASIF.py --path_layer1 $path_layer1 --path_layer2 $path_layer2 --k 250 --p 7 --keys $keys
# python3 experiments/experiment_ASIF.py --path_layer1 $path_layer1 --path_layer2 $path_layer2 --k 250 --p 8 --keys $keys
# python3 experiments/experiment_ASIF.py --path_layer1 $path_layer1 --path_layer2 $path_layer2 --k 250 --p 9 --keys $keys
# python3 experiments/experiment_ASIF.py --path_layer1 $path_layer1 --path_layer2 $path_layer2 --k 200 --p 4 --keys $keys
# python3 experiments/experiment_ASIF.py --path_layer1 $path_layer1 --path_layer2 $path_layer2 --k 150 --p 4 --keys $keys
# python3 experiments/experiment_ASIF.py --path_layer1 $path_layer1 --path_layer2 $path_layer2 --k 100 --p 4 --keys $keys
# python3 experiments/experiment_ASIF.py --path_layer1 $path_layer1 --path_layer2 $path_layer2 --k 50 --p 4 --keys $keys
# python3 experiments/experiment_ASIF.py --path_layer1 $path_layer1 --path_layer2 $path_layer2 --k 10 --p 4 --keys $keys
# python3 experiments/experiment_ASIF.py --path_layer1 $path_layer1 --path_layer2 $path_layer2 --k 100 --p 3 --keys $keys
# python3 experiments/experiment_ASIF.py --path_layer1 $path_layer1 --path_layer2 $path_layer2 --k 100 --p 2 --keys $keys
# python3 experiments/experiment_ASIF.py --path_layer1 $path_layer1 --path_layer2 $path_layer2 --k 100 --p 1 --keys $keys


path_layer3="../experiments/layers/embeddings_layer9_encodecmae_base.json"
path_layer4="../experiments/layers/embeddings_layer0_glove.json"

# python3 experiments/experiment_ASIF.py --path_layer1 $path_layer3 --path_layer2 $path_layer4 --k 250 --p 1 --keys $keys
# python3 experiments/experiment_ASIF.py --path_layer1 $path_layer3 --path_layer2 $path_layer4 --k 500 --p 1 --keys $keys
# python3 experiments/experiment_ASIF.py --path_layer1 $path_layer3 --path_layer2 $path_layer4 --k 750 --p 1 --keys $keys
# python3 experiments/experiment_ASIF.py --path_layer1 $path_layer3 --path_layer2 $path_layer4 --k 250 --p 2 --keys $keys
# python3 experiments/experiment_ASIF.py --path_layer1 $path_layer3 --path_layer2 $path_layer4 --k 500 --p 2 --keys $keys
# python3 experiments/experiment_ASIF.py --path_layer1 $path_layer3 --path_layer2 $path_layer4 --k 750 --p 2 --keys $keys
# python3 experiments/experiment_ASIF.py --path_layer1 $path_layer3 --path_layer2 $path_layer4 --k 250 --p 3 --keys $keys
# python3 experiments/experiment_ASIF.py --path_layer1 $path_layer3 --path_layer2 $path_layer4 --k 500 --p 3 --keys $keys
# python3 experiments/experiment_ASIF.py --path_layer1 $path_layer3 --path_layer2 $path_layer4 --k 750 --p 3 --keys $keys
# python3 experiments/experiment_ASIF.py --path_layer1 $path_layer3 --path_layer2 $path_layer4 --k 250 --p 4 --keys $keys
# python3 experiments/experiment_ASIF.py --path_layer1 $path_layer3 --path_layer2 $path_layer4 --k 500 --p 4 --keys $keys
# python3 experiments/experiment_ASIF.py --path_layer1 $path_layer3 --path_layer2 $path_layer4 --k 750 --p 4 --keys $keys
# python3 experiments/experiment_ASIF.py --path_layer1 $path_layer3 --path_layer2 $path_layer4 --k 250 --p 5 --keys $keys
# python3 experiments/experiment_ASIF.py --path_layer1 $path_layer3 --path_layer2 $path_layer4 --k 250 --p 6 --keys $keys
# python3 experiments/experiment_ASIF.py --path_layer1 $path_layer3 --path_layer2 $path_layer4 --k 250 --p 7 --keys $keys
# python3 experiments/experiment_ASIF.py --path_layer1 $path_layer3 --path_layer2 $path_layer4 --k 250 --p 8 --keys $keys
# python3 experiments/experiment_ASIF.py --path_layer1 $path_layer3 --path_layer2 $path_layer4 --k 250 --p 9 --keys $keys
# python3 experiments/experiment_ASIF.py --path_layer1 $path_layer3 --path_layer2 $path_layer4 --k 200 --p 4 --keys $keys
# python3 experiments/experiment_ASIF.py --path_layer1 $path_layer3 --path_layer2 $path_layer4 --k 150 --p 4 --keys $keys
# python3 experiments/experiment_ASIF.py --path_layer1 $path_layer3 --path_layer2 $path_layer4 --k 100 --p 4 --keys $keys
# python3 experiments/experiment_ASIF.py --path_layer1 $path_layer3 --path_layer2 $path_layer4 --k 50 --p 4 --keys $keys
# python3 experiments/experiment_ASIF.py --path_layer1 $path_layer3 --path_layer2 $path_layer4 --k 10 --p 4 --keys $keys
# python3 experiments/experiment_ASIF.py --path_layer1 $path_layer3 --path_layer2 $path_layer4 --k 100 --p 3 --keys $keys
# python3 experiments/experiment_ASIF.py --path_layer1 $path_layer3 --path_layer2 $path_layer4 --k 100 --p 2 --keys $keys
# python3 experiments/experiment_ASIF.py --path_layer1 $path_layer3 --path_layer2 $path_layer4 --k 100 --p 1 --keys $keys


# python3 experiments/experiment_ASIF.py --path_layer1 $path_layer1 --path_layer2 $path_layer4 --k 200 --p 4 --keys $keys
# python3 experiments/experiment_ASIF.py --path_layer1 $path_layer1 --path_layer2 $path_layer4 --k 150 --p 4 --keys $keys
# python3 experiments/experiment_ASIF.py --path_layer1 $path_layer1 --path_layer2 $path_layer4 --k 100 --p 4 --keys $keys
# python3 experiments/experiment_ASIF.py --path_layer1 $path_layer1 --path_layer2 $path_layer4 --k 50 --p 4 --keys $keys
# python3 experiments/experiment_ASIF.py --path_layer1 $path_layer1 --path_layer2 $path_layer4 --k 10 --p 4 --keys $keys
# python3 experiments/experiment_ASIF.py --path_layer1 $path_layer1 --path_layer2 $path_layer4 --k 100 --p 3 --keys $keys
# python3 experiments/experiment_ASIF.py --path_layer1 $path_layer1 --path_layer2 $path_layer4 --k 100 --p 2 --keys $keys
# python3 experiments/experiment_ASIF.py --path_layer1 $path_layer1 --path_layer2 $path_layer4 --k 100 --p 1 --keys $keys

# path_layer5="../experiments/layers/embeddings_layer3_BEATs_iter3.json"
# python3 experiments/experiment_ASIF.py --path_layer1 $path_layer5 --path_layer2 $path_layer4 --k 250 --p 1 --keys $keys
# python3 experiments/experiment_ASIF.py --path_layer1 $path_layer5 --path_layer2 $path_layer4 --k 500 --p 1 --keys $keys
# python3 experiments/experiment_ASIF.py --path_layer1 $path_layer5 --path_layer2 $path_layer4 --k 750 --p 1 --keys $keys
# python3 experiments/experiment_ASIF.py --path_layer1 $path_layer5 --path_layer2 $path_layer4 --k 250 --p 2 --keys $keys
# python3 experiments/experiment_ASIF.py --path_layer1 $path_layer5 --path_layer2 $path_layer4 --k 500 --p 2 --keys $keys
# python3 experiments/experiment_ASIF.py --path_layer1 $path_layer5 --path_layer2 $path_layer4 --k 750 --p 2 --keys $keys
# python3 experiments/experiment_ASIF.py --path_layer1 $path_layer5 --path_layer2 $path_layer4 --k 250 --p 3 --keys $keys
# python3 experiments/experiment_ASIF.py --path_layer1 $path_layer5 --path_layer2 $path_layer4 --k 500 --p 3 --keys $keys
# python3 experiments/experiment_ASIF.py --path_layer1 $path_layer5 --path_layer2 $path_layer4 --k 750 --p 3 --keys $keys
# python3 experiments/experiment_ASIF.py --path_layer1 $path_layer5 --path_layer2 $path_layer4 --k 250 --p 4 --keys $keys
# python3 experiments/experiment_ASIF.py --path_layer1 $path_layer5 --path_layer2 $path_layer4 --k 500 --p 4 --keys $keys
# python3 experiments/experiment_ASIF.py --path_layer1 $path_layer5 --path_layer2 $path_layer4 --k 750 --p 4 --keys $keys
# python3 experiments/experiment_ASIF.py --path_layer1 $path_layer5 --path_layer2 $path_layer4 --k 250 --p 5 --keys $keys
# python3 experiments/experiment_ASIF.py --path_layer1 $path_layer5 --path_layer2 $path_layer4 --k 250 --p 6 --keys $keys
# python3 experiments/experiment_ASIF.py --path_layer1 $path_layer5 --path_layer2 $path_layer4 --k 250 --p 7 --keys $keys
# python3 experiments/experiment_ASIF.py --path_layer1 $path_layer5 --path_layer2 $path_layer4 --k 250 --p 8 --keys $keys
# python3 experiments/experiment_ASIF.py --path_layer1 $path_layer5 --path_layer2 $path_layer4 --k 250 --p 9 --keys $keys
# python3 experiments/experiment_ASIF.py --path_layer1 $path_layer5 --path_layer2 $path_layer4 --k 200 --p 4 --keys $keys
# python3 experiments/experiment_ASIF.py --path_layer1 $path_layer5 --path_layer2 $path_layer4 --k 150 --p 4 --keys $keys
# python3 experiments/experiment_ASIF.py --path_layer1 $path_layer5 --path_layer2 $path_layer4 --k 100 --p 4 --keys $keys
# python3 experiments/experiment_ASIF.py --path_layer1 $path_layer5 --path_layer2 $path_layer4 --k 50 --p 4 --keys $keys
# python3 experiments/experiment_ASIF.py --path_layer1 $path_layer5 --path_layer2 $path_layer4 --k 10 --p 4 --keys $keys
# python3 experiments/experiment_ASIF.py --path_layer1 $path_layer5 --path_layer2 $path_layer4 --k 100 --p 3 --keys $keys
# python3 experiments/experiment_ASIF.py --path_layer1 $path_layer5 --path_layer2 $path_layer4 --k 100 --p 2 --keys $keys
# python3 experiments/experiment_ASIF.py --path_layer1 $path_layer5 --path_layer2 $path_layer4 --k 100 --p 1 --keys $keys

# path_layer6="../experiments/layers/embeddings_layer10_bert-base-uncased.json"
# path_layer7="../experiments/layers/embeddings_layer5_BEATs_iter3.json"

# python3 experiments/experiment_ASIF.py --path_layer1 $path_layer7 --path_layer2 $path_layer6 --k 250 --p 1 --keys $keys
# python3 experiments/experiment_ASIF.py --path_layer1 $path_layer7 --path_layer2 $path_layer6 --k 500 --p 1 --keys $keys
# python3 experiments/experiment_ASIF.py --path_layer1 $path_layer7 --path_layer2 $path_layer6 --k 750 --p 1 --keys $keys
# python3 experiments/experiment_ASIF.py --path_layer1 $path_layer7 --path_layer2 $path_layer6 --k 250 --p 2 --keys $keys
# python3 experiments/experiment_ASIF.py --path_layer1 $path_layer7 --path_layer2 $path_layer6 --k 500 --p 2 --keys $keys
# python3 experiments/experiment_ASIF.py --path_layer1 $path_layer7 --path_layer2 $path_layer6 --k 750 --p 2 --keys $keys
# python3 experiments/experiment_ASIF.py --path_layer1 $path_layer7 --path_layer2 $path_layer6 --k 250 --p 3 --keys $keys
# python3 experiments/experiment_ASIF.py --path_layer1 $path_layer7 --path_layer2 $path_layer6 --k 500 --p 3 --keys $keys
# python3 experiments/experiment_ASIF.py --path_layer1 $path_layer7 --path_layer2 $path_layer6 --k 750 --p 3 --keys $keys
# python3 experiments/experiment_ASIF.py --path_layer1 $path_layer7 --path_layer2 $path_layer6 --k 250 --p 4 --keys $keys
# python3 experiments/experiment_ASIF.py --path_layer1 $path_layer7 --path_layer2 $path_layer6 --k 500 --p 4 --keys $keys
# python3 experiments/experiment_ASIF.py --path_layer1 $path_layer7 --path_layer2 $path_layer6 --k 750 --p 4 --keys $keys
# python3 experiments/experiment_ASIF.py --path_layer1 $path_layer7 --path_layer2 $path_layer6 --k 250 --p 5 --keys $keys
# python3 experiments/experiment_ASIF.py --path_layer1 $path_layer7 --path_layer2 $path_layer6 --k 250 --p 6 --keys $keys
# python3 experiments/experiment_ASIF.py --path_layer1 $path_layer7 --path_layer2 $path_layer6 --k 250 --p 7 --keys $keys
# python3 experiments/experiment_ASIF.py --path_layer1 $path_layer7 --path_layer2 $path_layer6 --k 250 --p 8 --keys $keys
# python3 experiments/experiment_ASIF.py --path_layer1 $path_layer7 --path_layer2 $path_layer6 --k 250 --p 9 --keys $keys
# python3 experiments/experiment_ASIF.py --path_layer1 $path_layer7 --path_layer2 $path_layer6 --k 200 --p 4 --keys $keys
# python3 experiments/experiment_ASIF.py --path_layer1 $path_layer7 --path_layer2 $path_layer6 --k 150 --p 4 --keys $keys
# python3 experiments/experiment_ASIF.py --path_layer1 $path_layer7 --path_layer2 $path_layer6 --k 100 --p 4 --keys $keys
# python3 experiments/experiment_ASIF.py --path_layer1 $path_layer7 --path_layer2 $path_layer6 --k 50 --p 4 --keys $keys
# python3 experiments/experiment_ASIF.py --path_layer1 $path_layer7 --path_layer2 $path_layer6 --k 10 --p 4 --keys $keys
# python3 experiments/experiment_ASIF.py --path_layer1 $path_layer7 --path_layer2 $path_layer6 --k 100 --p 3 --keys $keys
# python3 experiments/experiment_ASIF.py --path_layer1 $path_layer7 --path_layer2 $path_layer6 --k 100 --p 2 --keys $keys
# python3 experiments/experiment_ASIF.py --path_layer1 $path_layer7 --path_layer2 $path_layer6 --k 100 --p 1 --keys $keys



path_layer8="../experiments/layers/embeddings_layer9_encodecmae_mel256-ec-base.json"
#path_layer9="../experiments/layers/embeddings_layer0_bert-base-uncased.json"
path_layer9="../experiments/layers/embeddings_layer0_glove.json"

python3 experiments/experiment_ASIF.py --path_layer1 $path_layer8 --path_layer2 $path_layer9 --k 250 --p 1 --keys $keys
python3 experiments/experiment_ASIF.py --path_layer1 $path_layer8 --path_layer2 $path_layer9 --k 500 --p 1 --keys $keys
python3 experiments/experiment_ASIF.py --path_layer1 $path_layer8 --path_layer2 $path_layer9 --k 750 --p 1 --keys $keys
python3 experiments/experiment_ASIF.py --path_layer1 $path_layer8 --path_layer2 $path_layer9 --k 250 --p 2 --keys $keys
python3 experiments/experiment_ASIF.py --path_layer1 $path_layer8 --path_layer2 $path_layer9 --k 500 --p 2 --keys $keys
python3 experiments/experiment_ASIF.py --path_layer1 $path_layer8 --path_layer2 $path_layer9 --k 750 --p 2 --keys $keys
python3 experiments/experiment_ASIF.py --path_layer1 $path_layer8 --path_layer2 $path_layer9 --k 250 --p 3 --keys $keys
python3 experiments/experiment_ASIF.py --path_layer1 $path_layer8 --path_layer2 $path_layer9 --k 500 --p 3 --keys $keys
python3 experiments/experiment_ASIF.py --path_layer1 $path_layer8 --path_layer2 $path_layer9 --k 750 --p 3 --keys $keys
python3 experiments/experiment_ASIF.py --path_layer1 $path_layer8 --path_layer2 $path_layer9 --k 250 --p 4 --keys $keys
python3 experiments/experiment_ASIF.py --path_layer1 $path_layer8 --path_layer2 $path_layer9 --k 500 --p 4 --keys $keys
python3 experiments/experiment_ASIF.py --path_layer1 $path_layer8 --path_layer2 $path_layer9 --k 750 --p 4 --keys $keys
python3 experiments/experiment_ASIF.py --path_layer1 $path_layer8 --path_layer2 $path_layer9 --k 250 --p 5 --keys $keys
python3 experiments/experiment_ASIF.py --path_layer1 $path_layer8 --path_layer2 $path_layer9 --k 250 --p 6 --keys $keys
python3 experiments/experiment_ASIF.py --path_layer1 $path_layer8 --path_layer2 $path_layer9 --k 250 --p 7 --keys $keys
python3 experiments/experiment_ASIF.py --path_layer1 $path_layer8 --path_layer2 $path_layer9 --k 250 --p 8 --keys $keys
python3 experiments/experiment_ASIF.py --path_layer1 $path_layer8 --path_layer2 $path_layer9 --k 250 --p 9 --keys $keys
python3 experiments/experiment_ASIF.py --path_layer1 $path_layer8 --path_layer2 $path_layer9 --k 200 --p 4 --keys $keys
python3 experiments/experiment_ASIF.py --path_layer1 $path_layer8 --path_layer2 $path_layer9 --k 150 --p 4 --keys $keys
python3 experiments/experiment_ASIF.py --path_layer1 $path_layer8 --path_layer2 $path_layer9 --k 100 --p 4 --keys $keys
python3 experiments/experiment_ASIF.py --path_layer1 $path_layer8 --path_layer2 $path_layer9 --k 50 --p 4 --keys $keys
python3 experiments/experiment_ASIF.py --path_layer1 $path_layer8 --path_layer2 $path_layer9 --k 10 --p 4 --keys $keys
python3 experiments/experiment_ASIF.py --path_layer1 $path_layer8 --path_layer2 $path_layer9 --k 100 --p 3 --keys $keys
python3 experiments/experiment_ASIF.py --path_layer1 $path_layer8 --path_layer2 $path_layer9 --k 100 --p 2 --keys $keys
python3 experiments/experiment_ASIF.py --path_layer1 $path_layer8 --path_layer2 $path_layer9 --k 100 --p 1 --keys $keys