# Script used to run all ASIF experiments

# Experiment1.2: Can ASIF be equivalent to CKA metric?. Saved in results/experiment1.2_ASIF_CKA


path_layer1="../experiments/layers/embeddings_layer6_wav2vec2.json"
path_layer2="../experiments/layers/embeddings_layer0_glove.json"
keys="words_in_order1.json"

# python3 experiments/experiment_ASIF.py --path_layer1 $path_layer1 --path_layer2 $path_layer2 --k 250 --p 1 --keys $keys
# python3 experiments/experiment_ASIF.py --path_layer1 $path_layer1 --path_layer2 $path_layer2 --k 500 --p 1 --keys $keys
# python3 experiments/experiment_ASIF.py --path_layer1 $path_layer1 --path_layer2 $path_layer2 --k 750 --p 1 --keys $keys
# python3 experiments/experiment_ASIF.py --path_layer1 $path_layer1 --path_layer2 $path_layer2 --k 1000 --p 1 --keys $keys
# python3 experiments/experiment_ASIF.py --path_layer1 $path_layer1 --path_layer2 $path_layer2 --k 1250 --p 1 --keys $keys
# python3 experiments/experiment_ASIF.py --path_layer1 $path_layer1 --path_layer2 $path_layer2 --k 1500 --p 1 --keys $keys
# python3 experiments/experiment_ASIF.py --path_layer1 $path_layer1 --path_layer2 $path_layer2 --k 1750 --p 1 --keys $keys
# python3 experiments/experiment_ASIF.py --path_layer1 $path_layer1 --path_layer2 $path_layer2 --k 2000 --p 1 --keys $keys
# python3 experiments/experiment_ASIF.py --path_layer1 $path_layer1 --path_layer2 $path_layer2 --k 250 --p 2 --keys $keys
# python3 experiments/experiment_ASIF.py --path_layer1 $path_layer1 --path_layer2 $path_layer2 --k 500 --p 2 --keys $keys
# python3 experiments/experiment_ASIF.py --path_layer1 $path_layer1 --path_layer2 $path_layer2 --k 750 --p 2 --keys $keys
# python3 experiments/experiment_ASIF.py --path_layer1 $path_layer1 --path_layer2 $path_layer2 --k 1000 --p 2 --keys $keys
# python3 experiments/experiment_ASIF.py --path_layer1 $path_layer1 --path_layer2 $path_layer2 --k 1250 --p 2 --keys $keys
# python3 experiments/experiment_ASIF.py --path_layer1 $path_layer1 --path_layer2 $path_layer2 --k 1500 --p 2 --keys $keys
# python3 experiments/experiment_ASIF.py --path_layer1 $path_layer1 --path_layer2 $path_layer2 --k 1750 --p 2 --keys $keys
# python3 experiments/experiment_ASIF.py --path_layer1 $path_layer1 --path_layer2 $path_layer2 --k 2000 --p 2 --keys $keys
# python3 experiments/experiment_ASIF.py --path_layer1 $path_layer1 --path_layer2 $path_layer2 --k 250 --p 3 --keys $keys
# python3 experiments/experiment_ASIF.py --path_layer1 $path_layer1 --path_layer2 $path_layer2 --k 500 --p 3 --keys $keys
# python3 experiments/experiment_ASIF.py --path_layer1 $path_layer1 --path_layer2 $path_layer2 --k 750 --p 3 --keys $keys
# python3 experiments/experiment_ASIF.py --path_layer1 $path_layer1 --path_layer2 $path_layer2 --k 1000 --p 3 --keys $keys
# python3 experiments/experiment_ASIF.py --path_layer1 $path_layer1 --path_layer2 $path_layer2 --k 1250 --p 3 --keys $keys
# python3 experiments/experiment_ASIF.py --path_layer1 $path_layer1 --path_layer2 $path_layer2 --k 1500 --p 3 --keys $keys
# python3 experiments/experiment_ASIF.py --path_layer1 $path_layer1 --path_layer2 $path_layer2 --k 1750 --p 3 --keys $keys
# python3 experiments/experiment_ASIF.py --path_layer1 $path_layer1 --path_layer2 $path_layer2 --k 2000 --p 3 --keys $keys
# python3 experiments/experiment_ASIF.py --path_layer1 $path_layer1 --path_layer2 $path_layer2 --k 250 --p 4 --keys $keys
# python3 experiments/experiment_ASIF.py --path_layer1 $path_layer1 --path_layer2 $path_layer2 --k 500 --p 4 --keys $keys
# python3 experiments/experiment_ASIF.py --path_layer1 $path_layer1 --path_layer2 $path_layer2 --k 750 --p 4 --keys $keys
# python3 experiments/experiment_ASIF.py --path_layer1 $path_layer1 --path_layer2 $path_layer2 --k 1000 --p 4 --keys $keys
# python3 experiments/experiment_ASIF.py --path_layer1 $path_layer1 --path_layer2 $path_layer2 --k 1250 --p 4 --keys $keys
# python3 experiments/experiment_ASIF.py --path_layer1 $path_layer1 --path_layer2 $path_layer2 --k 1500 --p 4 --keys $keys
# python3 experiments/experiment_ASIF.py --path_layer1 $path_layer1 --path_layer2 $path_layer2 --k 1750 --p 4 --keys $keys
# python3 experiments/experiment_ASIF.py --path_layer1 $path_layer1 --path_layer2 $path_layer2 --k 2000 --p 4 --keys $keys

# python3 experiments/experiment_ASIF.py --path_layer1 $path_layer1 --path_layer2 $path_layer2 --k 250 --p 5 --keys $keys
# python3 experiments/experiment_ASIF.py --path_layer1 $path_layer1 --path_layer2 $path_layer2 --k 250 --p 6 --keys $keys
# python3 experiments/experiment_ASIF.py --path_layer1 $path_layer1 --path_layer2 $path_layer2 --k 250 --p 7 --keys $keys
# python3 experiments/experiment_ASIF.py --path_layer1 $path_layer1 --path_layer2 $path_layer2 --k 250 --p 8 --keys $keys
# python3 experiments/experiment_ASIF.py --path_layer1 $path_layer1 --path_layer2 $path_layer2 --k 250 --p 9 --keys $keys

# Do we have the same tendency with ASIF zero-shot?

python3 experiments/experiment_ASIF.py --path_layer1 "../experiments/layers/embeddings_layer0_wav2vec2.json" --path_layer2 $path_layer2 --k 250 --p 6 --keys $keys
python3 experiments/experiment_ASIF.py --path_layer1 "../experiments/layers/embeddings_layer1_wav2vec2.json" --path_layer2 $path_layer2 --k 250 --p 6 --keys $keys
python3 experiments/experiment_ASIF.py --path_layer1 "../experiments/layers/embeddings_layer2_wav2vec2.json" --path_layer2 $path_layer2 --k 250 --p 6 --keys $keys
python3 experiments/experiment_ASIF.py --path_layer1 "../experiments/layers/embeddings_layer3_wav2vec2.json" --path_layer2 $path_layer2 --k 250 --p 6 --keys $keys
python3 experiments/experiment_ASIF.py --path_layer1 "../experiments/layers/embeddings_layer4_wav2vec2.json" --path_layer2 $path_layer2 --k 250 --p 6 --keys $keys
python3 experiments/experiment_ASIF.py --path_layer1 "../experiments/layers/embeddings_layer5_wav2vec2.json" --path_layer2 $path_layer2 --k 250 --p 6 --keys $keys
python3 experiments/experiment_ASIF.py --path_layer1 "../experiments/layers/embeddings_layer7_wav2vec2.json" --path_layer2 $path_layer2 --k 250 --p 6 --keys $keys
python3 experiments/experiment_ASIF.py --path_layer1 "../experiments/layers/embeddings_layer8_wav2vec2.json" --path_layer2 $path_layer2 --k 250 --p 6 --keys $keys
python3 experiments/experiment_ASIF.py --path_layer1 "../experiments/layers/embeddings_layer9_wav2vec2.json" --path_layer2 $path_layer2 --k 250 --p 6 --keys $keys
python3 experiments/experiment_ASIF.py --path_layer1 "../experiments/layers/embeddings_layer10_wav2vec2.json" --path_layer2 $path_layer2 --k 250 --p 6 --keys $keys
python3 experiments/experiment_ASIF.py --path_layer1 "../experiments/layers/embeddings_layer11_wav2vec2.json" --path_layer2 $path_layer2 --k 250 --p 6 --keys $keys

# Experiment3_text_vs_speech: Can ASIF be equivalent to CKA metric?. Saved in results/experiment3_text_vs_speech