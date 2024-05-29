# Script used to run all CKA experiments

# Experiment2.1: How similar are the representations produced by the same model?. Saved in results/experiment2.1_same_model
# python3 experiments/experiment_CKA.py --model1 'bert-base-uncased' --layer1 12 --model2 'bert-base-uncased' --layer2 12
# python3 experiments/experiment_CKA.py --model1 'wav2vec2' --layer1 12 --model2 'wav2vec2' --layer2 12
# python3 experiments/experiment_CKA.py --model1 'BEATs_iter3' --layer1 12 --model2 'BEATs_iter3' --layer2 12
# python3 experiments/experiment_CKA.py --model1 'encodecmae_base' --layer1 10 --model2 'encodecmae_base' --layer2 10

# Experiment2.2: Same model trained with different dataset: Saved in results/experiment2.2_dif_dataset
# python3 experiments/experiment_CKA.py --model1 'encodecmae_mel256-ec-base_st-nopn' --layer1 10 --model2 'encodecmae_mel256-ec-base_st-nopn' --layer2 10
# python3 experiments/experiment_CKA.py --model1 'encodecmae_mel256-ec-base-as' --layer1 10 --model2 'encodecmae_mel256-ec-base-as' --layer2 10
# python3 experiments/experiment_CKA.py --model1 'encodecmae_mel256-ec-base-fma' --layer1 10 --model2 'encodecmae_mel256-ec-base-fma' --layer2 10
# python3 experiments/experiment_CKA.py --model1 'encodecmae_mel256-ec-base-ll' --layer1 10 --model2 'encodecmae_mel256-ec-base-ll' --layer2 10

# Experiment3.1: Text vs Speech (BERT vs W2V):
# python3 experiments/experiment_CKA.py --model1 'bert-base-uncased' --layer1 12 --model2 'wav2vec2' --layer2 12

# # Experiment3.2: Speech vs Audio: 
# python3 experiments/experiment_CKA.py --model1 'encodecmae_base' --layer1 10 --model2 'wav2vec2' --layer2 12
# python3 experiments/experiment_CKA.py --model1 'encodecmae_mel256-ec-base-ll' --layer1 10 --model2 'wav2vec2' --layer2 12
# python3 experiments/experiment_CKA.py --model1 'encodecmae_mel256-ec-base-as' --layer1 10 --model2 'wav2vec2' --layer2 12
# python3 experiments/experiment_CKA.py --model1 'encodecmae_mel256-ec-base-fma' --layer1 10 --model2 'wav2vec2' --layer2 12
# python3 experiments/experiment_CKA.py --model1 'encodecmae_mel256-ec-base_st-nopn' --layer1 10 --model2 'wav2vec2' --layer2 12
# python3 experiments/experiment_CKA.py --model1 'BEATs_iter3' --layer1 12 --model2 'wav2vec2' --layer2 12

# # Experiment3.3: Text vs Audio:
# python3 experiments/experiment_CKA.py --model1 'encodecmae_base' --layer1 10 --model2 'bert-base-uncased' --layer2 12
# python3 experiments/experiment_CKA.py --model1 'BEATs_iter3' --layer1 12 --model2 'bert-base-uncased' --layer2 12
# python3 experiments/experiment_CKA.py --model1 'encodecmae_base' --layer1 10 --model2 'glove' --layer2 10
# python3 experiments/experiment_CKA.py --model1 'BEATs_iter3' --layer1 12 --model2 'glove' --layer2 12


########### Other experiments

# python3 experiments/experiment_CKA.py --model1 'encodecmae_base' --layer1 10 --model2 'encodecmae_mel256-ec-base_st-nopn' --layer2 10
# python3 experiments/experiment_CKA.py --model1 'encodecmae_base' --layer1 10 --model2 'encodecmae_mel256-ec-base-as' --layer2 10
# python3 experiments/experiment_CKA.py --model1 'encodecmae_base' --layer1 10 --model2 'encodecmae_mel256-ec-base-fma' --layer2 10
# python3 experiments/experiment_CKA.py --model1 'encodecmae_base' --layer1 10 --model2 'encodecmae_mel256-ec-base-ll' --layer2 10
# python3 experiments/experiment_CKA.py --model1 'encodecmae_base' --layer1 10 --model2 'encodecmae_mel256-ec-base' --layer2 10

# python3 experiments/experiment_CKA.py --model1 'bert-base-uncased' --layer1 12 --model2 'encodecmae_mel256-ec-base_st-nopn' --layer2 10
# python3 experiments/experiment_CKA.py --model1 'bert-base-uncased' --layer1 12 --model2 'encodecmae_mel256-ec-base-as' --layer2 10
# python3 experiments/experiment_CKA.py --model1 'bert-base-uncased' --layer1 12 --model2 'encodecmae_mel256-ec-base-fma' --layer2 10
# python3 experiments/experiment_CKA.py --model1 'bert-base-uncased' --layer1 12 --model2 'encodecmae_mel256-ec-base-ll' --layer2 10
# python3 experiments/experiment_CKA.py --model1 'bert-base-uncased' --layer1 12 --model2 'encodecmae_mel256-ec-base' --layer2 10

# python3 experiments/experiment_CKA.py --model1 'wav2vec2' --layer1 12 --model2 'encodecmae_mel256-ec-base_st-nopn' --layer2 10
# python3 experiments/experiment_CKA.py --model1 'wav2vec2' --layer1 12 --model2 'encodecmae_mel256-ec-base-as' --layer2 10
# python3 experiments/experiment_CKA.py --model1 'wav2vec2' --layer1 12 --model2 'encodecmae_mel256-ec-base-fma' --layer2 10
# python3 experiments/experiment_CKA.py --model1 'wav2vec2' --layer1 12 --model2 'encodecmae_mel256-ec-base-ll' --layer2 10
# python3 experiments/experiment_CKA.py --model1 'wav2vec2' --layer1 12 --model2 'encodecmae_mel256-ec-base' --layer2 10

# python3 experiments/experiment_CKA.py --model1 'encodecmae_mel256-ec-base' --layer1 10 --model2 'encodecmae_mel256-ec-base_st-nopn' --layer2 10
# python3 experiments/experiment_CKA.py --model1 'encodecmae_mel256-ec-base' --layer1 10 --model2 'encodecmae_mel256-ec-base-as' --layer2 10
# python3 experiments/experiment_CKA.py --model1 'encodecmae_mel256-ec-base' --layer1 10 --model2 'encodecmae_mel256-ec-base-fma' --layer2 10
# python3 experiments/experiment_CKA.py --model1 'encodecmae_mel256-ec-base' --layer1 10 --model2 'encodecmae_mel256-ec-base-ll' --layer2 10
# python3 experiments/experiment_CKA.py --model1 'encodecmae_mel256-ec-base_st-nopn' --layer1 10 --model2 'encodecmae_mel256-ec-base-as' --layer2 10
# python3 experiments/experiment_CKA.py --model1 'encodecmae_mel256-ec-base_st-nopn' --layer1 10 --model2 'encodecmae_mel256-ec-base-fma' --layer2 10
# python3 experiments/experiment_CKA.py --model1 'encodecmae_mel256-ec-base_st-nopn' --layer1 10 --model2 'encodecmae_mel256-ec-base-ll' --layer2 10
# python3 experiments/experiment_CKA.py --model1 'encodecmae_mel256-ec-base-as' --layer1 10 --model2 'encodecmae_mel256-ec-base-fma' --layer2 10
# python3 experiments/experiment_CKA.py --model1 'encodecmae_mel256-ec-base-as' --layer1 10 --model2 'encodecmae_mel256-ec-base-ll' --layer2 10
# python3 experiments/experiment_CKA.py --model1 'encodecmae_mel256-ec-base-fma' --layer1 10 --model2 'encodecmae_mel256-ec-base-ll' --layer2 10