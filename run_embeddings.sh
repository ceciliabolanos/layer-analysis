#!/bin/bash
# Loop from 0 to 9
#for i in {0..9}
#do
#    # Run the Python script with the current value of i
#    python3 embeddings/matrix_embeddings.py --layer $i --model 'encodecmae_mel256-ec-base' --words_path 'words_in_order1.json'
#done

#for i in {0..9}
#do
#    # Run the Python script with the current value of i
#    python3 embeddings/matrix_embeddings.py --layer $i --model 'encodecmae_mel256-ec-base_st-nopn' --words_path 'words_in_order1.json'
#done

#for i in {0..9}
#do
    # Run the Python script with the current value of i
#    python3 embeddings/matrix_embeddings.py --layer $i --model 'encodecmae_mel256-ec-base-as' --words_path 'words_in_order1.json'
#done

for i in {2..9}
do
    # Run the Python script with the current value of i
    python3 embeddings/matrix_embeddings.py --layer $i --model 'encodecmae_mel256-ec-base-fma' --words_path 'words_in_order1.json'
done

for i in {0..9}
do
    # Run the Python script with the current value of i
    python3 embeddings/matrix_embeddings.py --layer $i --model 'encodecmae_mel256-ec-base-ll' --words_path 'words_in_order1.json'
done
