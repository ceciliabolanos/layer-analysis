from scipy.spatial.distance import cosine
import editdistance
import numpy as np
import torch
import numpy as np
from tqdm import tqdm
import random
random.seed(42)

def phonetic_distance(w1, w2, lexicon):
    w1_phones = lexicon[w1.upper()]
    w2_phones = lexicon[w2.upper()]
    return editdistance.eval(w1_phones, w2_phones) / max(len(w1_phones), len(w2_phones))

def semantic_distance(w1, w2, glove_embeddings):
    w1_emb = glove_embeddings[w1]
    w2_emb = glove_embeddings[w2]
    return cosine(w1_emb, w2_emb)


def words_to_indices(word_list):
    word_dict = {}
    for index, word in enumerate(word_list):
        if word in word_dict:
            word_dict[word].append(index)
        else:
            word_dict[word] = [index]
    return word_dict

def unique_ordered(elements, indices):
    seen = set()
    unique = []
    indices = indices.numpy()
    indices_similarity = []
    for i in range(len(elements)):
        if elements[i] not in seen:
            unique.append(elements[i])
            seen.add(elements[i])
            indices_similarity.append(indices[i])
    return unique, indices_similarity


def get_mean_std_similarity(words_appears, d_matrix_w2v):
    result_dict_w2v = {}
    for key in tqdm(words_appears):
        for value in words_appears:
            distances_w2v = []
            for index1 in words_appears[key]:
                for index2 in words_appears[value]:
                    distances_w2v.append(d_matrix_w2v[index1, index2])

            if distances_w2v:
                std_distance_w2v = np.std(distances_w2v)
                mean_distance_w2v = np.mean(distances_w2v)
                result_dict_w2v[(key, value)] = [mean_distance_w2v, std_distance_w2v]

    return result_dict_w2v

def tuple_to_string(obj):
    if isinstance(obj, dict):
        return {str(key): tuple_to_string(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [tuple_to_string(item) for item in obj]
    else:
        return obj


def get_values(similarity, result_dict, unique_words):
    similarity_tensors = [torch.tensor(d) for d in similarity]
    similarity_tensor = torch.stack(similarity_tensors)

    all_vecinos = []
    count = []

    for threshold in np.arange(0.05, 0.80, 0.05):
        values_list = []
        words_dict = {}

        for i in tqdm(range(len(similarity_tensor)), desc=f'Threshold: {threshold:.2f}'):
            values, indices = torch.topk(similarity_tensor[i], k=8001, largest=False)
            mask = values < threshold
            nuevos_indices = indices[mask]
            all_words_reordered = [unique_words[indice] for indice in nuevos_indices]
            words, indices_similarity = unique_ordered(all_words_reordered, nuevos_indices)
            words_dict[unique_words[i]] = words
             #         
        vecinos = []
        for key, values in words_dict.items():
            for value in values[1:]:  # do not count the current word
                lejania = result_dict[(key,value)][0]
                vecinos.append(lejania)

        promedio_distancia = np.mean(vecinos)
        count.append(len(vecinos))

        # Store results for this threshold
        all_vecinos.append(promedio_distancia)

    return all_vecinos, count

def unique_ordered1(elements, indices):
    seen = set()
    unique = []
    indices = indices
    indices_similarity = []
    for i in range(len(elements)):
        if elements[i] not in seen:
            unique.append(elements[i])
            seen.add(elements[i])
            indices_similarity.append(indices[i])
    return unique, indices_similarity

def get_values_random(semantic, phonetic,count_sem, count_ph,result_dict, unique_words):
    semantic_tensors = [torch.tensor(d) for d in semantic]
    semantic_tensor = torch.stack(semantic_tensors)

    phonetic_tensors = [torch.tensor(d) for d in phonetic]
    phonetic_tensor = torch.stack(phonetic_tensors)

    all_vecinos = []
    count = []
    for threshold in np.arange(0.05, 0.80, 0.05):
        values_list = []
        words_dict = {}
        indices_below_threshold_sem = set()
        indices_below_threshold_ph = set()
   
        for i in tqdm(range(len(semantic_tensor)), desc=f'Threshold: {threshold:.2f}'):
            values_sem, indices_sem = torch.topk(semantic_tensor[i], k=8001, largest=False)
            mask_sem = values_sem > 0.7
            indices_below_threshold_sem.update(indices_sem[mask_sem].tolist()) 

            # Obtener los valores e índices para el tensor fonético
            values_ph, indices_ph = torch.topk(phonetic_tensor[i], k=8001, largest=False)
            mask_ph = values_ph > 0.7
            indices_below_threshold_ph.update(indices_ph[mask_ph].tolist()) 

            intersection_indices = indices_below_threshold_sem & indices_below_threshold_ph
            intersection_indices = list(intersection_indices)
            all_words_reordered = [unique_words[indice] for indice in intersection_indices]
            words, indices_similarity = unique_ordered1(all_words_reordered, intersection_indices)
            words_dict[unique_words[i]] = words

        vecinos = []
      
        for key, values in words_dict.items():
            for value in values[1:]:  # do not count the current word
                lejania = result_dict[(key,value)][0]
                vecinos.append(lejania)
            
        random_vecinos = random.sample(vecinos, 10000)
        
        promedio_distancia = np.mean(random_vecinos)
        count.append(len(random_vecinos))
        all_vecinos.append(promedio_distancia)

    return all_vecinos, count