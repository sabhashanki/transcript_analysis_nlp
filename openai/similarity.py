import pickle
import os
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

path_to_pkl = 'embeddings/topic_embeddings/'
pkl_files = [pos_pkl for pos_pkl in os.listdir(path_to_pkl) if pos_pkl.endswith('.pkl')]
topic_embed = {}
for index, topic_name in enumerate(pkl_files):
    with open(os.path.join(path_to_pkl, topic_name), 'rb') as pkl_file:
        output = np.array(pickle.load(pkl_file))
        name = topic_name[:-4]
        topic_embed[name] = output
print(topic_embed.keys())

path_to_pkl = 'embeddings/summary_embeddings/'
pkl_files = [pos_pkl for pos_pkl in os.listdir(path_to_pkl) if pos_pkl.endswith('.pkl')]
summ_embed = {}
for index, summ_name in enumerate(pkl_files):
    with open(os.path.join(path_to_pkl, summ_name), 'rb') as pkl_file:
        output = np.array(pickle.load(pkl_file))
        summ_embed[summ_name[8:]] = output
print(summ_embed.keys())

final_result = {}
for summ, s_embed in summ_embed.items():
    summary_topic = []
    for topic, t_embed in topic_embed.items():
        output = cosine_similarity(s_embed.reshape(-1,1), t_embed.reshape(-1,1))
        summary_topic.append(output)
    final_result[summ] = list(topic_embed)[summary_topic.index(max(summary_topic))]
print(final_result)




