import pickle

with open(f'./embeddings/topic_embeddings/Arts.pkl', 'rb') as file:
    output = pickle.load(file)
    print(output)
