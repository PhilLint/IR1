from train_word2vec import *

##### REPORT
# init dataset and model
w2vec = word2vec(200, min_freq=150, lr=0.001, device="cpu", subset=10)
w2vec.load_model("./save_model/model_step_10000.pt")

# get vector list
vector = w2vec.get_embedding_weights()

# get top k similar word
sim_list = w2vec.most_similar('vice', top_k=10)
print(sim_list)
# get top k similar word
sim_list = w2vec.most_similar('chang', top_k=10)
print(sim_list)
# get top k similar word
sim_list = w2vec.most_similar('control', top_k=10)
print(sim_list)
# get top k similar word
sim_list = w2vec.most_similar('among', top_k=10)
print(sim_list)
# get top k similar word
sim_list = w2vec.most_similar('oper', top_k=10)
print(sim_list)

# MORE DATA USED WHILE TRAINING
# init dataset and model
w2vec = word2vec(200, min_freq=150, lr=0.001, device="cpu", subset=15)
w2vec.load_model("./save_model/model_step_5000.pt")

# get vector list
vector = w2vec.get_embedding_weights()

# get top k similar word
sim_list = w2vec.most_similar('vice', top_k=10)
print(sim_list)
# get top k similar word
sim_list = w2vec.most_similar('chang', top_k=10)
print(sim_list)
# get top k similar word
sim_list = w2vec.most_similar('control', top_k=10)
print(sim_list)
# get top k similar word
sim_list = w2vec.most_similar('among', top_k=10)
print(sim_list)
# get top k similar word
sim_list = w2vec.most_similar('oper', top_k=10)
print(sim_list)

