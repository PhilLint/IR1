import torch
import torch.nn as nn

def aggregate_embeddings(w2vec, doc_id):
    """

    Args:
        w2vec:
        doc_id:

    Returns:

    """
    embeddings = w2vec.get_embedding_weights()
    word_embeddings = []
    for word in w2vec.data_indices[doc_id]:
        if word in list(w2vec.id2word.keys()):
            word_embeddings.append(torch.tensor(embeddings[word]))

    mean = torch.mean(torch.stack(word_embeddings), dim=0)
    return mean

def aggregate_query_emb(w2vec, query):
    """

    Args:
        w2vec:
        query:

    Returns:

    """
    embeddings = w2vec.get_embedding_weights()
    word_embeddings = []
    word_ids = []
    for word in query:
        if word in w2vec.word2id.keys():
            word_ids.append(w2vec.word2id[word])

    if len(word_ids) == 0:
        return None

    for word in word_ids:
        word_embeddings.append(torch.tensor(embeddings[word]))

    mean = torch.mean(torch.stack(word_embeddings), dim=0)

    return mean

def get_similarities(doc_emb, query_emb):
    """

    Args:
        doc_emb:
        query_emb:

    Returns:

    """

    cos = nn.CosineSimilarity(dim=1, eps=1e-6)
    output = cos(query_emb, doc_emb)
    return output

def get_ranking(similarities, topk=10):
    """

    Args:
        similarities:
        topk:

    Returns:

    """
    out_dict = {}
    ranking = similarities.argsort(descending=True)
    vals = similarities[ranking]
    ranking = ranking[:topk]

    return ranking, vals

def get_doc(w2vec, ranking, topk=10):
    """

    Args:
        w2vec:
        ranking:
        topk:

    Returns:

    """

    doc_list = []
    doc_names = []
    for i, doc in enumerate(ranking):
        if i == topk:
            break
        doc_names.append(w2vec.doc_ids[doc.item()])


        words = []
        docs = w2vec.data_indices
        for indices in docs:
            for idx in indices:
                words.append(w2vec.id2word[idx])

        doc_list.append(words)


    return doc_list, doc_names




