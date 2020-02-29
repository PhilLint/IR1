import collections

import read_ap as read_ap
import os
import pickle as pkl
from collections import defaultdict, Counter

docs_by_id = read_ap.get_processed_docs()
keys = docs_by_id.keys()
docs_by_id_subset = {k: docs_by_id[k] for k in list(keys)[1:10]}
del docs_by_id

index_path = "./word2vec"

if os.path.exists(index_path):

    with open(index_path, "rb") as reader:
        index = pkl.load(reader)

    self.ii = index["ii"]
    self.df = index["df"]
else:
    self.ii = defaultdict(list)
    self.df = defaultdict(int)

    doc_ids = list(docs.keys())

    print("Building Index")
    # build an inverted index
    for doc_id in tqdm(doc_ids):
        doc = docs[doc_id]

        counts = Counter(doc)
        for t, c in counts.items():
            self.ii[t].append((doc_id, c))
        # count df only once - use the keys
        for t in counts:
            self.df[t] += 1

    with open(index_path, "wb") as writer:
        index = {
            "ii": self.ii,
            "df": self.df
        }
        pkl.dump(index, writer)

print('reading data...')
with open("file.txt", 'r', encoding='utf-8') as f:
    data = f.read().split()
print('corpus size', len(data))
vocab_size = len(data)
count = collections.Counter(data).most_common()
# dict((word,v )for word,v in cn.items() if v > 1 )
words = data
dictionary = dict()
for word, _ in count:
    dictionary[word] = len(dictionary)

data = list()
unk_count = 0
for word in words:
    if word in dictionary:
        index = dictionary[word]
    else:
        index = 0  # UNK index is 0
        unk_count += 1
    data.append(index)
count[0][1] = unk_count
reversed_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
