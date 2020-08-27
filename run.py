# -*- coding: utf-8 -*-
import os
import json
import pandas as pd
from ftfy import fix_text
# with open('data.json', encoding="utf-8") as f:
    
#     p = json.load(f)
    
df_json = pd.read_json('data (1).json', encoding='utf8')
# print(df_json.head())
df = df_json
x = df.content
#print(x.head())
import numpy as np
puct_set = set([c for c in '!"#$%&\'()*+,./:;<=>?@[\\]^`{|}~'])
X = np.array(x)
# print(X)
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
# count_vect = CountVectorizer(analyzer='word', token_pattern=r'\w{1,}')
# count_vect.fit(X)
# X_data_count = count_vect.transform(X)
documents = []
for x in X:
    a = x.lower()
    # print(a)
    documents.append(a)

import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
def removeRedundant(text,redundantSet):
    words = text.split()
    for i in range(0,len(words)):
        if words[i].count('_') == 0 and (words[i] in redundantSet or words[i].isdigit()):
            words[i] = ''
        else:
            sub_words = words[i].split('_')
            if any(w in redundantSet or w.isdigit() for w in sub_words):
                words[i] = ''
    words = [w for w in words if w != '']
    words = ' '.join(words)
    return words
def preprocessing(text):
    text = removeRedundant(text,puct_set)
    return text

clean_documents = []
for i in documents:
    clean_document = preprocessing(i)
    # print(clean_document)
    clean_documents.append(clean_document)
print('ok')

vectorizer = TfidfVectorizer(token_pattern = "\S+", min_df = 2)
vectors = vectorizer.fit_transform(clean_documents)
print ("Tf-idf shape: " + str(vectors.shape))
svd = TruncatedSVD(n_components=100, n_iter=10, random_state=42)
svd_vectors = svd.fit_transform(vectors)
# print ("Document 1's Vector : ")
# print (svd_vectors[0])
from sklearn.metrics.pairwise import pairwise_distances
import numpy as np
def distance(vecs):
    vec1 = vecs[0]
    vecAll = vecs[1]
    Dis_matrix = pairwise_distances(vec1,vecAll,metric = 'cosine',n_jobs=1)
    Dis_matrix = Dis_matrix.astype(np.float16)
    return Dis_matrix
def chunks_vec(l, n):
    for i in range(0, l.shape[0], n):
        yield l[i:i + n]

vector_chunks = list(chunks_vec(svd_vectors,1000))
vector_chunks = [(i,svd_vectors) for i in vector_chunks]
Dis_matrix = []
for vector_chunk in vector_chunks:
    Dis = distance(vector_chunk)
    Dis_matrix.append(Dis)

Dis_matrix = np.vstack(Dis_matrix)


print ('cosine distance between Document 1 and Document 2 : ', Dis_matrix[0][2])
print ('cosine distance between Document 2 and Document 3 : ', Dis_matrix[2][3])

from copy import deepcopy

THRESHOLD = 0.1

graph = deepcopy(Dis_matrix)
graph[graph <= THRESHOLD] = 2
graph[graph != 2] = 0
graph[graph == 2] = 1
graph = graph.astype(np.int8)
from scipy.sparse.csgraph import connected_components
res = connected_components(graph,directed=False)

from collections import OrderedDict
cluster_labels = res[1]
num_cluster = res[0]
res_cluster = OrderedDict()

for i in range(0,len(cluster_labels)):
    if cluster_labels[i] in res_cluster: res_cluster[cluster_labels[i]].append(i)
    else: res_cluster[cluster_labels[i]] = [i]
print("------------==============--------------")
res_cluster = [res_cluster[i] for i in range(0,num_cluster)]
res_cluster = [sorted(r) for r in res_cluster if len(r) > 1]
res_cluster.sort(key=len,reverse=True)
print ("Number of cluster: ", len(res_cluster))
print ("Number of clustered documents: ", len([j for i in res_cluster for j in i]))
print ("Number of noise documents: ", len(documents) - len([j for i in res_cluster for j in i]))
print("-----------==========----------")
# print(len(res_cluster[0]))
number = len(res_cluster)
checks = []
for i in range(0,number):
    # print('Cluster' + ' ' + str(i) + ' ' + 'has :' + str(len(res_cluster[i])) ) 
    # print(len(res_cluster[i]))
    check = 'Cluster' + ' '   + str(i) + ' ' + 'has :' + str(len(res_cluster[i])) + '\n'
    # print(check)
    checks.append(check)
    # print(checks)
with open('results.txt', "w") as f:
    f.writelines(checks)
    f.close()

