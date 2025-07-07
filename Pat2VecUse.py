from gensim.models.doc2vec import Doc2Vec

pat2vec_model = Doc2Vec.load('pat2vec_dim10.model')

print(pat2vec_model.infer_vector(["M54.1", "J06.9", "401", "R51"]))