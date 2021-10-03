# [.py files]

Since get_embeddings.ipynb did not work well in my environment, I created the python script files (e.g., get_embeddings.py). They work with V100 x 4 environment.

# [split_embeddings_to_dic.ipynb]

Since the obtained embeddings of the full train data was one file and it was huge (16GB), this notebook split them into 100 dictionary data. Then, I was able to load the embeddings data onto Kaggle envrionment.

# [glr2021-inference.ipynb]

This notebook is the version 27 of my submission notebook of the 2021 competition. The main idea is the same as the original `blend_ranking.ipynb`. The differences are: 1) The only config1 and config7 were used, 2) The full embeddings (4,132,914) were filted by the private train data. Also, made it work in Kaggle environment.
