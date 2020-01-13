from surprise import Reader
from surprise import Dataset
# from surprise import evaluate, print_perf
import os

# path to dataset file
file_path = os.path.expanduser('~/.surprise_data/ml-100k/ml-100k/u.data')

# As we're loading a custom dataset, we need to define a reader. In the
# movielens-100k dataset, each line has the following format:
# 'user item rating timestamp', separated by '\t' characters.
reader = Reader(line_format='user item rating timestamp', sep='\t')

data = Dataset.load_from_file(file_path, reader=reader)
data.split(n_folds=5)  # data can now be used normally