# Indexing system
# Piero Pettenà - January 2024

import os
import pandas as pd
import numpy as np
from collections import Counter
import string
from sklearn.decomposition import TruncatedSVD

svd = TruncatedSVD(n_components=100, n_iter=7, random_state=42)
