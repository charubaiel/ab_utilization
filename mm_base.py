import os
import re
import warnings
from io import BytesIO

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests as r
from tqdm import tqdm

warnings.simplefilter(action='ignore')
tqdm.pandas()
plt.style.use('ggplot')
pd.set_option('use_inf_as_na', True)
plt.rcParams['figure.figsize'] = (20, 7)
