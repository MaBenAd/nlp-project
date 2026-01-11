import os
import json
import random
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader

from transformers import (
    BertTokenizer,
    BertForMultipleChoice,
    get_linear_schedule_with_warmup,
)

from sklearn.metrics import accuracy_score

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
