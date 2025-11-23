#encoding=utf8

print('importing...')
import time
import sys

import os
import numpy as np
import pickle
import json
import torch

# load Q
action_space = 100
CHECKPOINT_FILE = 'checkpoint.pth'
JSON_FILE = 'checkpoint.json'

obj = torch.load(CHECKPOINT_FILE)
print(obj.keys())
for key in ['step_i', 'loss']: 
    print('-' * 100)
    print('key: %s' % (key))
    print('value: ', obj[key])

with open(JSON_FILE, 'r', encoding='utf-8') as f: 
    obj_information = json.load(f)

print('-' * 100)
print(obj_information)
