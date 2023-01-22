"""Train a language model from raw samples or argmax."""

import click
import yaml
import math
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import json
from tqdm import tqdm
import math
#from data import data
from dataclasses import dataclass
import collections
#from utils import utils
#from utils import config_utils
#from models import models
import torch
from torch import nn

#from pretrain_from_samples import build
import transformers
from tasks.seq import SequenceLMModel

def print_topk(logits, tokenizer, length, count=10):
  distrib = torch.softmax(logits, dim=-1)
  sorted_distrib, sorted_indices = torch.sort(distrib, descending=True)
  for i in range(count):
    print(tokenizer.decode(sorted_indices[0,length-2,i]), sorted_distrib[0,length-2,i])
  is_prob = distrib[0,length-2,tokenizer(' is')['input_ids'][0]]
  are_prob = distrib[0,length-2,tokenizer(' are')['input_ids'][0]]
  print('~~ are', are_prob.item(), '~~ is', is_prob.item(), '~~ ratio:', is_prob.item() / are_prob.item())


def visualize_word(word, tokenizer, model, count=20, contents=None):
    print(word)
    tokens = tokenizer(word)['input_ids']*512
    tokens = torch.tensor(tokens).unsqueeze(0).to('cuda')
    #logits1, attns1, contents1 = model({'input_ids': tokens}, return_components=True)
    if contents is None:
      contents = model.transformer.content_model(tokens) #(bs, nv, s, d)
      contents = contents[0,:,0,:] #(nv, d)

    for i in range(contents.shape[0]):
      print(i)
      logits = contents[i,:] @ model.lm_head.weight.t() # (vocab,)
      sorted_logits, sorted_indices = torch.sort(logits, descending=True)
      for i in range(count):
        print(tokenizer.decode(sorted_indices[i]), sorted_logits[i])
      for i in range(count):
        print(tokenizer.decode(sorted_indices[-i-1]), sorted_logits[-i-1])
    return contents
    print()
    print()
    print()



def load_non_optimized_model(model):
  model.to('cuda')
  config = model.config
  for k in vars(config):
    if 'fused' in k or 'flash' in k:
      setattr(config, k, False)
  newmodel = type(model)(config)
  newmodel.to('cuda')
  newmodel.load_state_dict(model.state_dict())
  return newmodel

@click.command()
@click.option('--config', default=None,
    help='path to the config override file')
def train(config):
  no_reduction_ce = nn.CrossEntropyLoss(reduction='none')
  tokenizer = transformers.AutoTokenizer.from_pretrained('gpt2')
  #model = SequenceLMModel.load_from_checkpoint('checkpoints/backpack-micro-flash-fp16/step_100000.ckpt')
  model = SequenceLMModel.load_from_checkpoint('checkpoints/backpack-small-flash-fp16/last.ckpt')
  #model = load_non_optimized_model(model.model)
  model = model.model.to('cuda')

  model.eval()
  for param in model.parameters():
    param.requires_grad = False

  #_avg = visualize_word(' bank', tokenizer, model)
  #man_avg = visualize_word(' San', tokenizer, model)
  #king_avg = visualize_word(' Francisco', tokenizer, model)
  #queen_avg = visualize_word(' queen', tokenizer, model)
  #woman_avg = visualize_word(' woman', tokenizer, model)
  #woman_avg = visualize_word(' mix', tokenizer, model,
  #    contents=king_avg - man_avg + woman_avg)

  #man_avg = visualize_word(  ' driver', tokenizer, model)
  #king_avg = visualize_word( ' truck', tokenizer, model)
  #queen_avg = visualize_word(' believes', tokenizer, model)
  #woman_avg = visualize_word(' Doctor', tokenizer, model)
  #woman_avg = visualize_word(' mix', tokenizer, model,
  #    contents=(king_avg + man_avg + woman_avg + queen_avg)/4)

  #man_avg = visualize_word(  '.', tokenizer, model)
  #king_avg = visualize_word( ' into', tokenizer, model)
  #queen_avg = visualize_word('The', tokenizer, model)
  #woman_avg = visualize_word(' nurse', tokenizer, model)
  #woman_avg = visualize_word(' mix', tokenizer, model,
  #    contents=(king_avg + man_avg + woman_avg + queen_avg)/4)

  print('Ready!')
  while True:
    _ = visualize_word(  input().strip('\n'), tokenizer, model)



if __name__ == '__main__':
  exp, config = train()
