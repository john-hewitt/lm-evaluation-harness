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

def localize(target_word, contexts, model, nv, vocsize, tokenizer):
  plus_stats = torch.zeros(vocsize, nv).cuda()
  mins_stats = torch.zeros(vocsize, nv).cuda()
  for context in contexts:
    print(tokenizer.convert_ids_to_tokens(tokenizer(context)['input_ids']))
    tokens = tokenizer(context)['input_ids']
    length = len(tokens)
    tokens = tokens + tokenizer('<|endoftext|>')['input_ids']*(512-len(tokens))
    tokens = torch.tensor(tokens).unsqueeze(0).to('cuda')
    
    with torch.autocast(device_type='cuda', dtype=torch.float16):
      content = model.model.transformer.content_model(tokens) # (bs, nv, s, d)
      #print(content.shape)
      log_distributions = content @ model.model.lm_head.weight.t() # (bs, nv, s, voc)
      #print(log_distributions.shape)
      _context_hiddens = model.model.transformer.gpt2_model(tokens) # (bs, nv, s, s)
      contextualization = model.model.transformer.contextualization_attn(_context_hiddens)

    # figure out weights for last word
    contextualization = contextualization[:,:,length-2, :] # predicting the last word (bs, nv, s)
    weighted_log_distributions = log_distributions * contextualization.unsqueeze(3) # (bs, nv, s, voc)
    #weighted_log_distributions = log_distributions
    weighted_log_positive = weighted_log_distributions[:,:,:,target_word]

    weighted_log_negative = weighted_log_distributions.clone()
    weighted_log_negative[:,:,:,target_word] = 0
    weighted_log_negative = weighted_log_negative.sum(dim=-1)

    for i in range(length-1):
      #print(tokens[0,i], tokenizer.convert_ids_to_tokens(tokens[0,i]))
      plus_stats[tokens[0,i],:] += weighted_log_positive[0,:,i]
    for i in range(length-1):
      mins_stats[tokens[0,i],:] += weighted_log_negative[0,:,i]

  sorted_plus, plus_indices = torch.sort(plus_stats.reshape(-1), descending=True)
  for i in range(10):
    indx = plus_indices[i]
    score = sorted_plus[i]
    word_indx = indx // plus_stats.shape[1]
    vec_indx = indx % plus_stats.shape[1]
    print(tokenizer.decode(word_indx), vec_indx, score)




@click.command()
@click.option('--config', default=None,
    help='path to the config override file')
def train(config):
  no_reduction_ce = nn.CrossEntropyLoss(reduction='none')
  tokenizer = transformers.AutoTokenizer.from_pretrained('gpt2')
  #model = SequenceLMModel.load_from_checkpoint('checkpoints/backpack-micro-flash-fp16/step_100000.ckpt')
  model = SequenceLMModel.load_from_checkpoint('checkpoints/backpack-small-flash-fp16/last.ckpt')
  model.to('cuda')

  model.eval()
  for param in model.parameters():
    param.requires_grad = False

  token_id = tokenizer(' she')['input_ids'][0]
  print(token_id)
  print(tokenizer.convert_ids_to_tokens(token_id))
  strings = [
      'The nurse came into the room. When X',
      'The nurse came into the room. Then X',
      'The nurse was with the patient. When X',
      'The nurse was with the patient. Then X',
      'I went to the talk to the nurse. What X',
      'I went over to the talk to the nurse. Why X',
      'I went over to the talk to the nurse; X',
      'I really liked my nurse. What I liked most was that X',
      'I really liked my nurse. In fact, X',
      'My nurse said that X',
      'My nurse believes that X',
      'This one nurse believes that X',
      ]
  #strings = [
  #    'The truck driver came into the room. When X',
  #    'The truck driver came into the room. Then X',
  #    'The truck driver was with the patient. When X',
  #    'The truck driver was with the patient. Then X',
  #    'I went to the talk to the truck driver. What X',
  #    'I went over to the talk to the truck driver. Why X',
  #    'I went over to the talk to the truck driver; X',
  #    'I really liked my truck driver. What I liked most was that X',
  #    'I really liked my truck driver. In fact, X',
  #    'My truck driver said that X',
  #    'My truck driver believes that X',
  #    'This one truck driver believes that X',
  #    'This one truck driver believes X',
  #    'My truck driver said X',
  #    'My truck driver believes X',
  #    ]
  localize(token_id, strings, model, 16, 50256, tokenizer)
  #localize(token_id, [' nurse X',], model, 16, 50256, tokenizer)



if __name__ == '__main__':
  exp, config = train()
