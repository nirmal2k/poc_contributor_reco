
import torch
import random
import re
# from transformers import AdamW, Adam
import torch.optim as optim
from transformers import get_linear_schedule_with_warmup
import numpy as np
from sentence_transformers import SentenceTransformer, util
from torch import nn

def get_optimized_params(bert_model):
    '''
    Function that extracts the list of parameters to be updated by the optimizer
    :param bert_model: The TwinBERT model for which the parameters should be trained
    :return: list of parameters to be updated by the optimizer
    '''
    param_optimizer = list(bert_model.named_parameters())
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    optimizer_parameters = [
        {
            "params": [
                p for n, p in param_optimizer if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.001,
        },
        {
            "params": [
                p for n, p in param_optimizer if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },  
    ]
    return optimizer_parameters

def chunks(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

def sim_matrix(a, b, eps=1e-8):
    a_n, b_n = a.norm(dim=1)[:, None], b.norm(dim=1)[:, None]
    a_norm = a / torch.max(a_n, eps * torch.ones_like(a_n))
    b_norm = b / torch.max(b_n, eps * torch.ones_like(b_n))
    sim_mt = torch.mm(a_norm, b_norm.transpose(0, 1))
    return sim_mt

def resize_token_type_embeddings(model, new_num_types: int, random_segment: bool):
    old_token_type_embeddings = model.embeddings.token_type_embeddings
    new_token_type_embeddings = nn.Embedding(new_num_types, old_token_type_embeddings.weight.size(1))
    if not random_segment:
        new_token_type_embeddings.weight.data[:old_token_type_embeddings.weight.size(0)] = old_token_type_embeddings.weight.data

    model.config.type_vocab_size = new_num_types    
    model.embeddings.token_type_embeddings = new_token_type_embeddings


def get_optimizer_scheduler(models, num_tr_samples, lr, bs, ne):
    model_parameters = []
    for model in models:
        model_parameters += get_optimized_params(model)
    optimizer = optim.Adam(model_parameters,
                      lr=lr)

    num_train_steps = int(num_tr_samples / bs*ne)
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=0,
                                                num_training_steps=num_train_steps)
    return optimizer, scheduler

