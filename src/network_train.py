import torch
from torch import nn
import  torch.nn.functional as F

def sim_matrix(a, b, norm=True, eps=1e-8):
    if norm:
        a_n, b_n = a.norm(dim=1)[:, None], b.norm(dim=1)[:, None]
        a_norm = a / torch.max(a_n, eps * torch.ones_like(a_n))
        b_norm = b / torch.max(b_n, eps * torch.ones_like(b_n))
        sim_mt = torch.mm(a_norm, b_norm.transpose(0, 1))
        return sim_mt
    else:
        return torch.mm(a, b.transpose(0, 1))

class network_embedding(nn.Module):
    def __init__(self,node_emb, tag_embs, device='cuda') -> None:
        super().__init__()
        
        self.node_embeddings = nn.Embedding.from_pretrained(torch.from_numpy(node_emb).float()).to(device)
        self.node_embeddings.weight.requires_grad = True        
        self.word_embeddings = nn.Embedding.from_pretrained(torch.from_numpy(tag_embs).float()).to(device)
        self.word_embeddings.weight.requires_grad = True 
        self.criterion = nn.CrossEntropyLoss()
        self.device = device

    def get_loss_w_neg(self, node_node, bs, num_samples, type_):
        pos_left, pos_right = zip(*[i[0] for i in node_node])
        neg_left, neg_right = zip(*[j for i in node_node for j in i[1:]])

        pos_le = self.node_embeddings(torch.LongTensor(pos_left).to(self.device))
        neg_le = self.node_embeddings(torch.LongTensor(neg_left).to(self.device))
        
        if type_=='user':            
            pos_re = self.node_embeddings(torch.LongTensor(pos_right).to(self.device))        
            neg_re = self.node_embeddings(torch.LongTensor(neg_right).to(self.device))
        else:
            pos_re = self.word_embeddings(torch.LongTensor(pos_right).to(self.device))        
            neg_re = self.word_embeddings(torch.LongTensor(neg_right).to(self.device))

        pos_dot = (pos_le*pos_re).sum(1)
        neg_dot = (neg_le* -neg_re).sum(1).reshape((bs, num_samples-1))
        loss = -(F.logsigmoid(pos_dot) + F.logsigmoid(neg_dot).sum(dim=1)).mean()
        return loss


    def forward(self, node_node, type_):      
        nl_loss = self.get_loss_w_neg(node_node, len(node_node), len(node_node[0]), type_)
        # nw_loss = self.get_loss_w_neg(node_word, len(node_word), len(node_word[0]), 'word')

        return nl_loss