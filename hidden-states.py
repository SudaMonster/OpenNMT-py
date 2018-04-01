import torch
import torch.nn as nn
import os
import argparse
import onmt
import pdb

parser = argparse.ArgumentParser()
parser.add_argument('--main')
parser.add_argument('--aux')
parser.add_argument('--model')

opt = parser.parse_args()

main_hidden_states = torch.load(opt.main)
aux_hidden_states = torch.load(opt.aux)

checkpoint = torch.load(opt.model, map_location='cpu')
W = checkpoint['generator']['0.weight']
b = checkpoint['generator']['0.bias'][None,:]

def h2p(h):
    # h : len * dim
    dim = h.size(1)
    p = torch.mm(
        h, W.view(dim, -1)
    ) + b
    p = torch.autograd.Variable(p)
    p = torch.nn.functional.softmax(p)
    return p

def KL(p1, p2):
    return torch.sum(
        p1 * torch.log(
            (p1 + 1e-9) / (p2 + 1e-9)
        ),
        dim=1
    )

for seq_main, seq_aux in zip(main_hidden_states, aux_hidden_states):
    #print(torch.sum((seq_aux - seq_main) * (seq_aux - seq_main), dim=1))
    p_main = h2p(seq_main.data)
    p_aux = h2p(seq_aux.data)
    print(KL(p_main, p_aux))
    #pdb.set_trace()
    #print(torch.nn.functional.cosine_similarity(seq_aux, seq_main))
    input()
    #for h_m, h_a in zip(seq_main, seq_aux):

