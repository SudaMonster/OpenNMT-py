import torch
import torch.nn as nn
import os
import argparse
import onmt
import pdb
import json

parser = argparse.ArgumentParser()
parser.add_argument('--main')
parser.add_argument('--aux')
parser.add_argument('--model')
parser.add_argument('--out')

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

def l2(h1, h2):
    return torch.sum((h1 - h2) * (h1 - h2), dim=1)

scores_list = []

for i, (seq_main, seq_aux) in enumerate(zip(main_hidden_states, aux_hidden_states)):
    if i % 100 == 0 and i > 0:
        print(i)

    #print(torch.sum((seq_aux - seq_main) * (seq_aux - seq_main), dim=1))
    p_main = h2p(seq_main.data)
    p_aux = h2p(seq_aux.data)
    #print(KL(p_main, p_aux).data.tolist())
    scores_list.append(
        {
            'idx' : i,
            'KL' : KL(p_main, p_aux).data.tolist(),
            'L2' : l2(seq_aux, seq_main).data.tolist()
        }
    )

model_name = opt.model.split('/')[-1]
with open(
        os.path.join(
            opt.out,
            model_name + '.score.json'
            ),
        'w+'
        ) as f:
    json.dump(scores_list, f, indent=4)
    #input()
    #print(scores_list[-1])

    #pdb.set_trace()
    #print(torch.nn.functional.cosine_similarity(seq_aux, seq_main))
    #input()
    #for h_m, h_a in zip(seq_main, seq_aux):

