import os
import argparse
import csv
import pickle
import numpy as np
from tqdm import tqdm, trange
import torch
from torch import nn
from transformers import AutoModel, AutoTokenizer, AutoConfig
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset

class Model(nn.Module):
    def __init__(self, model_name):
        super(Model, self).__init__()

        self.bert = AutoModel.from_pretrained(model_name)

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None,):

        outputs = self.bert(input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids,
                            position_ids=None,
                            head_mask=None)
        token_embs = outputs[0]
        pooled_output = outputs[1]

        return pooled_output

parser = argparse.ArgumentParser()

# Input
parser.add_argument("--entities_tsv", default=None, type=str, required=True, help="The path of entities list tsv file")
parser.add_argument("--texts_dict", default=None, type=str, required=True, help="The path of texts dict")
parser.add_argument("--mode", default=None, type=str, required=True, help="noaug or aug or aug+name")

# Output
parser.add_argument("--output_dir", default=None, type=str, required=True, help="The directory to save output files")

# Parameter
parser.add_argument("--gpu", default='0', type=str, help="")
parser.add_argument("--do_lower_case", action="store_true", help="")
parser.add_argument("--text_type", default="description", type=str, help="")

args = parser.parse_args()


with open(args.texts_dict, 'rb') as f:
    texts = pickle.load(f)

mask = [] # For "aug+name" mode
coverage = {}
bert_inputs = []
with open(args.entities_tsv, encoding='utf-8', newline='') as f:
    for cols in csv.reader(f, delimiter='\t'):

        if args.mode == 'noaug':
            text_type = args.text_type
        else:
            text_type = 'name'

        sp = cols[1].split('::')
        if len(sp) == 3:
            kb_type, nid, text_type = sp
            key = '::'.join(sp[:2])
            mask.append(1)
        elif len(sp) == 2:
            kb_type, nid = sp
            key = cols[1]
            mask.append(0)

        assert key in texts
        if text_type not in texts[key]: # For "noaug" mode
            text_type = 'name'

        text = texts[key][text_type]

        if isinstance(text, list):
            text = ','.join(text)

        print(cols[1], text_type, text[:30])
        bert_inputs.append(text)

model_name = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract"
config = AutoConfig.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name, do_lower_case=args.do_lower_case)

all_input_ids = []
all_attention_mask = []
all_token_type_ids = []
for bi in bert_inputs:
    rtn = tokenizer(bi, add_special_tokens=True, padding='max_length', truncation=True, max_length=512, return_token_type_ids=True, return_attention_mask=True)
    all_input_ids.append(rtn['input_ids'])
    all_attention_mask.append(rtn['attention_mask'])
    all_token_type_ids.append(rtn['token_type_ids'])

dataset = TensorDataset(torch.LongTensor(all_input_ids), torch.LongTensor(all_attention_mask), torch.LongTensor(all_token_type_ids))

model = Model(model_name)
model.to(torch.device("cuda:"+args.gpu))

model.eval()
preds = None
eval_sampler = SequentialSampler(dataset)
eval_dataloader = DataLoader(dataset, sampler=eval_sampler, batch_size=256)
for batch in tqdm(eval_dataloader, desc="Converting"):
    batch = tuple(t.to(torch.device("cuda:"+args.gpu)) for t in batch)
    with torch.no_grad():
        inputs = {'input_ids':           batch[0],
                  'attention_mask':      batch[1],
                  'token_type_ids':      batch[2],
        }
        outputs = model(**inputs)

        if preds is None:
            preds = outputs.detach().cpu().numpy()
        else:
            preds = np.append(preds, outputs.detach().cpu().numpy(), axis=0)

if args.mode == 'aug+name':
    mask = np.expand_dims(mask, -1)
    eps, gamma = 2., 12.
    hidden_dim = preds.shape[1]
    init = (gamma + eps) / hidden_dim
    uniform_init = np.random.uniform(low=-init, high=-init, size=preds.shape)
    mask_ = (mask - 1) * -1
    preds = preds * mask + uniform_init * mask_

print(preds.shape)
np.save(emb_path, preds)
if args.mode == 'noaug':
    emb_path = os.path.join(args.output_dir, '{}_{}'.format(args.mode, args.text_type,))
elif args.mode == 'aug':
    emb_path = os.path.join(args.output_dir, '{}_{}'.format(args.mode))
np.save(emb_path, preds)
