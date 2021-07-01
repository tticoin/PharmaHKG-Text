import os
import sys
import shutil
import json

_, tsv_dir, new_tsv_dir = sys.argv

if not os.path.exists(new_tsv_dir):
    os.makedirs(new_tsv_dir)
shutil.copyfile(os.path.join(tsv_dir, 'train.tsv'), os.path.join(new_tsv_dir, 'train.tsv'))

with open(os.path.join(tsv_dir, 'valid.tsv')) as f:
    valid = f.read().strip().split('\n')
with open(os.path.join(tsv_dir, 'test.tsv')) as f:
    test = f.read().strip().split('\n')
with open(os.path.join(tsv_dir, 'relations.tsv')) as f:
    relations = f.read().strip().split('\n')

rels = {l.split('\t')[1]: l.split('\t')[0] for l in relations[:9]}

for k, v in rels.items():
    new_test = [l for l in test if l.split('\t')[1] == k]
    new_valid = [l for l in test if l.split('\t')[1] != k]
    new_valid = valid + new_valid

    with open(os.path.join(new_tsv_dir, 'valid_{}.tsv'.format(v)), 'w') as f:
        f.write('\n'.join(new_valid))
    with open(os.path.join(new_tsv_dir, 'test_{}.tsv'.format(v)), 'w') as f:
        f.write('\n'.join(new_test))
