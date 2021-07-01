import sys
import os
_, dir = sys.argv
with open(os.path.join(dir, 'entities.tsv'),'r') as f:
    entities = f.read().strip().split('\n')
    entities = [e.split('\t')[1] for e in entities]
name_tsv = ''
for e in entities:
    name_tsv += '{}\tname\t{}::name\n'.format(e, e)
with open(os.path.join(dir, 'name.tsv'),'w') as f:
    f.write(name_tsv)




