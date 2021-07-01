import sys
import pickle

_, entities_tsv_in, entity_name_dict_in, bert_input_tsv_out = sys.argv

with open(entities_tsv_in, 'r') as f:
    entities_tsv = f.read().strip().split('\n')

with open(entity_name_dict_in, 'rb') as f:
    entity_name_dict = pickle.load(f)

#is_relation = False
is_relation = True

bert_input_tsv = ''
for i, line in enumerate(entities_tsv):
    text = line.split('\t')[1]
    if is_relation:
        name = text
    else:
        key_, id_ = text.split('::')
        if id_ in entity_name_dict:
            name = entity_name_dict[id_]
            if name is None:
                name = 'NoText'
        else:
            name = 'NoText'
    bert_input_tsv += name + '\t0\n'

with open(bert_input_tsv_out, 'w') as f:
    f.write(bert_input_tsv)

