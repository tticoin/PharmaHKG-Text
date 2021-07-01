import sys
import os
from lxml import etree
import pickle
import random
import xml.etree.ElementTree as et

def dump_dict(dict_, path):
    with open(path, 'wb') as f:
        pickle.dump(dict_, f)

#_, xml_path = sys.argv
#_, xml_path, output_dir = sys.argv
_, xml_path, name_dict_out, desc_dict_out = sys.argv

mesh_name_dict = {}
mesh_desc_dict = {}

#root = etree.parse(xml_path, parser=etree.XMLParser())
tree = et.parse(xml_path)
root = tree.getroot()

triples = []
cnt = 0
#for concept in root.xpath('./*[local-name()="ConceptList"]'):

for child in root:
    mesh_id = child.find('DescriptorUI').text
    mesh_name = child.find('DescriptorName').find('String').text
    mesh_name_dict[mesh_id] = mesh_name
    for concept in child.find('ConceptList'):
        concept_name = concept.find('ConceptName').find('String').text
        scope_note = concept.find('ScopeNote')
        if mesh_name == concept_name and scope_note is not None:
            mesh_desc_dict[mesh_id] = scope_note.text

print(len(mesh_desc_dict))

dump_dict(mesh_name_dict, name_dict_out)
dump_dict(mesh_desc_dict, desc_dict_out)

    #drug_id = drug.xpath('./*[local-name()="drugbank-id"][@primary="true"]')[0].text
    #drug_name = drug.xpath('./*[local-name()="name"]')[0].text
    #ent_name_dict[drug_id] = drug_name

    #for key in ('targets', 'enzymes', 'carriers', 'transporters'):
    #    values = drug.xpath('./*[local-name()="' + key + '"]')[0]
    #    for value in values:
    #        be_id = value.xpath('./*[local-name()="id"]')[0].text
    #        be_name = value.xpath('./*[local-name()="name"]')[0].text
    #        ent_name_dict[be_id] = be_name
    #        triples.append((drug_id, key, be_id))

    ## Categories
    #for category in drug.xpath('./*[local-name()="categories"]')[0]:
    #    mesh_id = category.xpath('./*[local-name()="mesh-id"]')[0].text
    #    if mesh_id is not None:
    #        mesh_name = category.xpath('./*[local-name()="category"]')[0].text
    #        ent_name_dict[mesh_id] = mesh_name
    #        triples.append((drug_id, 'category', mesh_id))

#drug_freq = {}
#protein_freq = {}
#mesh_freq = {}
#for t in triples:
#    head, rel, tail = t
#    if head in drug_freq:
#        drug_freq[head] += 1
#    else:
#        drug_freq[head] = 1
#
#    if rel == 'category':
#        if tail in mesh_freq:
#            mesh_freq[tail] += 1
#        else:
#            mesh_freq[tail] = 1
#    else:
#        if tail in protein_freq:
#            protein_freq[tail] += 1
#        else:
#            protein_freq[tail] = 1
#print(len(drug_freq))
#print(len(mesh_freq))
#print(len(protein_freq))

#random.seed(0)
#shuffled_triples = random.sample(triples, len(triples))
#n_valid = 5000
#n_test = 5000
#train = shuffled_triples[:-(n_valid+n_test)]
#valid = shuffled_triples[-(n_valid+n_test):-n_test]
#test = shuffled_triples[-n_test:]
#
#print(len(train), len(valid), len(test))
#
#if not os.path.exists(output_dir):
#    os.makedirs(output_dir)
#with open(os.path.join(output_dir, 'train.txt'), 'w') as f:
#    f.write('\n'.join(['\t'.join(t) for t in train]) + '\n')
#with open(os.path.join(output_dir, 'valid.txt'), 'w') as f:
#    f.write('\n'.join(['\t'.join(t) for t in valid]) + '\n')
#with open(os.path.join(output_dir, 'test.txt'), 'w') as f:
#    f.write('\n'.join(['\t'.join(t) for t in test]) + '\n')
#
#with open(ent_name_dict_out, 'wb') as f:
#    pickle.dump(ent_name_dict, f)
