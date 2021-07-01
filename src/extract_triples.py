import sys
import os
from lxml import etree
import xml.etree.ElementTree as et
import pickle
import random
import argparse
from collections import defaultdict
import csv

parser = argparse.ArgumentParser()

# Input
parser.add_argument("--drugbank_xml_path", default=None, type=str, required=True, help="DrugBank xml file path")
parser.add_argument("--uniprot_xml_path", default=None, type=str, required=True, help="UniProt xml file path")
parser.add_argument("--mesh_xml_path", default=None, type=str, required=True, help="MeSH xml file path")
parser.add_argument("--smpdb_tsv_path", default=None, type=str, required=True, help="smpdb tsv file path")

# Output
parser.add_argument("--output_dir", default=None, type=str, help="The directory to save output files")

# Parameter
parser.add_argument("--data_split_ratio", nargs='+', default=[98, 1, 1], type=int, help="Data split ratio")
parser.add_argument("--add_atc_hierarchy", action="store_true", help="Whethter to add ATC hierarychical information into KG")
parser.add_argument("--seed", default=0, type=int, help="Random seed for shuffling tripiles order")

args = parser.parse_args()

# Set seed
random.seed(0)

triples = defaultdict(list)
entities = set()
texts = defaultdict(dict)

def add_triple(d, key, head, tail):
    d[key].append((head, tail))
    
# UniProt textual information
uniprot_root = etree.parse(args.uniprot_xml_path, parser=etree.XMLParser())
for protein in uniprot_root.xpath('./*[local-name()="entry"]'):
    p_synonyms = []
    gene_names = []
    #uniprot_id = protein.xpath('./*[local-name()="accession"]')[0].text
    uniprot_name = protein.xpath('./*[local-name()="name"]')[0].text
    protein_ = protein.xpath('./*[local-name()="protein"]')[0]
    recommended_name = protein_.xpath('./*[local-name()="recommendedName"]')[0]
    recommended_name_full = recommended_name.xpath('./*[local-name()="fullName"]')[0].text
    alternative_names = protein_.xpath('./*[local-name()="alternativeName"]')
    for alternative_name in alternative_names:
        alternative_name_full = alternative_name.xpath('./*[local-name()="fullName"]')[0].text
        if alternative_name_full is not None: p_synonyms.append(alternative_name_full)
    gene = protein.xpath('./*[local-name()="gene"]')
    if len(gene) != 0:
        gene_name = gene[0].xpath('./*[local-name()="name"]')[0].text
    function = protein.xpath('./*[local-name()="comment"][@type="function"]')
    if len(function) != 0:
        function_text = function[0].xpath('./*[local-name()="text"]')[0].text

    # UniProt entries have multiple IDs
    for uniprot_id in protein.xpath('./*[local-name()="accession"]'):
        uniprot_id = uniprot_id.text

        if recommended_name_full is not None: texts['UNIPROT::'+uniprot_id]['name'] = recommended_name_full
        if gene_name is not None: texts['UNIPROT::'+uniprot_id]['gene-name'] = gene_name
        if function_text is not None: texts['UNIPROT::'+uniprot_id]['description'] = function_text
        if len(p_synonyms) != 0: texts['UNIPROT::'+uniprot_id]['synonyms'] = p_synonyms

# MeSH textual information
mesh_tree = et.parse(args.mesh_xml_path)
mesh_root = mesh_tree.getroot()
for child in mesh_root:
    mesh_id = child.find('DescriptorUI').text
    mesh_name = child.find('DescriptorName').find('String').text
    texts['MESH::'+mesh_id]['name'] = mesh_name
    m_synonyms = []
    for concept in child.find('ConceptList'):
        concept_name = concept.find('ConceptName').find('String').text
        scope_note = concept.find('ScopeNote')
        if mesh_name == concept_name and scope_note is not None:
            mesh_desc = scope_note.text
            texts['MESH::'+mesh_id]['description'] = mesh_desc
        term_list = concept.find('TermList')
        for term in term_list:
            term_name = term.find('String').text
            m_synonyms.append(term_name)
    if len(m_synonyms) != 0: texts['MESH::'+mesh_id]['synonyms'] = m_synonyms

# Pathway
with open(args.smpdb_tsv_path, encoding='utf-8', newline='') as f:
    for cols in csv.reader(f):
        smpdb_id, pw_id, pathway_name, pathway_subject, pathway_description = cols
        texts['SMPDB::'+smpdb_id]['name'] = pathway_name
        texts['SMPDB::'+smpdb_id]['description'] = pathway_description


# DrugBank
root = etree.parse(args.drugbank_xml_path, parser=etree.XMLParser())
for drug in root.xpath('./*[local-name()="drug"]'):
    drugbank_id = drug.xpath('./*[local-name()="drugbank-id"][@primary="true"]')[0].text
    drugbank_id = 'DRUGBANK::'+drugbank_id
    drug_name = drug.xpath('./*[local-name()="name"]')[0].text
    drug_desc = drug.xpath('./*[local-name()="description"]')[0].text
    drug_synonyms = drug.xpath('./*[local-name()="synonyms"]')[0]
    syn_list = [s.text for s in drug_synonyms if s.text is not None]
    drug_indication = drug.xpath('./*[local-name()="indication"]')[0].text
    drug_PD = drug.xpath('./*[local-name()="pharmacodynamics"]')[0].text
    drug_MoA = drug.xpath('./*[local-name()="mechanism-of-action"]')[0].text
    drug_metabo = drug.xpath('./*[local-name()="metabolism"]')[0].text

    if drug_name is not None: texts[drugbank_id]['name'] = drug_name
    if drug_desc is not None: texts[drugbank_id]['description'] = drug_desc
    if drug_indication is not None: texts[drugbank_id]['indication'] = drug_indication
    if drug_PD is not None: texts[drugbank_id]['pharmacodynamics'] = drug_PD
    if drug_MoA is not None: texts[drugbank_id]['mechanism-of-action'] = drug_MoA
    if drug_metabo is not None: texts[drugbank_id]['metabolism'] = drug_metabo
    if len(syn_list) != 0: texts[drugbank_id]['synonyms'] = syn_list

    # Categories
    for category in drug.xpath('./*[local-name()="categories"]')[0]:
        mesh_id = category.xpath('./*[local-name()="mesh-id"]')[0].text
        if mesh_id is not None:
            mesh_id = 'MESH::'+mesh_id
            if mesh_id not in texts: continue
            add_triple(triples, 'category', drugbank_id, mesh_id)
            entities.add(drugbank_id)
            entities.add(mesh_id)

    # Proteins
    for key in ('targets', 'enzymes', 'carriers', 'transporters'):
        values = drug.xpath('./*[local-name()="' + key + '"]')[0]
        for value in values:
            be_id = value.xpath('./*[local-name()="id"]')[0].text
            polypeptide = value.xpath('./*[local-name()="polypeptide"]')
            if len(polypeptide) != 0:
                uniprot_id = polypeptide[0].get('id')
                uniprot_id = 'UNIPROT::'+uniprot_id
                add_triple(triples, key, drugbank_id, uniprot_id)
                entities.add(drugbank_id)
                entities.add(uniprot_id)

                u_name = polypeptide[0].xpath('./*[local-name()="name"]')[0].text
                u_func = polypeptide[0].xpath('./*[local-name()="specific-function"]')[0].text
                u_genename = polypeptide[0].xpath('./*[local-name()="gene-name"]')[0].text
                u_syn_list = []
                for u_syn in polypeptide[0].xpath('./*[local-name()="synonyms"]')[0]:
                    if u_syn.text is not None: u_syn_list.append(u_syn.text)
                if u_name is not None: texts[uniprot_id]['name'] = u_name
                if u_func is not None: texts[uniprot_id]['description'] = u_func
                if u_genename is not None: texts[uniprot_id]['gene-name'] = u_genename
                if len(u_syn_list) != 0: texts[uniprot_id]['synonyms'] = u_syn_list

    # ATC-code
    for atc in drug.xpath('./*[local-name()="atc-codes"]')[0]:
        atc_prev = None
        for atc_child in atc:
            atc_id = atc_child.attrib['code']
            atc_id = 'ATC::'+atc_id
            atc_name = atc_child.text
            texts[atc_id]['name'] = atc_name
            add_triple(triples, 'atc', drugbank_id, atc_id)
            if atc_prev is not None: add_triple(triples, 'atc-hierarchy', atc_prev, atc_id)
            atc_prev = atc_id
            entities.add(drugbank_id)
            entities.add(atc_id)

    # Pathway
    for pathway in drug.xpath('./*[local-name()="pathways"]')[0]:
        smpdb_id = pathway.xpath('./*[local-name()="smpdb-id"]')[0].text
        smpdb_id = 'SMPDB::'+smpdb_id
        if smpdb_id not in texts: continue
        add_triple(triples, 'drug-pathway', drugbank_id, smpdb_id)
        entities.add(drugbank_id)
        entities.add(smpdb_id)
        for enzyme in pathway.xpath('./*[local-name()="enzymes"]')[0]:
            p_uniprot_id = 'UNIPROT::'+enzyme.text
            if p_uniprot_id not in texts: continue
            add_triple(triples, 'enzyme-pathway', p_uniprot_id, smpdb_id)
            entities.add(p_uniprot_id)

    # DDI
    for int_drug in drug.xpath('./*[local-name()="drug-interactions"]')[0]:
        int_drugbank_id = int_drug.xpath('./*[local-name()="drugbank-id"]')[0].text
        int_drugbank_id = 'DRUGBANK::'+int_drugbank_id
        add_triple(triples, 'interaction', drugbank_id, int_drugbank_id)
        entities.add(drugbank_id)
        entities.add(int_drugbank_id)

for k, v in triples.items():
    print(k, len(v))

# Save textual information
if not os.path.exists(args.output_dir):
    os.makedirs(args.output_dir)
with open(os.path.join(args.output_dir, 'texts.dict'), 'wb') as f:
    pickle.dump(texts, f)
with open(os.path.join(args.output_dir, 'entities.set'), 'wb') as f:
    pickle.dump(entities, f)

# Construct KG
shuffled_triples = {}
for k, v in triples.items():
    shuffled_triples[k] = list(dict.fromkeys(v))
    random.shuffle(shuffled_triples[k])

data_split_ratio = args.data_split_ratio

train_tsv, valid_tsv, test_tsv = '', '', ''
for k, v in shuffled_triples.items():
    #n_train = len(v) // sum(data_split_ratio) * data_split_ratio[0]
    n_valid = len(v) // sum(data_split_ratio) * data_split_ratio[1]
    n_test = len(v) // sum(data_split_ratio) * data_split_ratio[2]

    if args.add_atc_hierarchy and k == 'atc-hierarchy':
        train, valid, test = v, [], []
    else:
        train, valid, test = v[:-n_valid-n_test], v[-n_valid-n_test:-n_test], v[-n_test:]
    
    for (head, tail) in train:
        train_tsv += '{}\t{}\t{}\n'.format(head, k, tail)
    for (head, tail) in valid:
        valid_tsv += '{}\t{}\t{}\n'.format(head, k, tail)
    for (head, tail) in test:
        test_tsv += '{}\t{}\t{}\n'.format(head, k, tail)

aug_tsv = ''
for entity in entities:
    type_, id_ = entity.split('::')
    texts_d = texts[entity]
    for k, v in texts_d.items():
        if k == 'name': continue
        aug_tsv += '{}\t{}\t{}::{}\n'.format(entity, k, entity, k)

with open(os.path.join(args.output_dir, 'train.tsv'), 'w') as f: f.write(train_tsv)
with open(os.path.join(args.output_dir, 'valid.tsv'), 'w') as f: f.write(valid_tsv)
with open(os.path.join(args.output_dir, 'test.tsv'), 'w') as f: f.write(test_tsv)
with open(os.path.join(args.output_dir, 'aug.tsv'), 'w') as f: f.write(aug_tsv)
