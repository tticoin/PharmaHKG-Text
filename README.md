# PharmaHKG-Text
The implementation of Representing a Heterogeneous Pharmaceutical Knowledge-Graph with Textual Information

## Usage
### Constructing dataset
Data files of DrugBank, UniProt, MeSH and SMPDB can be freely donwloaded from the official website. (You need to create an account to download DrugBank file.)
```
export DRUGBANK_DATA=(DrugBank XML file path)
export UNIPROT_DATA=(UniProt Swiss-Prot XML file path)
export MESH_DATA=(MeSH description XML file path)
export SMPDB_DATA=(SMPDB TSV file path)

sh construct_kg.sh
```

### Preparing BERT embeddings
```
sh prepare_bert_embs.sh
```

### Link prediction with textual embeddings
```
cd dgl-ke/python/dglke

python3 train.py \
  --model_name SimplE \
  --dataset PharmaHKG \
  --data_path ../../../dataset/noaug \
  --data_file train.tsv valid.tsv test.tsv \
  --format 'raw_udd_hrt' \
  --batch_size 4096 \
  --log_interval 1000 \
  --neg_sample_size 1 \
  --hidden_dim 768 \
  --lr 0.25 \
  --neg_sample_size_eval -1 \
  --negative_sampling_for_hetero \
  --loss_genre Logistic \
  --gpu 0 \
  --mix_cpu_gpu \
  --num_epochs 100 \
  --valid \
  --eval_interval_epoch 100 \
  --save_path OUTPUT_DIR \
  --entity_emb_file noaug_name \
  --async_update
```
When you use the (B) Alignemnt method, please add ```--do_alignment```.
When using the (C) Augmentation method, please set ```--data_path``` to ```dataset/aug``` and set the ```---entity_emb_file``` to ```aug```.


## Acknowledgement
This work was supported by JSPS KAKENHI Grant Number 20k11962.
