python3 dgl-ke/python/dglke/train.py \
  --dataset DRUGBANK \
  --data_path dataset/noaug \
  --data_file train.tsv valid.tsv test.tsv \
  --format 'raw_udd_hrt' \
  --batch_size 4096 \
  --gpu 0 \
  --mix_cpu_gpu \
  --num_epochs 1

echo "Creating BERT embeddings for (A) \"Initialization\" method and (B) \"Alignment\" method"
for text_type in name description synonyms;do
python3 src/get_bert_emb.py \
  --entities_tsv dataset/noaug/entities.tsv \
  --texts_dict dataset/noaug/texts.dict \
  --do_lower_case \
  --mode noaug \
  --text_type $text_type \
  --output_dir dataset/noaug
done


python3 dgl-ke/python/dglke/train.py \
  --dataset DRUGBANK \
  --data_path dataset/aug \
  --data_file train.tsv valid.tsv test.tsv \
  --format 'raw_udd_hrt' \
  --batch_size 4096 \
  --gpu 0 \
  --mix_cpu_gpu \
  --num_epochs 1

echo "Creating BERT embeddings for (C) \"Augmentation\" method"
python3 src/get_bert_emb.py \
  --entities_tsv dataset/aug/entities.tsv \
  --texts_dict dataset/aug/texts.dict \
  --do_lower_case \
  --mode aug \
  --output_dir dataset/aug
