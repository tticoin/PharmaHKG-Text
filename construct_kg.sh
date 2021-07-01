mkdir -p dataset/noaug

echo "Constructing a dataset from KGs. It takes a while ..." 
python3 src/extract_triples.py \
  --drugbank_xml_path $DRUGBANK_DATA \
  --uniprot_xml_path $UNIPROT_DATA \
  --mesh_xml_path $MESH_DATA \
  --smpdb_tsv_path $SMPDB_DATA \
  --add_atc_hierarchy \
  --data_split_ratio 90 5 5 \
  --seed 0 \
  --output_dir dataset/noaug
echo "Extracting relation triples is finished !"

dir=dataset/noaug/
for f in train.tsv valid.tsv test.tsv;do
    mv ${dir}$f ${dir}_$f
    cat ${dir}_$f | grep -v DB06517 | grep -v DB15351 | grep -v DB05697 > ${dir}__$f
    cat ${dir}__${f} | sed -e "s/drug-pathway/pathway/g" | sed -e "s/enzyme-pathway/pathway/g" > ${dir}${f}
    rm ${dir}_$f ${dir}__$f
done

cp -r dataset/noaug dataset/aug
mv dataset/aug/train.tsv dataset/aug/_
cat dataset/aug/_ dataset/aug/aug.tsv > dataset/aug/train.tsv
rm dataset/aug/_

