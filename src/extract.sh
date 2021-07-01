python3 extract_triples.py \
  --drugbank_xml_path /data/asada.13003/full_database.xml \
  --uniprot_xml_path /data/asada.13003/for_docker/database/uniprot_sprot.xml \
  --mesh_xml_path /data/asada.13003/for_docker/database/MESH/desc2021.xml \
  --smpdb_tsv_path /data/asada.13003/for_docker/database/smpdb_pathways.csv \
  --add_atc_hierarchy \
  --data_split_ratio 90 5 5 \
  --seed 0 \
  --output_dir ~/fin_data/
