dir=~/fin_data/noaug/
for f in train.tsv valid.tsv test.tsv;do
    mv ${dir}$f ${dir}_$f
    cat ${dir}_$f | grep -v DB06517 | grep -v DB15351 | grep -v DB05697 > ${dir}__$f
    cat ${dir}__${f} | sed -e "s/drug-pathway/pathway/g" | sed -e "s/enzyme-pathway/pathway/g" > ${dir}${f}
    rm ${dir}_$f ${dir}__$f
done
