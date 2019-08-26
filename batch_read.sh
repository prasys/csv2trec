#!/bin/bash

EXT=tsv
for i in *; do
    if [ "${i}" != "${i%.${EXT}}" ];then
    	python csv2xml.py $i
        echo "Export $i to TREC compatible files as results"
    fi
done