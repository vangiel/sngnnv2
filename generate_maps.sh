#!/bin/bash

if [ ! -d "./unlabelled_data" ]
then
  curl -f ljmanso.com/files/SocNav2_unlabelled_data.tgz --output unlabelled.tgz || {
    echo "Download failed"
    exit 1
    }
  mkdir "unlabelled_data"
  tar zxvf unlabelled.tgz -C ./unlabelled_data
  rm unlabelled.tgz
fi

python3 showcase.py "best_model" "unlabelled_data/SocNav2_unlabelled_data/A000000.json" 5