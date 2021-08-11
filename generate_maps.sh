#!/bin/bash

if [ ! -d "./unlabelled_data" ]
then
  curl -f ljmanso.com/files/SocNav2_unlabelled_data.tgz --output unlabelled.tgz || {
    echo "Download failed"
    exit 1
    }
  mkdir "unlabelled_data"
  tar zxvf unlabelled.tgz -C ./unlabelled_data/
  rm unlabelled.tgz
  find unlabelled_data/SocNav2_unlabelled_data/ -name '*.*' -exec mv {} unlabelled_data/ \;
  cd unlabelled_data/ && rm -rf SocNav2_unlabelled_data/ && cd ../
fi

TEMPFILE=/tmp/$$.tmp
echo 0 > $TEMPFILE
FILES="unlabelled_data/*.json"

for f in $FILES
do
  echo "Processing $f"
  COUNTER=$[$(cat $TEMPFILE) + 1]
  echo $COUNTER > $TEMPFILE

  python3 showcase.py "best_model" "$f" 100
  cp "$f" images_dataset/

  if [ $COUNTER -eq 10 ]
  then
    echo 0 > $TEMPFILE
    wait
  fi
done
unlink $TEMPFILE

