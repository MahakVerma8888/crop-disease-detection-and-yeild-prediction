#!/bin/sh
# Fetch PlantVillage subset in chunks, repacking between chunks (many tiny
# packs make git progressively slower).
REPO=/home/claude/data_raw/pv; TSV=/home/claude/data_raw/oid_path.tsv; OUT=/home/claude/data_raw/img
N=$(wc -l < $TSV); i=0
while [ $i -lt $N ]; do
  python3 -I /home/claude/crop-ai/scripts/fetch_images.py $REPO $TSV $OUT $i $((i+400)) 6
  git -C $REPO repack -a -d -q
  i=$((i+400))
done
echo ALLDONE
