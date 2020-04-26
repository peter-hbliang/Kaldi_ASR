#!/bin/bash

# Copyright 2017 NCTU sl707 Jian-Hong, Lai.

if [ "$#" -ne 3 ]; then
  echo "Usage: $0 <data-dir> <num-ark> <final-epoch>"
  echo "e.g.: compute_prob.sh exp/lstm/log 22 20"
  exit 1;
fi

dir=$1
num_ark=$2
final_epoch=$3

log_dir=$dir/log

# Chech directory.
[ ! -d $log_dir ] && echo "$0: no such directory $log_dir" && exit 1;
[ -f $log_dir/train_valid_prob.csv ] && rm $log_dir/train_valid_prob.csv;

final_iter=$(($num_ark*$final_epoch-1))
for x in $(seq 1 $final_iter);
do
  train="compute_prob_train"
  valid="compute_prob_valid"
  temp=".$x"
  temp2=".log"
  log=$log_dir/$train$temp$temp2
  log2=$log_dir/$valid$temp$temp2
  log_grep=$(grep "'output'" $log | cut -d " " -f 12)
  log_grep2=$(grep "'output'" $log2 | cut -d " " -f 12)
  echo "$x,$log_grep,$log_grep2" >> $log_dir/train_valid_prob.csv
done

