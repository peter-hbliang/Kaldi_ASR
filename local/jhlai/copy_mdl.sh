#!/bin/bash

# Copyright 2017 NCTU sl707 Jian-Hong, Lai.

if [ "$#" -ne 3 ]; then
  echo "Usage: $0 <data-dir> <final-epoch> <num-ark>"
  echo "e.g.: compute_prob.sh exp/lstm/ 20 22"
  exit 1;
fi

dir=$1
final_epoch=$2
num_ark=$3

mdl_dir=$dir/epoch-mdl

mkdir -p $mdl_dir

for x in $(seq 1 $final_epoch);
do
  iter=$(($x*$num_ark))
  cp $dir/$iter.mdl $mdl_dir/$x.mdl
done

