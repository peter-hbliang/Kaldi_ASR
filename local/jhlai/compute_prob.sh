#!/bin/bash

# Copyright 2017 NCTU sl707 Jian-Hong, Lai.

stage=-2
cmd=run.pl

if [ "$#" -ne 2 ]; then
  echo "Usage: $0 <data-dir> <final-epoch>"
  echo "e.g.: compute_prob.sh exp/lstm/ 20"
  exit 1;
fi

dir=$1
final_epoch=$2

num_iter=$(($final_epoch/5))
num_remain=$(($final_epoch%5))
mdl_dir=$dir/epoch-mdl
egs_dir=$dir/egs

if [ $stage -le -2 ]
then
  echo "Total iterations: $num_iter"
  for x in $(seq 1 $num_iter);
  do
    echo "Computing train/valid prob."
    echo "Iter: $x/$num_iter"
    for y in $(seq 1 5)
    do
      temp=$((($x-1)*5))
      epoch=$(($temp+$y))
      $cmd $dir/log/compute_prob_valid_epoch.$epoch.log \
        nnet3-compute-prob "nnet3-am-copy --raw=true $mdl_dir/$epoch.mdl - |" \
          "ark:nnet3-merge-egs ark:$egs_dir/valid_diagnostic.egs ark:- |" &
      $cmd $dir/log/compute_prob_train_epoch.$epoch.log \
        nnet3-compute-prob "nnet3-am-copy --raw=true $mdl_dir/$epoch.mdl - |" \
          "ark:nnet3-merge-egs ark:$egs_dir/train_diagnostic.egs ark:- |" &
    done
    wait
  done
fi

if [ $stage -le -1 ]
then
  if [ $num_remain -ne 0 ]
  then
    x=$(($num_iter+1))
    
    for y in $(seq 1 $num_remain)
    do
      temp=$((($x-1)*5))
      epoch=$(($temp+$y))
      $cmd $dir/log/compute_prob_valid_epoch.$epoch.log \
        nnet3-compute-prob "nnet3-am-copy --raw=true $mdl_dir/$epoch.mdl - |" \
          "ark:nnet3-merge-egs ark:$egs_dir/valid_diagnostic.egs ark:- |" &
      $cmd $dir/log/compute_prob_train_epoch.$epoch.log \
        nnet3-compute-prob "nnet3-am-copy --raw=true $mdl_dir/$epoch.mdl - |" \
          "ark:nnet3-merge-egs ark:$egs_dir/train_diagnostic.egs ark:- |" &
      wait
    done
  fi
fi

