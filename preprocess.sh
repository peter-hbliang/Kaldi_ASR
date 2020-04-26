#!/bin/bash
# $1 = 音檔目錄; $2 = 音檔抽特徵後之存放位置



echo "make fake wav.scp, utt2spk, spk2utt, text"
num_of_jobs=$1
wav_data=$2
test_data_dir=$3
current_path=`pwd`
if [ -d $test_data_dir ]; then
   echo "The dir '$test_data_dir' exists."
   exit
fi

mkdir $test_data_dir

for i in ${wav_data}/*
  do
  file=`basename $i`
  echo "${file} ${file}" >> $test_data_dir/utt2spk
  echo "${file} ${current_path}/${i}" >> $test_data_dir/wav.scp
  done

cp $test_data_dir/utt2spk $test_data_dir/spk2utt
cp $test_data_dir/utt2spk $test_data_dir/text

echo "make fbank & compute cmvn"
steps/make_fbank.sh --nj $num_of_jobs $test_data_dir $test_data_dir/log $test_data_dir/fbank
steps/compute_cmvn_stats.sh $test_data_dir $test_data_dir/log $test_data_dir/cmvn
echo "done!"
