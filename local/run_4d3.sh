#!/bin/bash


# run_4d3.sh is as run_4d.sh, but using a newer version of the scripts that
# dump the egs with several frames of labels.


train_stage=-10
use_gpu=true

stage=1

ini_learning=$1
final_learning=$2
p_input=$3
p_output=$4
num_hidden=$5
num_epoch=$6
dir=exp/nnet2-hid${num_hidden}-lr${ini_learning}_${final_learning}-batch512-p_${p_input}_${p_output}-tri2b34hrs_epo${num_epoch}

. cmd.sh
. ./path.sh
. utils/parse_options.sh


if $use_gpu; then
  if ! cuda-compiled; then
    cat <<EOF && exit 1 
This script is intended to be used with GPUs but you have not compiled Kaldi with CUDA 
If you want to use GPUs (and have them), go to src/, and configure and make on a machine
where "nvcc" is installed.
EOF
  fi
  parallel_opts="-l gpu=1" 
  num_threads=1
  minibatch_size=512
else
  num_threads=16
  minibatch_size=128
  parallel_opts="-pe smp $num_threads" 
fi


if [ $stage -eq 1 ]; then
 #if true || [ ! -f $dir/final.mdl ]; then
  steps/nnet2/train_pnorm_simple2.sh --stage $train_stage \
     --samples-per-iter 400000 \
     --num-threads "$num_threads" \
     --minibatch-size "$minibatch_size" \
     --parallel-opts "$parallel_opts" \
     --num-jobs-nnet 4 \
     --num-epochs $num_epoch --add-layers-period 2 \
     --num-hidden-layers $num_hidden \
     --mix-up 0 \
     --initial-learning-rate $ini_learning --final-learning-rate $final_learning \
     --cmd "$decode_cmd" \
     --pnorm-input-dim $p_input \
     --pnorm-output-dim $p_output \
     data/train data/lang exp/tri2b/1800 $dir  || exit 1;
  #default: epoch=21 p-input=5000 p-output=500
 #fi
echo "nnet2 training success"
./steps/nnet2/decode.sh --config conf/decode.config --cmd run.pl --nj 8 exp/tri2b/1800/graph/ data/test/ $dir/decode final
./steps/nnet2/decode.sh --config conf/decode.config --cmd run.pl --nj 8 exp/tri2b/1800/graph/ data/test/ $dir/decode_120 120
./steps/nnet2/decode.sh --config conf/decode.config --cmd run.pl --nj 8 exp/tri2b/1800/graph/ data/test/ $dir/decode_112 112
./steps/nnet2/decode.sh --config conf/decode.config --cmd run.pl --nj 8 exp/tri2b/1800/graph/ data/test/ $dir/decode_104 104
./steps/nnet2/decode.sh --config conf/decode.config --cmd run.pl --nj 8 exp/tri2b/1800/graph/ data/test/ $dir/decode_96 96
./steps/nnet2/decode.sh --config conf/decode.config --cmd run.pl --nj 8 exp/tri2b/1800/graph/ data/test/ $dir/decode_88 88
./steps/nnet2/decode.sh --config conf/decode.config --cmd run.pl --nj 8 exp/tri2b/1800/graph/ data/test/ $dir/decode_80 80
#./../s5/local/decode_nnet2.sh exp/tri2b_mmi/1800_re_20-10/graph_80k/ data/test_utt/ $dir/decode_beam17-lm13-max9000-final
#./../s5/local/decode_nnet2.sh exp/tri2b_mmi/1800_re_20-10/graph_80k/ ../COSPRO $dir/decode_beam17-lm13-max9000-final
exit 1
fi

if [ $stage -eq 2 ]; then
steps/nnet2/decode.sh --config conf/decode.config --cmd "$decode_cmd" --nj 20 \
  --transform-dir exp/tri3b/decode \
  exp/tri3b/graph data/test $dir/decode  &

steps/nnet2/decode.sh --config conf/decode.config --cmd "$decode_cmd" --nj 20 \
  --transform-dir exp/tri3b/decode_ug \
  exp/tri3b/graph_ug data/test $dir/decode_ug

wait
fi
