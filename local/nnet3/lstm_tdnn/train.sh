#!/bin/bash


# run_tdnn_lstm_1a.sh is a TDNN+LSTM system.  Compare with the TDNN
# system in run_tdnn_1a.sh.  Configuration is similar to
# the same-named script run_tdnn_lstm_1a.sh in
# egs/tedlium/s5_r2/local/nnet3/tuning.

# It's a little better than the TDNN-only script on dev93, a little
# worse on eval92.

# steps/info/nnet3_dir_info.pl exp/nnet3/tdnn_lstm1a_sp
# exp/nnet3/tdnn_lstm1a_sp: num-iters=102 nj=3..10 num-params=8.8M dim=40+100->3413 combine=-0.55->-0.54 loglike:train/valid[67,101,combined]=(-0.63,-0.55,-0.55/-0.71,-0.63,-0.63) accuracy:train/valid[67,101,combined]=(0.80,0.82,0.82/0.76,0.78,0.78)



# local/nnet3/compare_wer.sh --looped --online exp/nnet3/tdnn1a_sp exp/nnet3/tdnn_lstm1a_sp 2>/dev/null
# local/nnet3/compare_wer.sh --looped --online exp/nnet3/tdnn1a_sp exp/nnet3/tdnn_lstm1a_sp
# System                tdnn1a_sp tdnn_lstm1a_sp
#WER dev93 (tgpr)                9.18      8.54
#             [looped:]                    8.54
#             [online:]                    8.57
#WER dev93 (tg)                  8.59      8.25
#             [looped:]                    8.21
#             [online:]                    8.34
#WER dev93 (big-dict,tgpr)       6.45      6.24
#             [looped:]                    6.28
#             [online:]                    6.40
#WER dev93 (big-dict,fg)         5.83      5.70
#             [looped:]                    5.70
#             [online:]                    5.77
#WER eval92 (tgpr)               6.15      6.52
#             [looped:]                    6.45
#             [online:]                    6.56
#WER eval92 (tg)                 5.55      6.13
#             [looped:]                    6.08
#             [online:]                    6.24
#WER eval92 (big-dict,tgpr)      3.58      3.88
#             [looped:]                    3.93
#             [online:]                    3.88
#WER eval92 (big-dict,fg)        2.98      3.38
#             [looped:]                    3.47
#             [online:]                    3.53
# Final train prob        -0.7200   -0.5492
# Final valid prob        -0.8834   -0.6343
# Final train acc          0.7762    0.8154
# Final valid acc          0.7301    0.7849


set -e -o pipefail

# First the options that are passed through to run_ivector_common.sh
# (some of which are also used in this script directly).
stage=13
nj=10
train_set="tcc300_aishell/train"
test_set="fbank/test"
gmm=tri2b-meng        # this is the source gmm-dir that we'll use for alignments; it
                 # should have alignments for the specified training data.
num_threads_ubm=32
nnet3_affix=       # affix for exp dirs, e.g. it was _cleaned in tedlium.

# Options which are not passed through to run_ivector_common.sh
affix=1a  #affix for TDNN+LSTM directory e.g. "1a" or "1b", in case we change the configuration.
common_egs_dir=
reporting_email=

# LSTM options
train_stage=-3
label_delay=5

# training chunk-options
chunk_width=20
chunk_left_context=40
chunk_right_context=0

# training options
srand=0
remove_egs=false

#decode options
test_online_decoding=true  # if true, it will run the last decoding stage.

. ./cmd.sh
. ./path.sh
. ./utils/parse_options.sh

if ! cuda-compiled; then
  cat <<EOF && exit 1
This script is intended to be used with GPUs but you have not compiled Kaldi with CUDA
If you want to use GPUs (and have them), go to src/, and configure and make on a machine
where "nvcc" is installed.
EOF
fi

#local/nnet3/run_ivector_common.sh \
#  --stage $stage --nj $nj \
#  --train-set $train_set --gmm $gmm \
#  --num-threads-ubm $num_threads_ubm \
#  --nnet3-affix "$nnet3_affix"



gmm_dir=exp/${gmm}
ali_dir=exp/${gmm}-ali-tccai
lang=data/lang
dir=exp/lstm-tdnn-tccai-epoch5
train_data_dir=data/${train_set}
# train_ivector_dir=exp/nnet3${nnet3_affix}/ivectors_${train_set}_sp_hires

#for f in $train_data_dir/feats.scp $train_ivector_dir/ivector_online.scp \
for f in $train_data_dir/feats.scp \
    $gmm_dir/graph-ann/HCLG.fst \
    $ali_dir/ali.1.gz $gmm_dir/final.mdl; do
  [ ! -f $f ] && echo "$0: expected file $f to exist" && exit 1
done

if [ $stage -le 12 ]; then
  mkdir -p $dir
  echo "$0: creating neural net configs using the xconfig parser";

  num_targets=$(tree-info $ali_dir/tree |grep num-pdfs|awk '{print $2}')

  mkdir -p $dir/configs
  cat <<EOF > $dir/configs/network.xconfig
 
  input dim=40 name=input
  fixed-affine-layer name=lda input=Append(-1,0,1) affine-transform-file=exp/cldnn2-new/configs/lda.mat
  idct-layer name=idct input=input dim=40 cepstral-lifter=22 affine-transform-file=exp/cldnn2-new/configs/idct.mat

  conv-relu-batchnorm-layer name=cnn1 input=idct height-in=40 height-out=20 height-subsample-out=2 time-offsets=-1,0,1 height-offsets=-1,0,1 num-filters-out=256 max-change=0.25
  conv-relu-batchnorm-layer name=cnn2 input=cnn1 height-in=20 height-out=20 time-offsets=-1,0,1 height-offsets=-1,0,1 num-filters-out=128
  
  relu-batchnorm-layer name=affine1 input=lda dim=512

  relu-batchnorm-layer name=tdnn1 input=cnn2 dim=512
  relu-batchnorm-layer name=tdnn2 input=Append(-1,0,1,affine1) dim=512
  relu-batchnorm-layer name=tdnn3 input=Append(-1,0,1) dim=512

  fast-lstmp-layer name=lstm1 cell-dim=512 recurrent-projection-dim=256 non-recurrent-projection-dim=256 delay=-3 decay-time=20
  relu-batchnorm-layer name=tdnn4 input=Append(-3,0,3) dim=512
  relu-batchnorm-layer name=tdnn5 input=Append(-3,0,3) dim=512
  relu-batchnorm-layer name=tdnn6 input=Append(-3,0,3) dim=512
  fast-lstmp-layer name=lstm2 cell-dim=512 recurrent-projection-dim=256 non-recurrent-projection-dim=256 delay=-3 decay-time=20
  relu-batchnorm-layer name=tdnn7 input=Append(-3,0,3) dim=512
  relu-batchnorm-layer name=tdnn8 input=Append(-3,0,3) dim=512
  relu-batchnorm-layer name=tdnn9 input=Append(-3,0,3) dim=512
  fast-lstmp-layer name=lstm3 cell-dim=512 recurrent-projection-dim=256 non-recurrent-projection-dim=256 delay=-3 decay-time=20

  output-layer name=output-xent input=lstm3 output-delay=5 dim=1569 max-change=1.5


EOF
  steps/nnet3/xconfig_to_configs.py --xconfig-file $dir/configs/network.xconfig --config-dir $dir/configs/
fi


if [ $stage -le 13 ]; then
  if [[ $(hostname -f) == *.clsp.jhu.edu ]] && [ ! -d $dir/egs/storage ]; then
    utils/create_split_dir.pl \
     /export/b0{3,4,5,6}/$USER/kaldi-data/egs/tedlium-$(date +'%m_%d_%H_%M')/s5_r2/$dir/egs/storage $dir/egs/storage
  fi

 time steps/nnet3/train_rnn.py --stage=$train_stage \
    --cmd="run.pl" \
    --feat.cmvn-opts="--norm-means=false --norm-vars=false" \
    --trainer.srand=$srand \
    --trainer.max-param-change=2.0 \
    --trainer.num-epochs=5 \
    --trainer.deriv-truncate-margin=10 \
    --trainer.samples-per-iter=20000 \
    --trainer.optimization.num-jobs-initial=1 \
    --trainer.optimization.num-jobs-final=1 \
    --trainer.optimization.initial-effective-lrate=0.01 \
    --trainer.optimization.final-effective-lrate=0.001 \
    --trainer.optimization.shrink-value 0.99 \
    --trainer.rnn.num-chunk-per-minibatch=100 \
    --trainer.optimization.momentum=0.5 \
    --egs.chunk-width=$chunk_width \
    --egs.chunk-left-context=$chunk_left_context \
    --egs.chunk-right-context=$chunk_right_context \
    --egs.chunk-left-context-initial=0 \
    --egs.chunk-right-context-final=0 \
    --egs.dir="$common_egs_dir" \
    --cleanup.remove-egs=$remove_egs \
    --use-gpu=true \
    --feat-dir=$train_data_dir \
    --ali-dir=$ali_dir \
    --lang=$lang \
    --reporting.email="$reporting_email" \
    --dir=$dir  || exit 1;
fi

# decode part
test_data_dir=data/${test_set}
if [ $stage -le 14 ]; then
  utils/mkgraph.sh $lang $dir $dir/graph
  time steps/nnet3/decode_looped.sh --nj $nj $dir/graph $test_data_dir $dir/decode-final-tcc300-loop
fi

exit 0;
