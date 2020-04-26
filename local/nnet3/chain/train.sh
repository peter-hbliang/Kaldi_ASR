#!/bin/bash


# This was modified from run_tdnn_1a.sh; it's the
# first attempt at a setup including convolutional components.


# steps/info/chain_dir_info.pl exp/chain/cnn_tdnn1a_sp
# exp/chain/cnn_tdnn1a_sp: num-iters=102 nj=2..5 num-params=5.5M dim=40+100->2889 combine=-0.062->-0.058 xent:train/valid[67,101,final]=(-1.00,-1.09,-0.913/-1.08,-1.16,-1.05) logprob:train/valid[67,101,final]=(-0.055,-0.058,-0.047/-0.071,-0.077,-0.074)

# The following table compares chain (TDNN+LSTM, TDNN, CNN+TDNN).
# The CNN+TDNN doesn't seem to have any advantages versus the TDNN (and it's
# about 5 times slower per iteration).  But it's not well tuned.
# And the num-params is fewer (5.5M vs 7.6M for TDNN).

# local/chain/compare_wer.sh exp/chain/tdnn_lstm1a_sp exp/chain/tdnn1a_sp exp/chain/cnn_tdnn1a_sp
# System                tdnn_lstm1a_sp tdnn1a_sp cnn_tdnn1a_sp
#WER dev93 (tgpr)                7.48      7.87      9.02
#WER dev93 (tg)                  7.41      7.61      8.60
#WER dev93 (big-dict,tgpr)       5.64      5.71      6.97
#WER dev93 (big-dict,fg)         5.40      5.10      6.12
#WER eval92 (tgpr)               5.67      5.23      5.56
#WER eval92 (tg)                 5.46      4.87      5.05
#WER eval92 (big-dict,tgpr)      3.69      3.24      3.40
#WER eval92 (big-dict,fg)        3.28      2.71      2.73
# Final train prob        -0.0341   -0.0414   -0.0532
# Final valid prob        -0.0506   -0.0634   -0.0752
# Final train prob (xent)   -0.5643   -0.8216   -1.0857
# Final valid prob (xent)   -0.6648   -0.9208   -1.1505



set -e -o pipefail

# First the options that are passed through to run_ivector_common.sh
# (some of which are also used in this script directly).
train_cmd="run.pl"
stage=16  # stage 16 for chain training
nj=10
train_set="tcc300_ner_aishell_sp/train"
test_set="fbank/test"
gmm="gmm_2018_sp/tri4a"        # this is the source gmm-dir that we'll use for alignments; it
                             # should have alignments for the specified training data.
num_threads_ubm=32
nnet3_affix=       # affix for exp dirs, e.g. it was _cleaned in tedlium.

# Options which are not passed through to run_ivector_common.sh
affix=1a  #affix for TDNN+LSTM directory e.g. "1a" or "1b", in case we change the configuration.
common_egs_dir=
reporting_email=

# LSTM/chain options
train_stage=6  #-10 for full training; -2 for skip egs
xent_regularize=0.1

# training chunk-options
chunk_width=150
# we don't need extra left/right context for TDNN systems.
chunk_left_context=40
chunk_right_context=0

# training options
srand=0
remove_egs=false

#decode options
test_online_decoding=false  # if true, it will run the last decoding stage.

# End configuration section.
echo "$0 $@"  # Print the command line for logging


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
ali_dir=exp/${gmm}-ali-tccnerai-sp
lat_dir=exp/${gmm}-lat-tccnerai-sp
dir=exp/chain-cldnn2-tccnerai-sp2
train_data_dir=data/${train_set}
# train_ivector_dir=exp/nnet3${nnet3_affix}/ivectors_${train_set}_sp_hires
lores_train_data_dir=data/mfcc/train

# note: you don't necessarily have to change the treedir name
# each time you do a new experiment-- only if you change the
# configuration in a way that affects the tree.
tree_dir=data/tree-chain-tccner
# the 'lang' directory is created by this script.
# If you create such a directory with a non-standard topology
# you should probably name it differently.
lang=data/lang-chain

for f in $train_data_dir/feats.scp \
    $lores_train_data_dir/feats.scp $gmm_dir/final.mdl \
    $ali_dir/ali.1.gz $gmm_dir/final.mdl; do
  [ ! -f $f ] && echo "$0: expected file $f to exist" && exit 1
done


if [ $stage -le 12 ]; then
  echo "$0: creating lang directory $lang with chain-type topology"
  # Create a version of the lang/ directory that has one state per phone in the
  # topo file. [note, it really has two states.. the first one is only repeated
  # once, the second one has zero or more repeats.]
  if [ -d $lang ]; then
    if [ $lang/L.fst -nt data/lang/L.fst ]; then
      echo "$0: $lang already exists, not overwriting it; continuing"
    else
      echo "$0: $lang already exists and seems to be older than data/lang..."
      echo " ... not sure what to do.  Exiting."
      exit 1;
    fi
  else
    cp -r data/lang $lang
    silphonelist=$(cat $lang/phones/silence.csl) || exit 1;
    nonsilphonelist=$(cat $lang/phones/nonsilence.csl) || exit 1;
    # Use our special topology... note that later on may have to tune this
    # topology.
    steps/nnet3/chain/gen_topo.py $nonsilphonelist $silphonelist >$lang/topo
  fi
fi

if [ $stage -le 13 ]; then
  # Get the alignments as lattices (gives the chain training more freedom).
  # use the same num-jobs as the alignments
  steps/align_fmllr_lats.sh --nj 10 --cmd run.pl ${lores_train_data_dir} \
    data/lang $gmm_dir $lat_dir
  rm $lat_dir/fsts.*.gz # save space
fi

if [ $stage -le 14 ]; then
  # Build a tree using our new topology.  We know we have alignments for the
  # speed-perturbed data (local/nnet3/run_ivector_common.sh made them), so use
  # those.  The num-leaves is always somewhat less than the num-leaves from
  # the GMM baseline.
   if [ -f $tree_dir/final.mdl ]; then
     echo "$0: $tree_dir/final.mdl already exists, refusing to overwrite it."
     exit 1;
  fi
  steps/nnet3/chain/build_tree.sh \
    --frame-subsampling-factor 3 \
    --context-opts "--context-width=3 --central-position=1" \
    --cmd "$train_cmd" 3500 ${lores_train_data_dir} \
    $lang $ali_dir $tree_dir
fi


if [ $stage -le 15 ]; then
  mkdir -p $dir
  echo "$0: creating neural net configs using the xconfig parser";

  num_targets=$(tree-info $tree_dir/tree |grep num-pdfs|awk '{print $2}')
  learning_rate_factor=$(echo "print 0.5/$xent_regularize" | python2.7)

  mkdir -p $dir/configs
  cat <<EOF > $dir/configs/network.xconfig
  input dim=40 name=input

  fixed-affine-layer name=lda input=Append(-2,-1,0,1,2) affine-transform-file=exp/chain-cldnn2/configs/lda.mat
  
  conv-layer name=cnn1 input=Append(input@-4,input@-3,input@-2,input@-1,input@0,input@1,input@2,input@3,input@4) height-in=40 height-out=20 height-subsample-out=2 time-offsets=-2,-1,0,1,2 height-offsets=-4,-3,-2,-1,0,1,2,3,4 num-filters-out=128 max-change=0.25
  conv-layer name=cnn2 input=cnn1 height-in=20 height-out=20 time-offsets=-1,0,1 height-offsets=-1,0,1 num-filters-out=256
  relu-batchnorm-layer name=affine input=Append(cnn2,lda) dim=256

  fast-lstmp-layer name=lstm1 cell-dim=512 recurrent-projection-dim=256 non-recurrent-projection-dim=256 decay-time=20 delay=-3
  fast-lstmp-layer name=lstm2 cell-dim=512 recurrent-projection-dim=256 non-recurrent-projection-dim=256 decay-time=20 delay=-3
  fast-lstmp-layer name=lstm3 cell-dim=512 recurrent-projection-dim=256 non-recurrent-projection-dim=256 decay-time=20 delay=-3

  relu-batchnorm-layer name=dnn1 dim=2048
  relu-batchnorm-layer name=dnn2 dim=2048

  ## adding the layers for chain branch
  relu-batchnorm-layer name=prefinal-chain dim=512 target-rms=0.5
  output-layer name=output include-log-softmax=false dim=$num_targets max-change=1.5

  # adding the layers for xent branch
  # This block prints the configs for a separate output that will be
  # trained with a cross-entropy objective in the 'chain' models... this
  # has the effect of regularizing the hidden parts of the model.  we use
  # 0.5 / args.xent_regularize as the learning rate factor- the factor of
  # 0.5 / args.xent_regularize is suitable as it means the xent
  # final-layer learns at a rate independent of the regularization
  # constant; and the 0.5 was tuned so as to make the relative progress
  # similar in the xent and regular final layers.
  relu-batchnorm-layer name=prefinal-xent input=dnn2 dim=512 target-rms=0.5
  output-layer name=output-xent dim=$num_targets learning-rate-factor=$learning_rate_factor max-change=1.5
EOF
  steps/nnet3/xconfig_to_configs.py --xconfig-file $dir/configs/network.xconfig --config-dir $dir/configs/
fi


if [ $stage -le 16 ]; then
  if [[ $(hostname -f) == *.clsp.jhu.edu ]] && [ ! -d $dir/egs/storage ]; then
    utils/create_split_dir.pl \
     /export/b0{3,4,5,6}/$USER/kaldi-data/egs/tedlium-$(date +'%m_%d_%H_%M')/s5_r2/$dir/egs/storage $dir/egs/storage
  fi

  time steps/nnet3/chain/train.py --stage=$train_stage \
    --cmd="run.pl" \
    --feat.online-ivector-dir=$train_ivector_dir \
    --feat.cmvn-opts="--norm-means=false --norm-vars=false" \
    --chain.xent-regularize $xent_regularize \
    --chain.leaky-hmm-coefficient=0.1 \
    --chain.l2-regularize=0.00005 \
    --chain.apply-deriv-weights=false \
    --chain.lm-opts="--num-extra-lm-states=2000" \
    --trainer.srand=$srand \
    --trainer.max-param-change=2.0 \
    --trainer.num-epochs=4 \
    --trainer.frames-per-iter=1500000 \
    --trainer.optimization.num-jobs-initial=1 \
    --trainer.optimization.num-jobs-final=1 \
    --trainer.optimization.initial-effective-lrate=0.001 \
    --trainer.optimization.final-effective-lrate=0.0001 \
    --trainer.optimization.shrink-value=1.0 \
    --trainer.num-chunk-per-minibatch=128 \
    --trainer.optimization.momentum=0.0 \
    --egs.chunk-width=$chunk_width \
    --egs.chunk-left-context=$chunk_left_context \
    --egs.chunk-right-context=$chunk_right_context \
    --egs.chunk-left-context-initial=0 \
    --egs.chunk-right-context-final=0 \
    --egs.dir="$common_egs_dir" \
    --egs.opts="--frames-overlap-per-eg 0" \
    --cleanup.remove-egs=$remove_egs \
    --use-gpu=true \
    --reporting.email="$reporting_email" \
    --feat-dir=$train_data_dir \
    --tree-dir=$tree_dir \
    --lat-dir=$lat_dir \
    --dir=$dir  || exit 1;
fi

# decode part
if [ $stage -le 17 ]; then
  test_data_dir=data/${test_set}
  frames_per_chunk=$(echo $chunk_width | cut -d, -f1)
  utils/mkgraph.sh --self-loop-scale 1.0 $lang $tree_dir $dir/graph
  time steps/nnet3/decode.sh \
    --acwt 1.0 --post-decode-acwt 10.0 \
    --extra-left-context $chunk_left_context \
    --extra-right-context $chunk_right_context \
    --extra-left-context-initial 0 \
    --extra-right-context-final 0 \
    --frames-per-chunk $frames_per_chunk \
    --nj $nj $dir/graph $test_data_dir $dir/decode-final-tcc300
fi

exit 0;
