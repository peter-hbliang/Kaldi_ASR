#!/bin/bash
num_job=$1
data=$2
output=$3
output_lm=$4

./steps/nnet3/decode_noweight.sh --beam 8.0 --lattice_beam 15.0 \
				 --acwt 0.83 --post-decode-acwt 10.0 \
				 --extra-left-context 0 --extra-right-context 0 \
				 --extra-left-context-initial 0 --extra-right-context-final 0 \
				 --frames-per-chunk 150 \
				 --nj $num_job \
				 HCLG-120k-order2/ \
				 $data \
				 exp/chain-tdnnf/$output

./steps/lmrescore_const_arpa3.sh LM_carpa/prune_1e-16.carpa \
				 LM_carpa/fg_120k.carpa \
				 $data \
				 exp/chain-tdnnf/$output \
				 exp/chain-tdnnf/$output_lm

mkdir -p result/$output_lm
cp -r exp/chain-tdnnf/$output_lm/scoring_kaldi/penalty_0.0/*.txt result/$output_lm/
