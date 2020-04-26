#!/bin/bash
# Copyright 2012  Johns Hopkins University (Author: Daniel Povey).  Apache 2.0.

# MMI training (or optionally boosted MMI, if you give the --boost option).
# 4 iterations (by default) of Extended Baum-Welch update.
# after 4 iterations, we regenerate lattices and alignments
# Re-train aforementioned process 4 times (total 16 iterations)
#
# For the numerator we have a fixed alignment rather than a lattice--
# this actually follows from the way lattices are defined in Kaldi, which
# is to have a single path for each word (output-symbol) sequence.

# Begin configuration section.
cmd=run.pl
num_iters=4
regenerate_iters="0 1 2 3 5 7 9 11 13 15 17 19";
boost=0.0
cancel=true # if true, cancel num and den counts on each frame.
drop_frames=false # if true, ignore stats from frames where num + den
                       # have no overlap. 
tau=400
weight_tau=10
acwt=0.1
stage=0
beam=50.0
ali_beam=50.0
ali_retry_beam=100.0
lattice_beam=10.0
max_active=5000
max_mem=20000000 # This will stop the processes getting too large.
scale_opts="--transition-scale=1.0 --acoustic-scale=0.1 --self-loop-scale=0.1"
boost_silence=0.2
# End configuration section

echo "$0 $@"  # Print the command line for logging

[ -f ./path.sh ] && . ./path.sh; # source the path.
. parse_options.sh || exit 1;

if [ $# -ne 6 ]; then
  echo "Usage: steps/train_mmi.sh <data> <lang> <tri2b tree> <ali> <denlats> <exp>"
  echo " e.g.: steps/train_mmi.sh data/train_si84 data/lang exp/tri2b exp/tri2b_ali_si84 exp/tri2b_denlats_si84 exp/tri2b_mmi "
  echo "Main options (for others, see top of script file)"
  echo "  --boost <boost-weight>                           # (e.g. 0.1), for boosted MMI.  (default 0)"
  echo "  --cancel (true|false)                            # cancel stats (true by default)"
  echo "  --cmd (utils/run.pl|utils/queue.pl <queue opts>) # how to run jobs."
  echo "  --config <config-file>                           # config containing options"
  echo "  --stage <stage>                                  # stage to do partial re-run from."
  echo "  --tau                                            # tau for i-smooth to last iter (default 200)"
  
  exit 1;
fi

data=$1
lang=$2
tri2bdir=$3
alidir=$4
denlatdir=$5
dir=$6
mkdir -p $dir/log

for f in $data/feats.scp $alidir/{tree,final.mdl,ali.1.gz} $denlatdir/lat.1.gz; do
  [ ! -f $f ] && echo "$0: no such file $f" && exit 1;
done
nj=`cat $alidir/num_jobs` || exit 1;
[ "$nj" -ne "`cat $denlatdir/num_jobs`" ] && \
  echo "$alidir and $denlatdir have different num-jobs" && exit 1;

sdata=$data/split$nj
splice_opts=`cat $alidir/splice_opts 2>/dev/null`
cmvn_opts=`cat $alidir/cmvn_opts 2>/dev/null`
mkdir -p $dir/log
cp $alidir/splice_opts $dir 2>/dev/null
cp $alidir/cmvn_opts $dir 2>/dev/null # cmn/cmvn option.
[[ -d $sdata && $data/feats.scp -ot $sdata ]] || split_data.sh $data $nj || exit 1;
echo $nj > $dir/num_jobs

cp $alidir/tree $dir
cp $alidir/final.mdl $dir/0.mdl

silphonelist=`cat $lang/phones/silence.csl` || exit 1;
oov=`cat $lang/oov.int`

# Set up features

if [ -f $alidir/final.mat ]; then feat_type=lda; else feat_type=delta; fi
echo "$0: feature type is $feat_type"

case $feat_type in
  delta) feats="ark,s,cs:apply-cmvn $cmvn_opts --utt2spk=ark:$sdata/JOB/utt2spk scp:$sdata/JOB/cmvn.scp scp:$sdata/JOB/feats.scp ark:- | add-deltas ark:- ark:- |";;
  lda) feats="ark,s,cs:apply-cmvn $cmvn_opts --utt2spk=ark:$sdata/JOB/utt2spk scp:$sdata/JOB/cmvn.scp scp:$sdata/JOB/feats.scp ark:- | splice-feats $splice_opts ark:- ark:- | transform-feats $alidir/final.mat ark:- ark:- |"
    cp $alidir/final.mat $dir    
    ;;
  *) echo "Invalid feature type $feat_type" && exit 1;
esac

[ -f $alidir/trans.1 ] && echo Using transforms from $alidir && \
  feats="$feats transform-feats --utt2spk=ark:$sdata/JOB/utt2spk ark,s,cs:$alidir/trans.JOB ark:- ark:- |"

lats="ark:gunzip -c $denlatdir/lat.JOB.gz|"
if [[ "$boost" != "0.0" && "$boost" != 0 ]]; then
  lats="$lats lattice-boost-ali --b=$boost --silence-phones=$silphonelist $alidir/final.mdl ark:- 'ark,s,cs:gunzip -c $alidir/ali.JOB.gz|' ark:- |"
fi
cp $alidir/ali.*.gz $dir/

x=0
while [ $x -lt $num_iters ]; do
  echo "Iteration $x of MMI training"
  # Note: the num and den states are accumulated at the same time, so we
  # can cancel them per frame.
  if [ $stage -le $x ]; then
    $cmd JOB=1:$nj $dir/log/acc.$x.JOB.log \
      gmm-rescore-lattice $dir/$x.mdl "$lats" "$feats" ark:- \| \
      lattice-to-post --acoustic-scale=$acwt ark:- ark:- \| \
      sum-post --drop-frames=$drop_frames --merge=$cancel --scale1=-1 \
      ark:- "ark,s,cs:gunzip -c $dir/ali.JOB.gz | ali-to-post ark:- ark:- |" ark:- \| \
      gmm-acc-stats2 $dir/$x.mdl "$feats" ark,s,cs:- \
      $dir/num_acc.$x.JOB.acc $dir/den_acc.$x.JOB.acc || exit 1;

    n=`echo $dir/{num,den}_acc.$x.*.acc | wc -w`;
    [ "$n" -ne $[$nj*2] ] && \
      echo "Wrong number of MMI accumulators $n versus 2*$nj" && exit 1;
    $cmd $dir/log/den_acc_sum.$x.log \
      gmm-sum-accs $dir/den_acc.$x.acc $dir/den_acc.$x.*.acc || exit 1;
    rm $dir/den_acc.$x.*.acc
    $cmd $dir/log/num_acc_sum.$x.log \
      gmm-sum-accs $dir/num_acc.$x.acc $dir/num_acc.$x.*.acc || exit 1;
    rm $dir/num_acc.$x.*.acc

  # note: this tau value is for smoothing towards model parameters, not
  # as in the Boosted MMI paper, not towards the ML stats as in the earlier
  # work on discriminative training (e.g. my thesis).  
  # You could use gmm-ismooth-stats to smooth to the ML stats, if you had
  # them available [here they're not available if cancel=true].

    $cmd $dir/log/update.$x.log \
      gmm-est-gaussians-ebw --tau=$tau $dir/$x.mdl $dir/num_acc.$x.acc $dir/den_acc.$x.acc - \| \
      gmm-est-weights-ebw --weight-tau=$weight_tau - $dir/num_acc.$x.acc $dir/den_acc.$x.acc $dir/$[$x+1].mdl || exit 1;
    rm $dir/{den,num}_acc.$x.acc
  fi
  
  #### Re-generate lattice and alignment
  if echo $regenerate_iters | grep -w $x >/dev/null; then
  	echo "regenerate lattice from $dir/$[$x+1].mdl"
  	$cmd JOB=1:$nj $dir/log/decode_den.JOB.log \
  	 gmm-latgen-faster$thread_string --beam=$beam --lattice-beam=$lattice_beam --acoustic-scale=$acwt \
  	  --max-mem=$max_mem --max-active=$max_active --word-symbol-table=$lang/words.txt $dir/$[$x+1].mdl  \
  	   $denlatdir/dengraph/HCLG.fst "$feats" "ark:|gzip -c >$dir/lat.JOB.gz" || exit 1;
  	lats="ark:gunzip -c $dir/lat.JOB.gz|"

		echo "regenerate alignment from $dir/$[$x+1].mdl"
		mdl="gmm-boost-silence --boost=$boost_silence `cat $lang/phones/optional_silence.csl` $dir/$[$x+1].mdl - |"
		tra="ark:utils/sym2int.pl --map-oov $oov -f 2- $lang/words.txt $sdata/JOB/text|";
  	$cmd JOB=1:$nj $dir/log/align.JOB.log \
  	  compile-train-graphs $tri2bdir/tree $dir/$[$x+1].mdl  $lang/L.fst "$tra" ark:- \| \
  	  gmm-align-compiled $scale_opts --beam=$ali_beam --retry-beam=$ali_retry_beam "$mdl" ark:- \
  	    "$feats" "ark,t:|gzip -c >$dir/ali.JOB.gz" || exit 1;	
		ali="ark,s,cs:gunzip -c $dir/ali.JOB.gz | ali-to-post ark:- ark:- |"
  fi
  
  
  # Some diagnostics: the objective function progress and auxiliary-function
  # improvement.

  tail -n 50 $dir/log/acc.$x.*.log | perl -e '$acwt=shift @ARGV; while(<STDIN>) { if(m/gmm-acc-stats2.+Overall weighted acoustic likelihood per frame was (\S+) over (\S+) frames/) { $tot_aclike += $1*$2; $tot_frames1 += $2; } if(m|lattice-to-post.+Overall average log-like/frame is (\S+) over (\S+) frames.  Average acoustic like/frame is (\S+)|) { $tot_den_lat_like += $1*$2; $tot_frames2 += $2; $tot_den_aclike += $3*$2; } } if (abs($tot_frames1 - $tot_frames2) > 0.01*($tot_frames1 + $tot_frames2)) { print STDERR "Frame-counts disagree $tot_frames1 versus $tot_frames2\n"; } $tot_den_lat_like /= $tot_frames2; $tot_den_aclike /= $tot_frames2; $tot_aclike *= ($acwt / $tot_frames1);  $num_like = $tot_aclike + $tot_den_aclike; $per_frame_objf = $num_like - $tot_den_lat_like; print "$per_frame_objf $tot_frames1\n"; ' $acwt > $dir/tmpf
  objf=`cat $dir/tmpf | awk '{print $1}'`;
  nf=`cat $dir/tmpf | awk '{print $2}'`;
  rm $dir/tmpf
  impr=`grep -w Overall $dir/log/update.$x.log | awk '{x += $10*$12;} END{print x;}'`
  impr=`perl -e "print ($impr*$acwt/$nf);"` # We multiply by acwt, and divide by $nf which is the "real" number of frames.
  echo "Iteration $x: objf was $objf, MMI auxf change was $impr" | tee $dir/objf.$x.log
  x=$[$x+1]
done

echo "MMI training finished"

rm $dir/final.mdl 2>/dev/null
ln -s $x.mdl $dir/final.mdl

exit 0;
