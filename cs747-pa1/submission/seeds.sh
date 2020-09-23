for i in `seq 0 49`; do
   python bandit_kl.py --randomSeed $i &
done