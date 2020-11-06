for i in `seq 0 49`; do
   python3 bandit_kl.py --randomSeed $i &
done
