python3 train.py 1 0 nyt-fine 0 0 0.5 > output/nyt_fine/bert_ttest_nofilter.txt
python3 train.py 1 0 nyt-fine 1 0 0.5 > output/nyt_fine/bert_ttest_top50.txt
python3 train.py 1 0 nyt-fine 2 0 0.5 > output/nyt_fine/bert_ttest_probability.txt
python3 train.py 1 0 nyt-fine 4 0 0.5 > output/nyt_fine/bert_ttest_random.txt
python3 train.py 1 0 nyt-fine 5 0 0.5 > output/nyt_fine/bert_ttest_stability.txt