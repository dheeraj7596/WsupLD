python3 roberta_main.py 1 5 nyt-coarse 0 0 0.5 > output/nyt_coarse/roberta_ttest_nofilter.txt
python3 roberta_main.py 1 5 nyt-coarse 1 0 0.5 > output/nyt_coarse/roberta_ttest_top50.txt
python3 roberta_main.py 1 5 nyt-coarse 2 0 0.5 > output/nyt_coarse/roberta_ttest_probability.txt
python3 roberta_main.py 1 5 nyt-coarse 4 0 0.5 > output/nyt_coarse/roberta_ttest_random.txt
python3 roberta_main.py 1 5 nyt-coarse 5 0 0.5 > output/nyt_coarse/roberta_ttest_stability.txt
