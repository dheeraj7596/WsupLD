python3 gpt2_main.py 1 2 nyt-fine 0 0 0.5 > output/nyt_fine/gpt2_ttest_nofilter.txt
python3 gpt2_main.py 1 2 nyt-fine 1 0 0.5 > output/nyt_fine/gpt2_ttest_top50.txt
python3 gpt2_main.py 1 2 nyt-fine 2 0 0.5 > output/nyt_fine/gpt2_ttest_probability.txt
python3 gpt2_main.py 1 2 nyt-fine 4 0 0.5 > output/nyt_fine/gpt2_ttest_random.txt
python3 gpt2_main.py 1 2 nyt-fine 5 0 0.5 > output/nyt_fine/gpt2_ttest_stability.txt