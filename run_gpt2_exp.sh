python3 gpt2_main.py 1 2 20news-fine-nomisc 0 0 0.5 > output/20news_fine_nomisc/gpt2_ttest_nofilter.txt
python3 gpt2_main.py 1 2 20news-fine-nomisc 1 0 0.5 > output/20news_fine_nomisc/gpt2_ttest_top50.txt
python3 gpt2_main.py 1 2 20news-fine-nomisc 2 0 0.5 > output/20news_fine_nomisc/gpt2_ttest_probability.txt
python3 gpt2_main.py 1 2 20news-fine-nomisc 4 0 0.5 > output/20news_fine_nomisc/gpt2_ttest_random.txt
python3 gpt2_main.py 1 2 20news-fine-nomisc 5 0 0.5 > output/20news_fine_nomisc/gpt2_ttest_stability.txt