python3 roberta_main.py 1 4 20news-coarse-nomisc 0 0 0.5 > output/20news_coarse_nomisc/roberta_ttest_nofilter.txt
python3 roberta_main.py 1 4 20news-coarse-nomisc 1 0 0.5 > output/20news_coarse_nomisc/roberta_ttest_top50.txt
python3 roberta_main.py 1 4 20news-coarse-nomisc 2 0 0.5 > output/20news_coarse_nomisc/roberta_ttest_probability.txt
python3 roberta_main.py 1 4 20news-coarse-nomisc 4 0 0.5 > output/20news_coarse_nomisc/roberta_ttest_random.txt
python3 roberta_main.py 1 4 20news-coarse-nomisc 5 0 0.5 > output/20news_coarse_nomisc/roberta_ttest_stability.txt
