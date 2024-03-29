gpu=$1

python3 train.py 1 ${gpu} 20news-coarse-nomisc 0 0 0.5 > output/20news_coarse_nomisc/bert_no_filter_0thresh.txt
python3 train.py 1 ${gpu} 20news-coarse-nomisc 2 0 0.5 > output/20news_coarse_nomisc/bert_prob_filter_0thresh.txt
python3 train.py 1 ${gpu} 20news-coarse-nomisc 3 0 0.5 > output/20news_coarse_nomisc/bert_upperbound_0thresh.txt
python3 train.py 1 ${gpu} 20news-coarse-nomisc 4 0 0.5 > output/20news_coarse_nomisc/bert_random_filter_0thresh.txt
python3 train.py 1 ${gpu} 20news-coarse-nomisc 5 0 0.5 > output/20news_coarse_nomisc/bert_stability_filter_0thresh.txt
python3 -i train.py 1 ${gpu} 20news-coarse-nomisc 7 0 0.5 > output/20news_coarse_nomisc/bert_o2unet_0thresh.txt