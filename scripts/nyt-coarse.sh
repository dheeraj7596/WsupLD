gpu=$1

python3 train.py 1 ${gpu} nyt-coarse 0 0 0.5 > output/nyt_coarse/bert_no_filter_0thresh.txt
python3 train.py 1 ${gpu} nyt-coarse 2 0 0.5 > output/nyt_coarse/bert_prob_filter_0thresh.txt
python3 train.py 1 ${gpu} nyt-coarse 3 0 0.5 > output/nyt_coarse/bert_upperbound_0thresh.txt
python3 train.py 1 ${gpu} nyt-coarse 4 0 0.5 > output/nyt_coarse/bert_random_filter_0thresh.txt
python3 train.py 1 ${gpu} nyt-coarse 5 0 0.5 > output/nyt_coarse/bert_stability_filter_0thresh.txt
python3 -i train.py 1 ${gpu} nyt-coarse 7 0 0.5 > output/nyt_coarse/bert_o2unet_0thresh.txt