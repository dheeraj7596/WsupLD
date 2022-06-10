gpu=$1

python3 train.py 1 ${gpu} nyt-coarse 8 0 0.3 > output/nyt_coarse/bert_prob_num_filter_0.3thresh.txt
python3 train.py 1 ${gpu} nyt-coarse 8 0 0.5 > output/nyt_coarse/bert_prob_num_filter_0.5thresh.txt
python3 train.py 1 ${gpu} nyt-coarse 8 0 0.7 > output/nyt_coarse/bert_prob_num_filter_0.7thresh.txt
python3 train.py 1 ${gpu} nyt-coarse 8 0 0.8 > output/nyt_coarse/bert_prob_num_filter_0.8thresh.txt
python3 train.py 1 ${gpu} nyt-coarse 8 0 0.9 > output/nyt_coarse/bert_prob_num_filter_0.9thresh.txt

python3 train.py 1 ${gpu} 20news-fine-nomisc 8 0 0.3 > output/20news_fine_nomisc/bert_prob_num_filter_0.3thresh.txt
python3 train.py 1 ${gpu} 20news-fine-nomisc 8 0 0.5 > output/20news_fine_nomisc/bert_prob_num_filter_0.5thresh.txt
python3 train.py 1 ${gpu} 20news-fine-nomisc 8 0 0.7 > output/20news_fine_nomisc/bert_prob_num_filter_0.7thresh.txt
python3 train.py 1 ${gpu} 20news-fine-nomisc 8 0 0.8 > output/20news_fine_nomisc/bert_prob_num_filter_0.8thresh.txt
python3 train.py 1 ${gpu} 20news-fine-nomisc 8 0 0.9 > output/20news_fine_nomisc/bert_prob_num_filter_0.9thresh.txt