gpu=$1

python3 train.py 1 ${gpu} nyt-coarse 6 0 0.5 1 > output/nyt_coarse/bert_batch_epoch_filter_1.txt
python3 train.py 1 ${gpu} nyt-coarse 6 0 0.5 3 > output/nyt_coarse/bert_batch_epoch_filter_3.txt
python3 train.py 1 ${gpu} nyt-coarse 6 0 0.5 5 > output/nyt_coarse/bert_batch_epoch_filter_5.txt
python3 train.py 1 ${gpu} nyt-coarse 6 0 0.5 15 > output/nyt_coarse/bert_batch_epoch_filter_15.txt
python3 train.py 1 ${gpu} nyt-coarse 6 0 0.5 30 > output/nyt_coarse/bert_batch_epoch_filter_30.txt
python3 -i train.py 1 ${gpu} nyt-coarse 6 0 0.5 40 > output/nyt_coarse/bert_batch_epoch_filter_40.txt