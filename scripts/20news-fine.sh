gpu=$1

python3 train.py 1 ${gpu} 20news-fine-nomisc 6 0 0.5 1 > output/20news_fine_nomisc/bert_batch_epoch_filter_1.txt
python3 train.py 1 ${gpu} 20news-fine-nomisc 6 0 0.5 3 > output/20news_fine_nomisc/bert_batch_epoch_filter_3.txt
python3 train.py 1 ${gpu} 20news-fine-nomisc 6 0 0.5 5 > output/20news_fine_nomisc/bert_batch_epoch_filter_5.txt
python3 train.py 1 ${gpu} 20news-fine-nomisc 6 0 0.5 15 > output/20news_fine_nomisc/bert_batch_epoch_filter_15.txt
python3 train.py 1 ${gpu} 20news-fine-nomisc 6 0 0.5 30 > output/20news_fine_nomisc/bert_batch_epoch_filter_30.txt
python3 -i train.py 1 ${gpu} 20news-fine-nomisc 6 0 0.5 40 > output/20news_fine_nomisc/bert_batch_epoch_filter_40.txt