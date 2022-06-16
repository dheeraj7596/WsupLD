gpu=$1

#python3 train.py 1 ${gpu} 20news-fine-nomisc 1 0 0.3 40 > output/20news_fine_nomisc/bert_top50_epoch_num_filter_0.3thresh.txt
#python3 train.py 1 ${gpu} 20news-fine-nomisc 1 0 0.5 40 > output/20news_fine_nomisc/bert_top50_epoch_num_filter_0.5thresh.txt
#python3 train.py 1 ${gpu} 20news-fine-nomisc 1 0 0.7 40 > output/20news_fine_nomisc/bert_top50_epoch_num_filter_0.7thresh.txt
#python3 train.py 1 ${gpu} 20news-fine-nomisc 1 0 0.8 40 > output/20news_fine_nomisc/bert_top50_epoch_num_filter_0.8thresh.txt
#python3 train.py 1 ${gpu} 20news-fine-nomisc 1 0 0.9 40 > output/20news_fine_nomisc/bert_top50_epoch_num_filter_0.9thresh.txt

python3 train.py 1 ${gpu} 20news-fine-nomisc 6 0 0.5 1 > output/20news_fine_nomisc/bert_batch_epoch_filter_1.txt
python3 train.py 1 ${gpu} 20news-fine-nomisc 6 0 0.5 3 > output/20news_fine_nomisc/bert_batch_epoch_filter_3.txt
python3 train.py 1 ${gpu} 20news-fine-nomisc 6 0 0.5 5 > output/20news_fine_nomisc/bert_batch_epoch_filter_5.txt
python3 train.py 1 ${gpu} 20news-fine-nomisc 6 0 0.5 15 > output/20news_fine_nomisc/bert_batch_epoch_filter_15.txt
python3 train.py 1 ${gpu} 20news-fine-nomisc 6 0 0.5 30 > output/20news_fine_nomisc/bert_batch_epoch_filter_30.txt
python3 -i train.py 1 ${gpu} 20news-fine-nomisc 6 0 0.5 40 > output/20news_fine_nomisc/bert_batch_epoch_filter_40.txt