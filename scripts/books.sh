gpu=$1

#python3 train.py 1 ${gpu} books 0 0 0.5 > output/books/bert_no_filter_0thresh.txt
python3 train.py 1 ${gpu} books 2 0 0.5 > output/books/bert_prob_filter_0thresh.txt
python3 train.py 1 ${gpu} books 3 0 0.5 > output/books/bert_upperbound_0thresh.txt
python3 train.py 1 ${gpu} books 4 0 0.5 > output/books/bert_random_filter_0thresh.txt
python3 train.py 1 ${gpu} books 5 0 0.5 > output/books/bert_stability_filter_0thresh.txt
python3 -i train.py 1 ${gpu} books 7 0 0.5 > output/books/bert_o2unet_0thresh.txt