python3 train.py 1 0 books 0 0 0.5 > output/books/bert_ttest_nofilter.txt
python3 train.py 1 0 books 1 0 0.5 > output/books/bert_ttest_top50.txt
python3 train.py 1 0 books 2 0 0.5 > output/books/bert_ttest_probability.txt
python3 train.py 1 0 books 4 0 0.5 > output/books/bert_ttest_random.txt
python3 train.py 1 0 books 5 0 0.5 > output/books/bert_ttest_stability.txt