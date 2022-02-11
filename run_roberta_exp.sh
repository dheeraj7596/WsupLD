python3 roberta_main.py 1 4 books 0 0 0.5 > output/books/roberta_ttest_nofilter.txt
python3 roberta_main.py 1 4 books 1 0 0.5 > output/books/roberta_ttest_top50.txt
python3 roberta_main.py 1 4 books 2 0 0.5 > output/books/roberta_ttest_probability.txt
python3 roberta_main.py 1 4 books 4 0 0.5 > output/books/roberta_ttest_random.txt
python3 roberta_main.py 1 4 books 5 0 0.5 > output/books/roberta_ttest_stability.txt
