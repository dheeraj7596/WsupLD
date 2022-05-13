gpu=$1

python3 xlnet_main.py 1 ${gpu} 20news-coarse-nomisc 7 0 0.5 > output/20news_coarse_nomisc/o2u_xlnet.txt
python3 xlnet_main.py 1 ${gpu} 20news-fine-nomisc 7 0 0.5 > output/20news_fine_nomisc/o2u_xlnet.txt
python3 xlnet_main.py 1 ${gpu} nyt-coarse 7 0 0.5 > output/nyt_coarse/o2u_xlnet.txt
python3 xlnet_main.py 1 ${gpu} nyt-fine 7 0 0.5 > output/nyt_fine/o2u_xlnet.txt
python3 xlnet_main.py 1 ${gpu} books 7 0 0.5 > output/books/o2u_xlnet.txt
python3 -i xlnet_main.py 1 ${gpu} agnews 7 0 0.5 > output/agnews/o2u_xlnet.txt