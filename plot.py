import matplotlib.pyplot as plt

if __name__ == "__main__":
    acc = [1.0613,
           0.1507,
           0.0428,
           0.0224,
           0.0189,
           0.0172,
           0.0141,
           0.0117,
           0.0084,
           0.0070,
           0.0068,
           0.0082,
           0.0036,
           0.0024,
           0.0042,
           0.0046,
           0.0020,
           0.0029,
           0.0013,
           0.0011,
           0.0025,
           9.7321e-04,
           0.0025,
           0.0013,
           1.9664e-04,
           9.4393e-05,
           8.7978e-05,
           4.9221e-05,
           4.8147e-05,
           3.9787e-05,
           3.1360e-05,
           2.7318e-05,
           2.3857e-05,
           1.9099e-05,
           1.8467e-05,
           1.6542e-05,
           1.5344e-05,
           1.3241e-05,
           1.0739e-05,
           1.2070e-05,
           9.5601e-06,
           8.9756e-06,
           8.1761e-06,
           7.1822e-06,
           6.8169e-06,
           6.5587e-06,
           5.1380e-06,
           4.9136e-06,
           4.4634e-06,
           4.2794e-06
           ]
    epoch = list(range(50))

    plt.plot(epoch, acc)
    plt.xlabel("Epoch", fontsize=22)
    # plt.ylabel("Training accuracy", fontsize=22)
    plt.ylabel("Training loss", fontsize=22)
    plt.xticks(fontsize=22)
    plt.yticks(fontsize=22)

    # plt.savefig('./nyt_coarse.png')
    plt.show()
