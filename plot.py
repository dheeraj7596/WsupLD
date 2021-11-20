import matplotlib.pyplot as plt

if __name__ == "__main__":
    # acc = [1.0613,
    #        0.1507,
    #        0.0428,
    #        0.0224,
    #        0.0189,
    #        0.0172,
    #        0.0141,
    #        0.0117,
    #        0.0084,
    #        0.0070,
    #        0.0068,
    #        0.0082,
    #        0.0036,
    #        0.0024,
    #        0.0042,
    #        0.0046,
    #        0.0020,
    #        0.0029,
    #        0.0013,
    #        0.0011,
    #        0.0025,
    #        9.7321e-04,
    #        0.0025,
    #        0.0013,
    #        1.9664e-04,
    #        9.4393e-05,
    #        8.7978e-05,
    #        4.9221e-05,
    #        4.8147e-05,
    #        3.9787e-05,
    #        3.1360e-05,
    #        2.7318e-05,
    #        2.3857e-05,
    #        1.9099e-05,
    #        1.8467e-05,
    #        1.6542e-05,
    #        1.5344e-05,
    #        1.3241e-05,
    #        1.0739e-05,
    #        1.2070e-05,
    #        9.5601e-06,
    #        8.9756e-06,
    #        8.1761e-06,
    #        7.1822e-06,
    #        6.8169e-06,
    #        6.5587e-06,
    #        5.1380e-06,
    #        4.9136e-06,
    #        4.4634e-06,
    #        4.2794e-06
    #        ]
    # epoch = list(range(50))
    #
    # plt.plot(epoch, acc)
    # plt.xlabel("Epoch", fontsize=22)
    # # plt.ylabel("Training accuracy", fontsize=22)
    # plt.ylabel("Training loss", fontsize=22)
    # plt.xticks(fontsize=22)
    # plt.yticks(fontsize=22)
    #
    # # plt.savefig('./nyt_coarse.png')
    # plt.show()

    # nyt-coarse
    # risk_epoch = [8284, 8282, 8271, 8289, 8341, 8335, 8414]
    # risk_prob = [8125, 8186, 8218, 8245, 8336, 8334, 8414]
    # coverage = [0.93, 0.938, 0.946, 0.945, 0.964, 0.965, 1]

    # 20news-fine
    # risk_epoch = [7442, 7508, 7627, 7625, 7644, 7771]
    # risk_prob = [7326, 7427, 7608, 7613, 7685, 7771]
    # coverage = [0.912, 0.927, 0.959, 0.9613, 0.9684, 1]
    #
    # plt.plot(coverage, risk_epoch, label="Epoch-based filter")
    # plt.plot(coverage, risk_prob, label="Probability-based filter")
    # plt.xlabel("Coverage")
    # plt.ylabel("Risk=#Correctly labeled samples")
    # plt.legend()
    # plt.title("20News-Fine")
    # plt.savefig("./data/plots/20news_fine_rc.png")

    perf = [0.72, 0.74, 0.59, 0.57, 0.54, 0.53]
    its = range(1, len(perf) + 1)

    plt.plot(its, perf)
    plt.ylabel("Macro F1")
    plt.xlabel("Iteration")
    plt.title("Performance vs Iteration")
    plt.yticks([0, 0.10, 0.20, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])
    plt.savefig("./data/plots/conwea_it.png")
    # plt.show()
