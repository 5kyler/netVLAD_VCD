
from VCD.datasets import VCDB
from VCD.utils import *
import csv

# vcdb_root , feature_path, topk, window, path_thr, score_thr


def performance(vcdb_root, feature_path):
    f = open("/workspace/result/result_resnet_16_512.csv", "w", newline='')
    w = csv.writer(f)
    w.writerow(['topk', 'window', 'path_thr', 'score_thr', 'fscore', 'precision', 'recall', 'time'])
    vcdb = VCDB(vcdb_root)
    topk = [50, 100, 150]    # 50 # 50     # 50   # 50 100 150
    feature_intv = 1
    window = [2, 3]   # 2   # 2 3        # 2 3 4 5             # 5 10 15 20
    path_thr = [2, 3]      # 2        # 2 4 5 8        # 3           # 3
    score_thr = [0.4, 0.5, 0.55]   # 0.5 0.55    # 0.4 0.5 0.6    # 0.3 0.4 0.5       # -1 0.3
    for topk_ele in topk:
        for window_ele in window:
            for path_thr_ele in path_thr:
                for score_thr_ele in score_thr:
                    param = [topk_ele, feature_intv, window_ele, path_thr_ele, score_thr_ele]
                    fscore, prec, rec, time = vcdb_partial_copy_detection(vcdb, feature_path, param)
                    w.writerow([topk_ele, window_ele, path_thr_ele, score_thr_ele, fscore, prec, rec, time])
    f.close()


if __name__ == '__main__':
    vcdb_root = '/mldisk/nfs_shared_/MLVD/VCDB-core'
    feature_path = '/mldisk/nfs_shared_/my_vlad/resnet-rmac/VCDB-core/k16_512'
    # 현재 512차원 pca 된 피쳐로 실험중
    performance(vcdb_root, feature_path)