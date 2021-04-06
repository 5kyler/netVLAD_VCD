import argparse

from VCD.datasets import VCDB
from VCD.utils import *

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Partial copy detection for VCDB.")
    parser.add_argument('--vcdb_root', type=str, required=True)
    parser.add_argument('--feature_path', type=str, required=True)

    # TN - parameters
    parser.add_argument('--topk', type=int, default=50)
    parser.add_argument('--feature_intv', type=int, default=1)
    parser.add_argument('--window', type=int, default=5)
    parser.add_argument('--path_thr', type=int, default=3)
    parser.add_argument('--score_thr', type=float, default=-1)

    args = parser.parse_args()
    print(args)

    param = [args.topk, args.feature_intv, args.window, args.path_thr, args.score_thr]
    vcdb = VCDB(args.vcdb_root)

    fscore, prec, rec, time = vcdb_partial_copy_detection(vcdb, args.feature_path, param)

    print(f'>> fscore: {fscore:.4f} ' \
          f'precision: {prec:.4f} ' \
          f'recall: {rec:.4f} ' \
          f'time: {time}')
