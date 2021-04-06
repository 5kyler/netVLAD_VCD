from multiprocessing import Pool
from tqdm import tqdm
import argparse
import os


def copy(cmd):
    os.system(cmd)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='copy directory')
    parser.add_argument('--src_dir', type=str, required=True)
    parser.add_argument('--dst_dir', type=str, required=True)
    parser.add_argument('-c', '--core', type=int, default=4)
    args = parser.parse_args()

    src_dir = args.src_dir  # '/mldisk/nfs_shared_/MLVD2/CC_WEB/videos'
    dst_dir = args.dst_dir  # '/hdd/ms/MLVD/CC_WEB/videos'

    videos = []
    for r, d, f in os.walk(src_dir):
        for v in f:
            videos.append(os.path.join(r, v))

    pool = Pool(args.core)
    bar = tqdm(videos)
    ret = []
    for src in videos:
        dst = os.path.join(dst_dir, os.path.relpath(src, src_dir))
        # dst = os.path.join(dst_dir, os.path.basename(src))
        if not os.path.exists(os.path.dirname(dst)):
            os.makedirs(os.path.dirname(dst))
        cmd = f'cp {src} {dst}'
        ret += [pool.apply_async(copy, args=[cmd], callback=lambda x: bar.update())]
    pool.close()
    pool.join()
    bar.close()
