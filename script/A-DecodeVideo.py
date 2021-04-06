from multiprocessing import Pool
from datetime import datetime
import argparse
import logging
import json
import tqdm
import os

from VCD.utils import *

logger = None


def parse_and_decode(path, dst, idx, decode_rate):
    if not os.path.exists(dst):
        os.makedirs(dst)

    code, meta = parse_video(path)
    frame_count = 0

    if code:
        code = decode_video(path, dst, decode_rate)
        frame_count = len(os.listdir(dst))
        code = code and frame_count > 0

    return code, idx, path, meta, frame_count


def update(bar, result):
    code, idx, path, _, _ = result
    if not code:
        logger.info(f'Fail {idx}: {path}')
    bar.update()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Extract frames')
    parser.add_argument('--root', type=str, help='root directory', required=True)
    parser.add_argument('--video_dir', type=str, default='videos')
    parser.add_argument('--frame_dir', type=str, default='frames')
    parser.add_argument('--meta_dir', type=str, default='meta')
    parser.add_argument('--fps', type=int, default=1)

    args = parser.parse_args()

    video_dir = os.path.abspath(os.path.join(args.root, args.video_dir))
    frame_dir = os.path.abspath(os.path.join(args.root, args.frame_dir))
    meta_path = os.path.abspath(os.path.join(args.root, args.meta_dir, 'metadata.json'))
    count_path = os.path.abspath(os.path.join(args.root, args.meta_dir, f'{args.frame_dir}.json'))
    fail_path = os.path.abspath(os.path.join(args.root, args.meta_dir, f'{args.frame_dir}-fail.json'))

    log = os.path.join(args.root, f'extract_{args.frame_dir}.log')
    if os.path.exists(log):
        log = os.path.join(args.root, f'extract_{args.frame_dir}_{datetime.now(tz=KST).strftime("%Y%m%d%H%M%S")}.log')

    logger = initialize_log(log)

    if os.path.exists(frame_dir) and len(os.listdir(frame_dir)) != 0:
        logger.info(f'Frame directory {frame_dir} is already exist')
        exit(1)
    if os.path.exists(meta_path):
        logger.info(f'Metadata {meta_path} is already exist')
        exit(1)
    if os.path.exists(count_path):
        logger.info(f'frame count {count_path} is already exist')
        exit(1)
    if not os.path.exists(os.path.dirname(meta_path)):
        os.makedirs(os.path.dirname(meta_path))

    videos = sorted(
        [os.path.relpath(os.path.join(r, v), video_dir) for r, d, f in os.walk(video_dir, followlinks=True) for v in f
         if v.split('.')[1].lower() in VIDEO_EXTENSION])

    logger.info(f'Decode {len(videos)} videos, {args.fps} fps')
    logger.info(f'Videos ... {video_dir}')
    logger.info(f'Frames ... {frame_dir}')
    logger.info(f'metadata ... {meta_path}')
    logger.info(f'frame count ... {count_path}')
    logger.info(f'log ... {os.path.abspath(log)}')

    bar = tqdm.tqdm(videos, mininterval=1, ncols=150)
    pool = Pool()
    results = [pool.apply_async(parse_and_decode,
                                args=[os.path.join(video_dir, v),
                                      os.path.join(frame_dir, v),
                                      c,
                                      args.fps],
                                callback=lambda ret: update(bar, ret))
               for c, v in enumerate(videos, start=1)]

    pool.close()
    pool.join()
    bar.close()

    metadata = dict()
    count = dict()
    fail = dict()
    for r in results:
        code, idx, path, meta, c = r.get()
        video = os.path.relpath(path, video_dir)
        meta['video'] = video
        if code:
            metadata[video] = meta
            count[video] = c
        else:
            meta['count'] = c
            fail[video] = meta

    json.dump(metadata, open(meta_path, 'w'))
    json.dump(count, open(count_path, 'w'))
    json.dump(fail, open(fail_path, 'w'))

    logger.info("DONE")

    logging.shutdown()
