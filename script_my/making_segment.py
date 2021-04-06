import subprocess
import os

# transform by level video BoW encoding
transform = ['original', 'resolution', 'framerate', 'brightness', 'crop', 'format', 'logo', 'rotate', 'flip', 'grayscale', 'border']
level = ['Heavy', 'Medium', 'Light']

for t in transform:
    video_dataset = os.path.join('/mldisk/nfs_shared_/js/features/simulated_dataset-1fps-resnet50-rmac', t)
    seg_path = os.path.join('/mldisk/nfs_shared_/my_vlad/simulated_features/5sec_segment_feature/dataset/resnet_rmac_simulated_dataset', t)
    if os.path.exists(video_dataset):
        if not os.path.isdir(seg_path):
            os.makedirs(seg_path)
        command = f'/opt/conda/bin/python -u /workspace/script_my/pooling.py --model Segment_MaxPool --frame_feature_path {video_dataset} --segment_feature_path {seg_path} --count 5'

        subprocess.call(command, shell=True)



# # simulated original video BoW encoding
# video_dataset = "/mldisk/nfs_shared_/js/SIMULATED_DATASET/videos/original_SD"
# feature_path = "/mldisk/nfs_shared_/js/LOCAL/localFeatures/simulated_dataset-1fps-mobilenet-60sec/original"
# BOW_path = "/mldisk/nfs_shared_/js/LOCAL/BOW_segment/simulated_dataset-1fps-mobilenet-60sec-k15000/original"
#
# if os.path.exists(BOW_path):
#     os.makedirs(BOW_path)
# if os.path.isdir(video_dataset):
#     command = '/opt/conda/bin/python -u C_BOW_encoding.py --model_path /mldisk/nfs_shared_/js/LOCAL/clusters/vcdb_core-mobilenet-15000.pkl --feature_path ' + feature_path + ' --BOW_path ' + BOW_path
#     subprocess.call(command, shell=True)