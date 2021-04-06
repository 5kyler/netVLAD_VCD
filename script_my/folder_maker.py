
import os

import shutil


hkseokPath = "/mldisk/nfs_shared_/hkseok/LocalFeatures/multiple/vcdb_core-1fps-mobilenet-5sec"  ##
savePath = "/mldisk/nfs_shared_/my/gift/vcdb_core-1fps-mobilenet-5sec"   ##

copyFolderPath ="/mldisk/nfs_shared_/js/features/VCDB_core/mobilenet_avg/core_dataset"

hkseokFeatures = os.listdir(hkseokPath)

folder = os.listdir(copyFolderPath)


for f in folder:
    if not os.path.exists(f'{savePath}/{f}'):
        os.makedirs(f'{savePath}/{f}')

    folder_path = os.path.join(copyFolderPath, f)
    folder_name = f

    for video in os.listdir(folder_path):
        shutil.copyfile(f'{hkseokPath}/{video}', f'{savePath}/{folder_name}/{video}')