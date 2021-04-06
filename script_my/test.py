import torch
import numpy as np
from sklearn.decomposition import PCA
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import os
# test: 영상 하나 의미함
# idx : 1개의 segment 의미
#
import shutil
testt = torch.load('/mldisk/nfs_shared_/my/NETVLAD/fivr_resnet50_rmac_feature_upgrade/core/--W0UxdTL-c.mp4.pth')
test = torch.load('/mldisk/nfs_shared_/my/NETVLAD/fivr_resnet50_rmac_feature_upgrade/core/-0cCW_eDkkU.mp4.pth')







positive_csv = os.path.join('/mldisk/nfs_shared_/my/NETVLAD/triplet_sampling/test_fivr_triplet.csv')

df = pd.read_csv(positive_csv)
drop_df = df.drop_duplicates(["anchor","positive","negative"], keep="last")
save_df = drop_df.to_csv('/workspace/script_my/netVLAD/csv_folder/test_chk.csv', columns=["anchor","positive","negative"], index = False)
import pdb; pdb.set_trace()











folder_path ='/mldisk/nfs_shared_/my_vlad/simulated_features/5sec_segment_feature/400/db_vlad'
descriptors = []
for file in os.listdir(folder_path):
    descriptor = torch.load(os.path.join(folder_path, file))
    descriptors.append(descriptor)
descriptor_set = np.vstack(descriptors)
print(descriptor_set.shape)

# def cosine_similarity(a, b):a
#
#     #return np.inner(a, b) / (np.linalg.norm(a) * (np.linalg.norm(b)))
#     return np.dot(a, b) / (np.linalg.norm(a) * (np.linalg.norm(b)))
#
# border = torch.load('/mldisk/nfs_shared_/my_vlad/simulated_features/5sec_segment_feature/400/sim_vlad/border/Light/0YY0RDbxSns.flv.pth')
# original = torch.load('/mldisk/nfs_shared_/my_vlad/simulated_features/5sec_segment_feature/400/sim_vlad/original/0YY0RDbxSns.flv.pth')
# #framerate = torch.load('/mldisk/nfs_shared_/my_vlad/simulated_features/5sec_segment_feature/resnet_rmac_sim_vlad/framerate/Light/0YY0RDbxSns.flv.pth')
# #vb = torch.load('/mldisk/nfs_shared_/my_vlad/simulated_features/5sec_segment_feature/resnet_rmac_db_vlad/baggio_penalty_1994/9423340949f55d7a089e881ccc35f23dff3fccf7.flv.pth')
# mmm = cosine_similarity(border[2], original[2])
#
#
# for i in range(30):
#     fff = cosine_similarity(border[i], original[i])
#     print(fff)


takepath = os.path.split('/mldisk/nfs_shared_/my/benchmark_data/datasets/oxford/jpg/all_souls_000000.jpg')


import pdb; pdb.set_trace()

frame_list = []
for frame in test:
    feature_list = []
    for feature in frame:
        feature_list.append(feature[0])
    frame_tensor = torch.stack(feature_list)
    import pdb;

    pdb.set_trace()
    pca = PCA(n_components=7, whiten=True)
    pca_feature = pca.fit_transform(frame_tensor)
    pca_feature = torch.tensor(pca_feature, dtype=torch.float32)
    pca_feature = torch.flatten(pca_feature)

    frame_list.append(pca_feature)
save_feature = torch.stack(frame_list)

tests = torch.load('/mldisk/nfs_shared_/my_vlad/test1/vlad/resnet_local/pca1225/baggio_penalty_1994/3504e360accbaccb1580befbb441f1019664c2bb.mp4.pth')




test1 = torch.load('/mldisk/nfs_shared_/hkseok/BOW/single/vcdb_core-1fps-res50-5sec/00274a923e13506819bd273c694d10cfa07ce1ec.flv.pth')
test3 = torch.load('/mldisk/nfs_shared_/my_vlad/resnet_segment_features/baggio_penalty_1994/3504e360accbaccb1580befbb441f1019664c2bb.mp4.pth')


