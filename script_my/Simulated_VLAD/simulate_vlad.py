from VCD.utils.load import *
from VCD.models.pooling import *
import numpy as np
import shutil
from sklearn.cluster import MiniBatchKMeans

###############################################################################################################################################

# 1. descriptor 를 vlad vector로 변환
## 1-1. descriptor 를 set(list)으로 만들기
### path : 디스크립터들이 저장된 폴더 경로  /workspace/features/VCDB/core_dataset
def getDescriptorSet(path):
    descriptors = []
    for folder in os.listdir(path):
        if folder == 'original':
            for descriptorPath in os.listdir(os.path.join(path, folder)):
                descriptor, _ = load_file(os.path.join(path, folder, descriptorPath))
                descriptors.append(descriptor)
        else:
            for level in os.listdir(os.path.join(path, folder)):
                for descriptorPath in os.listdir(os.path.join(path, folder, level)):
                    descriptor, _ = load_file(os.path.join(path, folder, level, descriptorPath))
                    descriptors.append(descriptor)
    descriptor_set = np.vstack(descriptors)
    print("descriptor_set 생성 완료")
    print(descriptor_set.shape)
    return descriptor_set


## 1-2. get dictionary (KMeans)
### descriptor_set : 추출된 디스크립터들 하나의 set로 묶음
### k : 클러스터링 하려는 갯수
def getDictionary(descriptor_set, k):
    print('kmeans clustering시작')
    dictionary = MiniBatchKMeans(n_clusters=k, random_state=0).fit(descriptor_set.astype('double'))
    print('kmeans clustering 완료')
    return dictionary


## 1-3. vlad 를 수행하는 코드
### x (M x D matrix) : 한 프레임에 대한 디스크립터
### dictionary : kmeans clustering으로 계산된 딕셔너리
def getVLAD(x, dictionary):
    predicted = dictionary.predict([x])  # x의 각 element 중에서 가장 가까운 군집을 예측함
    clusters = dictionary.cluster_centers_  # clustering 후 생성된 clusters
    k = dictionary.n_clusters  # 설정한 클러 스터수
    x = np.asmatrix(x)
    eps = 1e-6
    m, d = x.shape
    vlad_vector = np.zeros([k, d], dtype='float32')

    for i in range(k):
        if np.sum(predicted == i) > 0:
            vlad_vector[i] = np.sum(x[predicted == i, :] - clusters[i], axis=0)

    vlad_vector = vlad_vector.flatten()     # (1, k*d) 형태로 바꾸기
    vlad_vector = np.sign(vlad_vector) * np.sqrt(np.abs(vlad_vector))   # square_root normalize
    vlad_vector = vlad_vector / (np.sqrt(np.dot(vlad_vector, vlad_vector)) + eps)   # L2 normalize

    return vlad_vector


## 1-4. vlad를 pth 파일로 저장하는 코드
### path : 피쳐가 저장되어있는 경로
### dictionary : kmeansclustering으로 계산된 딕셔너리
def saveVLADVectorPth(q_path, dictionary):
    Q_PATH = '/mldisk/nfs_shared_/my_vlad/simulated_features/5sec_segment_feature/400/sim_vlad'

    # q : simulated dataset
    for folder in os.listdir(q_path):
        if not os.path.isdir(os.path.join(Q_PATH, folder)):
            os.mkdir(os.path.join(Q_PATH, folder))
        if folder == 'original':
            for featurePath in os.listdir(os.path.join(q_path, folder)):
                feature, m = load_file(os.path.join(q_path, folder, featurePath))
                vlad_list = []
                for row in range(0, len(feature)):
                    vlad = getVLAD(feature[row], dictionary)
                    vlad_list.append([vlad])
                vlad_features = np.vstack(vlad_list)
                vlad_features = torch.tensor(vlad_features, dtype=torch.float32)

                torch.save(vlad_features, Q_PATH + '/' + folder + '/' +featurePath)
        else:
            for level in os.listdir(os.path.join(q_path, folder)):
                if not os.path.isdir(os.path.join(Q_PATH, folder, level)):
                    os.mkdir(os.path.join(Q_PATH, folder, level))
                for featurePath in os.listdir(os.path.join(q_path, folder, level)):
                    feature, m = load_file(os.path.join(q_path, folder, level, featurePath))
                    vlad_list = []
                    for row in range(0, len(feature)):
                        vlad = getVLAD(feature[row], dictionary)
                        vlad_list.append([vlad])
                    vlad_features = np.vstack(vlad_list)
                    vlad_features = torch.tensor(vlad_features, dtype=torch.float32)

                    torch.save(vlad_features, Q_PATH + '/' + folder + '/' + level + '/' + featurePath)



def make_directory(path, move_path):
    os.makedirs(move_path)
    for r, d, f in os.walk(path):
        for file in f:
            full = os.path.join(r, file)
            shutil.copy(full, move_path)
    print('폴더 생성 및 랭킹계산 피쳐 저장')


if __name__ == '__main__':
    simulated_path = '/mldisk/nfs_shared_/my_vlad/simulated_features/5sec_segment_feature/400/sim'
    k = 16  # 사용할 클러스터수
    descriptor_set = getDescriptorSet(simulated_path)
    dictionary = getDictionary(descriptor_set, k)
    saveVLADVectorPth( simulated_path, dictionary)

    # path = ''
    # makepath = '/mldisk/nfs_shared_/my_vlad/simulated_features/5sec_segment_feature/resnet_rmac_testdb'
    # make_directory(path, makepath)