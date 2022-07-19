
import os
import random
import numpy as np
import sys
random.seed(7122)
sys.path.append('../')


def check_num_pc_changed(adv, ori):
    logits_mtx = np.logical_and.reduce(adv == ori, axis=0)
    return np.sum(logits_mtx == False)

path1 = "adv_face0424smile0513_rebuild_digtal(1).txt"
test_data_path1 = os.path.expanduser("~//yq_pointnet//test_face_data/" + path1)
ipt_ori1 = np.loadtxt(test_data_path1, delimiter=',')
ipt1 = ipt_ori1[0:4000]
print(ipt1)
#ipt1 = np.expand_dims(ipt1, 0)

path1 = "adv_face0424smile1.txt"
test_data_path2 = os.path.expanduser("~//yq_pointnet//test_face_data/" + path1)
ipt_ori2 = np.loadtxt(test_data_path2, delimiter=' ')
ipt2 = ipt_ori2[0:4000]
#ipt2 = np.expand_dims(ipt2, 0)
print(ipt2)
ipt = []
for i in range(4000):
    for j in range(4000):
        if ipt1[i][3] == ipt2[j][3] and ipt1[i][4] == ipt2[j][4]:
            ipt.append(ipt1[i]-ipt2[j])


np.savetxt(os.path.expanduser("~//yq_pointnet//test_face_data/" + "new_"+path1), ipt, fmt='%.04f')
print(ipt)


