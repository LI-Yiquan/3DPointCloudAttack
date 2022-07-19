import os
import json
import time

import numpy as np
import pandas as pd

data_root = os.path.expanduser('~//yq_pointnet//BosphorusDB')
assert os.path.exists(data_root), 'Dataset directory not found: %s' % data_root

split_radio = (0.7, 0.3, 0.0)
assert sum(split_radio) == 1
min_num_images_per_class = 0

min_num_train_images_per_class = 0

random_seed = 23337
np.random.seed(random_seed)


def _get_data_of_one_class(_cls_name, shuffle=True):
    cls_sum =0
    _cls_dir = os.path.realpath(os.path.join(data_root, _cls_name))
    assert os.path.exists(_cls_dir) and os.path.isdir(_cls_dir)
    _data = []

    class_name = _cls_name[2:5]
    for file in os.listdir(_cls_dir):
        file_name, file_ext = os.path.splitext(file)
        if file_ext != '.bnt':
            continue

        _data.append([os.path.join(_cls_dir, file),
                      class_name])
        cls_sum=cls_sum+1
    print(cls_sum)
    if shuffle:
        np.random.shuffle(_data)
    return _data


# train_data, eval_data, test_data, dirty_data = [[]] * 4
train_data, eval_data, test_data, dirty_data = [], [], [], []
count = 0
for cls_name in os.listdir(data_root):
    # Skip if it is not a folder

    if not os.path.isdir(os.path.join(data_root, cls_name)):
        continue


    _cls_dir = os.path.realpath(os.path.join(data_root, cls_name))
    if len([lists for lists in os.listdir(_cls_dir) if os.path.isfile(os.path.join(_cls_dir, lists))]) < 10:
        continue

    cls_data = _get_data_of_one_class(cls_name, shuffle=True)
    if len(cls_data) <= min_num_images_per_class:
        dirty_data.extend(cls_data)
        continue


    num_train_images = max(min_num_train_images_per_class,
                           int(len(cls_data) * split_radio[0]))
    num_eval_images = int((len(cls_data) - num_train_images) *
                          split_radio[1] / (1 - split_radio[0]))
    num_test_images = len(cls_data) - num_train_images - num_eval_images

    train_data.extend(cls_data[:num_train_images])
    eval_data.extend(cls_data[num_train_images:num_train_images + num_eval_images])
    test_data.extend(cls_data[-num_test_images:])
    count = count + 1
    if count >= 20:
        break


train_data = np.array(train_data)
train_df = pd.DataFrame({'cloud_point_path': train_data[:, 0] if len(train_data) > 0 else [],
                         'cls_name': train_data[:, 1] if len(train_data) > 0 else []})
train_df.to_csv(os.path.join(data_root, 'train.csv'), index=False, sep=',')

eval_data = np.array(eval_data)
eval_df = pd.DataFrame({'cloud_point_path': eval_data[:, 0] if len(eval_data) > 0 else [],
                        'cls_name': eval_data[:, 1] if len(eval_data) > 0 else []})
eval_df.to_csv(os.path.join(data_root, 'eval.csv'), index=False, sep=',')

test_data = np.array(test_data)
test_df = pd.DataFrame({'cloud_point_path': test_data[:, 0] if len(test_data) > 0 else [],
                        'cls_name': test_data[:, 1] if len(test_data) > 0 else []})
test_df.to_csv(os.path.join(data_root, 'test.csv'), index=False, sep=',')

dirty_data = np.array(dirty_data)
dirty_df = pd.DataFrame({'cloud_point_path': dirty_data[:, 0] if len(dirty_data) > 0 else [],
                         'cls_name': dirty_data[:, 1] if len(dirty_data) > 0 else []})
dirty_df.to_csv(os.path.join(data_root, 'dirty.csv'), index=False, sep=',')
print("everything finished!")
print("count=", count)

# In[30]:


train_df.head()

# In[30]:


