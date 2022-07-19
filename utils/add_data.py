import csv
import os

data_root = os.path.expanduser('~//yq_pointnet//BosphorusDB')
csv_path = data_root + "/train.csv"
with open(csv_path, "a+") as f:
    csv_write = csv.writer(f)
    data_row = ['/home/yqli/yq_pointnet/AddData/face04242.txt', '105']
    for i in range(5):
        csv_write.writerow(data_row)

