## _attack_
四个攻击算法:

[CTA](https://arxiv.org/abs/2110.04158) |
[CW](https://arxiv.org/abs/1809.07016) |
[ISO](https://arxiv.org/abs/2002.12222) (unfinished)|
[GeoA3](https://arxiv.org/abs/1912.11171) (unfinished)

    Eval_CTA/CW/ISO/GeoA3.py 测试攻击算法在两个人脸数据集上的效果，生成的对抗性点云保存在AdvData里

    Test_CTA/CW/ISO/GeoA3.py 攻击给定的单个点云数据，保存得到的对抗性点云 (保存在test_face_data文件夹)

    ..//AdvData//(DGCNN/PointNet/MSG/SSG/) 保存数据集的adv数据

## _BosphorusDB_
    Bosphorus人脸数据集

## _cls_
    训练过的人脸分类模型
    保存格式: cls//Bosphorus//PointNet_model_on_bosphorus.pth

## _dataset_
    三个训练时用的dataset
    AdvData_dataset.py 数据集为攻击过后生成的点云，用来做迁移性的实验
    bosphorus_dataset.py 数据集为Bosphorus
    eurecom_dataset.py 数据集为EURECOM

## _EURECOM_Kinect_Face_Dataset_
    EURECOM数据集

## _model_
    dgcnn.py          ------->DGCNN
    pointnet.py       ------->PointNet
    pointnet2_MSG.py  ------->PointNet++(MSG)
    pointnet2_SSG.py  ------->PointNet++(SSG)
    pointnet2_utils.py        

## _utils_
    get_bosphorus_csv.py 对Bosphorus数据建立train.csv test.csv (7:3)
    get_eurecom_csv.py   对EURECOM数据建立train.csv test.csv (7:3)
    readbnt.py           读取Bosphorus数据集的数据
    add_data.py          在数据集中添加新的点云数据
## _others_
    train.py 训练人脸分类模型 训练过的模型保存在../cls/里

# 人脸分类任务
On Bosphorus:

| model | Overall Acc | 
| :---: | :---: | 
| PointNet | -1 | 
| PointNet++(MSG) | -1 | 
| PointNet++(SSG)  | -1 | 
| DGCNN  | -1 | 


On EURECOM:

| model | Overall Acc | 
| :---: | :---: | 
| PointNet | -1 | 
| PointNet++(MSG) | -1 | 
| PointNet++(MSG)  | -1 | 
| DGCNN  | -1 | 

# 对抗性攻击
CW attack On Bosphorus:

| attack Acc | PointNet | PointNet++(MSG) | PointNet++(SSG) | DGCNN |
| :---: | :---: | :---: | :---: | :---: | 
| PointNet | -1 | -1 | -1 | -1 | 
| PointNet++(MSG) | -1 | -1 | -1 | -1 | 
| PointNet++(SSG)  | -1 | -1 | -1 | -1 | 
| DGCNN  | -1 | -1 | -1 | -1 | 

ISO attack On Bosphorus:

| attack Acc | PointNet | PointNet++(MSG) | PointNet++(SSG) | DGCNN |
| :---: | :---: | :---: | :---: | :---: | 
| PointNet | -1 | -1 | -1 | -1 | 
| PointNet++(MSG) | -1 | -1 | -1 | -1 | 
| PointNet++(SSG)  | -1 | -1 | -1 | -1 | 
| DGCNN  | -1 | -1 | -1 | -1 | 

CTA attack On Bosphorus:

| attack Acc | PointNet | PointNet++(MSG) | PointNet++(SSG) | DGCNN |
| :---: | :---: | :---: | :---: | :---: | 
| PointNet | -1 | -1 | -1 | -1 | 
| PointNet++(MSG) | -1 | -1 | -1 | -1 | 
| PointNet++(SSG)  | -1 | -1 | -1 | -1 | 
| DGCNN  | -1 | -1 | -1 | -1 | 

CW attack On EURECOM:

| attack Acc | PointNet | PointNet++(MSG) | PointNet++(SSG) | DGCNN |
| :---: | :---: | :---: | :---: | :---: | 
| PointNet | -1 | -1 | -1 | -1 | 
| PointNet++(MSG) | -1 | -1 | -1 | -1 | 
| PointNet++(SSG)  | -1 | -1 | -1 | -1 | 
| DGCNN  | -1 | -1 | -1 | -1 | 

ISO attack On EURECOM:

| attack Acc | PointNet | PointNet++(MSG) | PointNet++(SSG) | DGCNN |
| :---: | :---: | :---: | :---: | :---: | 
| PointNet | -1 | -1 | -1 | -1 | 
| PointNet++(MSG) | -1 | -1 | -1 | -1 | 
| PointNet++(SSG)  | -1 | -1 | -1 | -1 | 
| DGCNN  | -1 | -1 | -1 | -1 | 

CTA attack On EURECOM:

| attack Acc | PointNet | PointNet++(MSG) | PointNet++(SSG) | DGCNN |
| :---: | :---: | :---: | :---: | :---: | 
| PointNet | -1 | -1 | -1 | -1 | 
| PointNet++(MSG) | -1 | -1 | -1 | -1 | 
| PointNet++(SSG)  | -1 | -1 | -1 | -1 | 
| DGCNN  | -1 | -1 | -1 | -1 | 








