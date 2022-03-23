# SPTNet: Learning Spherical Point Transformation for Point Cloud Completion

This is the demonstrative code for CVPR submission 2686, in order to provide better understanding of network structure. The pretrained model is not contained due to size limitations of supplementary material, and will be released after the acceptance of this paper.
## Installation:
Our code is implemented in Python 3.5, PyTorch 1.2 and CUDA 10.0.
- Install python Dependencies
```python
cd 3D-AttriFlow
pip install -r requirements.txt
```

- Compile PyTorch 3rd-party modules.
```python
cd utils/ChamferDistancePytorch/chamfer3D
python setup.py install
cd -
cd utils/Pointnet2.PyTorch/pointnet2
python setup.py install
cd -
cd utils/emd
python setup.py install
cd -
```

## Dataset:
Single View Reconstruction
- Download ShapeNet data from : https://drive.google.com/drive/folders/1If_-t0Aw9Zps-gj5ttgaMSTqRwYms9Ag?usp=sharing
- unzip ShapeNetV1PointCloud.zip and unzip ShapeNetV1Renderings.zip to your data path
- make cache folders:
```
mkdir cache
mkdir cache_test
```
- You need to update the file path of the datasets in `cfgs/SVR.yaml` line 63:
```
pointcloud_path: 'Path/ShapeNetV1PointCloud'
image_path: 'Path/ShapeNetV1Renderings'
cache_path: 'Path/cache'
cache_path: 'Path/cache_test'
```

Point Cloud Completion
- Download MVP data from : https://drive.google.com/drive/folders/1XxZ4M_dOB3_OG1J6PnpNvrGTie5X9Vk_
- You need to update the file path of the datasets in `dataset_pc/dataset.py`:
```
if prefix=="train":
    self.file_path = 'Path/MVP_Train_CP.h5'
elif prefix=="val":
    self.file_path = 'Path/MVP_Test_CP.h5'
```

## Usage:
- To train a model: 
```python
python train_svr.py -c cfgs/SVR.yaml -gpu 0
or
python train_pc.py -c cfgs/PC.yaml -gpu 0
```
- To test a model:  
```python
python val_svr.py -c cfgs/SVR.yaml -gpu 0
or 
python val_pc.py -c cfgs/PC.yaml -gpu 0
```
