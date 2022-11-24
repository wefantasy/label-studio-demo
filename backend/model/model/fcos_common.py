_base_ = '../configs/fcos/fcos_r50_caffe_fpn_gn-head_mstrain_640-800_2x_coco.py'

import os
# We also need to change the num_classes in head to match the dataset's annotation
model = dict(bbox_head=dict(num_classes=1))

# Modify dataset related settings
dataset_type = 'COCODataset'
classes = ('face',)
data_root = "C:/Users/Fantasy/Desktop/dataset"
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=0,
    train=dict(img_prefix=data_root + '/images', classes=classes, ann_file=data_root + '/result.json'),
    val=dict(img_prefix=data_root + '/images', classes=classes, ann_file=data_root + '/result.json'),
    test=dict(img_prefix=data_root + '/images', classes=classes, ann_file=data_root + '/result.json'),
)
work_dir = os.path.join(data_root, 'work_dir')
runner = dict(type='EpochBasedRunner', max_epochs=100)
checkpoint_config = dict(interval=5)
# We can use the pre-trained Mask RCNN model to obtain higher performance
load_from = '../checkpoints/fcos_r50_caffe_fpn_gn-head_mstrain_640-800_2x_coco-d92ceeea.pth'