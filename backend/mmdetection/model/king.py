_base_ = '../configs/mask_rcnn/mask_rcnn_r50_caffe_fpn_mstrain-poly_1x_coco.py'
import os
# We also need to change the num_classes in head to match the dataset's annotation
model = dict(roi_head=dict(bbox_head=dict(num_classes=8), mask_head=dict(num_classes=8)))

# Modify dataset related settings
dataset_type = 'COCODataset'
classes = ('hero', 'soldier', 'Buff', 'grass', 'wild_monster', 'dragon', 'tower', 'crystal')
data_root = "D:/ProgramData/SyncThing/Work/000000创世纪/炼体/kingcoco128"
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=0,
    train=dict(img_prefix=data_root + '/images', classes=classes, ann_file=data_root + '/annotations/annotations.json'),
    val=dict(img_prefix=data_root + '/images', classes=classes, ann_file=data_root + '/annotations/annotations.json'),
    test=dict(img_prefix=data_root + '/images', classes=classes, ann_file=data_root + '/annotations/annotations.json'),
)
work_dir = os.path.join(data_root, 'work_dir')
runner = dict(type='EpochBasedRunner', max_epochs=100)
checkpoint_config = dict(interval=1)
# We can use the pre-trained Mask RCNN model to obtain higher performance
load_from = './checkpoints/mask_rcnn_r50_caffe_fpn_mstrain-poly_3x_coco_bbox_mAP-0.408__segm_mAP-0.37_20200504_163245-42aa3d00.pth'