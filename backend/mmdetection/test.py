from mmdet.apis import init_detector, inference_detector
import mmcv

config_file = r'D:\CommonWorkspace\mmdetection\mymodel\king.py'
checkpoint_file = r'D:\ProgramData\SyncThing\Work\000000创世纪\炼体\kingcoco128\work_dir\latest.pth'
device = 'cuda:0'
# 初始化检测器
model = init_detector(config_file, checkpoint_file, device=device)
# 推理演示图像
img = mmcv.imread(r'D:\ProgramData\SyncThing\Work\000000创世纪\炼体\kingcoco128\images\0.jpg')
result = inference_detector(model, img)
model.show_result(img, result)
model.show_result(img, result,mask_color=None, out_file=r'D:\CommonWorkspace\mmdetection\output\output.jpg')