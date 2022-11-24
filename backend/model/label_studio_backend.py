import os
import yaml
import json
import time
import shutil
import requests
import numpy as np
from label_studio_ml.model import LabelStudioMLBase
from label_studio_ml.utils import get_image_size, get_single_tag_keys
from label_studio_tools.core.utils.io import get_local_path
from label_studio_ml.utils import get_env

from mmdet.apis import init_detector, inference_detector
import mmcv

ROOT = os.path.join(os.path.dirname(__file__))
print('=> ROOT = ', ROOT)
os.environ['HOSTNAME'] = 'http://localhost:80'
os.environ['API_KEY'] = '37edbb42f1b3a73376548ea6c4bc7b3805d63453'
HOSTNAME = get_env('HOSTNAME')
API_KEY = get_env('API_KEY')

print('=> LABEL STUDIO HOSTNAME = ', HOSTNAME)
if not API_KEY:
    print('=> WARNING! API_KEY is not set')

with open(os.path.join(ROOT, "conf.yaml"), errors='ignore') as f:
    conf = yaml.safe_load(f)
print("================================config file:")
print(conf)


class MyModel(LabelStudioMLBase):

    def __init__(self, **kwargs):
        super(MyModel, self).__init__(**kwargs)
        print("================================__init__()")
        # Load model
        if self.train_output:
            self.detector = init_detector(conf['config_file'], self.train_output['model_path'], device=conf['device'])
        else:
            self.detector = init_detector(conf['config_file'], conf['checkpoint_file'], device=conf['device'])
        self.CLASSES = self.detector.CLASSES
        self.from_name, self.to_name, self.value, self.labels_in_config = get_single_tag_keys(self.parsed_label_config, 'RectangleLabels', 'Image')  # 前端获取任务属性
        self.labels_in_config = set(self.labels_in_config)  # 前端配置的标签列表

    def predict(self, tasks, **kwargs):
        print("================================predict()")
        print("images: ", kwargs)
        images = [get_local_path(task['data'][self.value], hostname=HOSTNAME, access_token=API_KEY) for task in tasks]
        for image_path in images:
            w, h = get_image_size(image_path)
            # 推理演示图像
            img = mmcv.imread(image_path)
            result = inference_detector(self.detector, img)
            bboxes = np.vstack(result)
            labels = [np.full(bbox.shape[0], i, dtype=np.int32) for i, bbox in enumerate(result)]
            labels = np.concatenate(labels)
            # FCOS算法结果会多出来两个分数极低的检测框，需要将其过滤掉
            scores = bboxes[:, -1]
            score_thr = 0.3
            inds = scores > score_thr
            bboxes = bboxes[inds, :]
            labels = labels[inds]
            results = []  # results需要放在list中再返回
            for id, bbox in enumerate(bboxes):
                label = self.CLASSES[labels[id]]
                if label not in self.labels_in_config:
                    print(label + ' label not found in project config.')
                    continue
                results.append({
                    'id': str(id),
                    'from_name': self.from_name,
                    'to_name': self.to_name,
                    'type': 'rectanglelabels',
                    'value': {
                        'rectanglelabels': [label],
                        'x': bbox[0] / w * 100,
                        'y': bbox[1] / h * 100,
                        'width': (bbox[2] - bbox[0]) / w * 100,
                        'height': (bbox[3] - bbox[1]) / h * 100
                    },
                    'score': float(bbox[4] * 100)
                })
            avgs = bboxes[:, -1]
            results = [{'result': results, 'score': np.average(avgs) * 100}]
            return results

    def fit(self, completions, batch_size=32, num_epochs=5, **kwargs):
        print("================================fit()")
        if completions:
            image_urls, image_labels = [], []
            for completion in completions:
                project_id = completion['project']
                u = completion['data'][self.value]
                image_urls.append(get_local_path(u, hostname=HOSTNAME, access_token=API_KEY))
                image_labels.append(completion['annotations'][0]['result'][0]['value'])
        elif kwargs.get('data'):
            project_id = kwargs['data']['project']['id']
            if not self.parsed_label_config:
                self.load_config(kwargs['data']['project']['label_config'])
        if self.gen_train_data(project_id):
            from tools.mytrain import MyDict, train
            args = MyDict()
            args.config = conf['config_file']
            data_root = os.path.join(conf['workdir'], "train")
            args.cfg_options = {}
            args.cfg_options['data_root'] = data_root
            args.cfg_options['runner'] = dict(type='EpochBasedRunner', max_epochs=num_epochs)
            args.cfg_options['data'] = dict(
                train=dict(img_prefix=data_root, ann_file=data_root + '/result.json'),
                val=dict(img_prefix=data_root, ann_file=data_root + '/result.json'),
                test=dict(img_prefix=data_root, ann_file=data_root + '/result.json'),
            )
            args.cfg_options['load_from'] = conf['checkpoint_file']
            args.work_dir = os.path.join(data_root, "work_dir")
            train(args)
            checkpoint_name = time.strftime("%Y%m%d%H%M%S", time.localtime(time.time())) + ".pth"
            shutil.copy(os.path.join(args.work_dir, "latest.pth"), os.path.join(conf['workdir'], checkpoint_name))
            print("model train complete!")
            return {'model_path': os.path.join(conf['workdir'], checkpoint_name)}
        else:
            raise "gen_train_data error"

    def gen_train_data(self, project_id):
        import zipfile
        import glob
        download_url = f'{HOSTNAME.rstrip("/")}/api/projects/{project_id}/export?export_type=COCO&download_all_tasks=false&download_resources=true'
        response = requests.get(download_url, headers={'Authorization': f'Token {API_KEY}'})
        zip_path = os.path.join(conf['workdir'], "train.zip")
        train_path = os.path.join(conf['workdir'], "train")

        with open(zip_path, 'wb') as file:
            file.write(response.content)  # 通过二进制写文件的方式保存获取的内容
            file.flush()
        f = zipfile.ZipFile(zip_path)  # 创建压缩包对象
        f.extractall(train_path)  # 压缩包解压缩
        f.close()
        os.remove(zip_path)
        if not os.path.exists(os.path.join(train_path, "images", str(project_id))):
            os.makedirs(os.path.join(train_path, "images", str(project_id)))
        for img in glob.glob(os.path.join(train_path, "images", "*.jpg")):
            basename = os.path.basename(img)
            shutil.move(img, os.path.join(train_path, "images", str(project_id), basename))
        return True