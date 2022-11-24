label-studio-ml init yolov5_backend --script yolov5/label_studio_backend.py --force
label-studio-ml start yolov5_backend --host 0.0.0.0 -p 8888

一种访问权限不允许的方式做了一个访问套接字的尝试。
-p 8888


Can't connect to ML backend http://127.0.0.1:8888/, health check failed. Make sure it is up and your firewall is properly configured.
http://198.18.0.1:8888/

FileNotFoundError: Can't resolve url, neither hostname or project_dir passed: /data/upload/1/db8f065a-000001.jpg

results.append({
                    'from_name': self.from_name,
                    'to_name': self.to_name,
                    'type': 'rectanglelabels',
                    'value': {
                        'rectanglelabels': [output_label],
                        'x': x / img_width * 100,   
                        'y': y / img_height * 100,
                        'width': (xmax - x) / img_width * 100,
                        'height': (ymax - y) / img_height * 100
                    },
                    'score': score
                })
x,y 为左上角坐标
w,h 为宽高

results.append({
                    'id': str(id),
                    'from_name': self.from_name,
                    'to_name': self.to_name,
                    'type': 'rectanglelabels',
                    'value': {
                        'rectanglelabels': ['0'],
                        'x': bbox[0] / w * 100,
                        'y': bbox[1]/ h * 100,
                        'width': bbox[2] / w * 100,
                        'height': bbox[3] / h * 100
                    },
                    'score': float(bbox[4] * 100)
                })
其中 id 必须为 str，否则前端不显示