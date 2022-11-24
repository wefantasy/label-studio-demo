label-studio-ml init model --script mmdetection/label_studio_backend.py --force
md .\model\mmdet
md .\model\model
md .\model\configs
md .\model\checkpoints
md .\model\tools
md .\model\workdir
xcopy .\mmdetection\mmdet .\model\mmdet /S /Y /Q
xcopy .\mmdetection\model .\model\model /S /Y /Q
xcopy .\mmdetection\configs .\model\configs  /S /Y /Q
xcopy .\mmdetection\checkpoints .\model\checkpoints  /S /Y /Q
xcopy .\mmdetection\tools .\model\tools  /S /Y /Q
copy .\mmdetection\conf.yaml .\model\conf.yaml
label-studio-ml start model --host 0.0.0.0 -p 8888