from tools.mytrain import train, MyDict

# args = parse_args()
# args.config = r"D:/CommonWorkspace/label-studio-ml-backend/mmdet/mymodel/fcos_common.py"
# data_root = r"D:/ProgramData/TempDir/train"
# args.cfg_options = {}
# args.cfg_options['data_root'] = r"D:/ProgramData/TempDir/train"
# args.cfg_options['runner'] = dict(type='EpochBasedRunner', max_epochs=1)
# args.cfg_options['data'] = dict(
#     train=dict(img_prefix=data_root, ann_file=data_root + '/result.json'),
#     val=dict(img_prefix=data_root, ann_file=data_root + '/result.json'),
#     test=dict(img_prefix=data_root, ann_file=data_root + '/result.json'),
# )
# args.cfg_options['load_from'] = r"C:/Users/Fantasy/Desktop/dataset/work_dir/latest.pth"
# args.work_dir = r"D:/ProgramData/TempDir/train/work_dir"
# train(args)

args = {}
args = MyDict()
args.config = r"D:/CommonWorkspace/label-studio-ml-backend/mmdet/mymodel/fcos_common.py"
data_root = r"D:/ProgramData/TempDir/train"
args.cfg_options = {}
args.cfg_options['data_root'] = r"D:/ProgramData/TempDir/train"
args.cfg_options['runner'] = dict(type='EpochBasedRunner', max_epochs=1)
args.cfg_options['data'] = dict(
    train=dict(img_prefix=data_root, ann_file=data_root + '/result.json'),
    val=dict(img_prefix=data_root, ann_file=data_root + '/result.json'),
    test=dict(img_prefix=data_root, ann_file=data_root + '/result.json'),
)
args.cfg_options['load_from'] = r"C:/Users/Fantasy/Desktop/dataset/work_dir/latest.pth"
args.work_dir = r"D:/ProgramData/TempDir/train/work_dir"
print(args)
train(args)