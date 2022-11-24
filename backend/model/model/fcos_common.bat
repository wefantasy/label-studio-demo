@REM python ../tools/misc/print_config.py fcos_common.py

@REM python tools/misc/browse_dataset.py train/king.py --show-interval 5

@REM python ../tools/train.py fcos_common.py

@REM python tools/test.py train/king.py D:\ProgramData\SyncThing\Work\000000创世纪\炼体\kingcoco128\work_dir\latest.pth --show

@REM python tools/analysis_tools/analyze_logs.py plot_curve ^
@REM     D:\ProgramData\SyncThing\Work\000000创世纪\炼体\kingcoco128\work_dir\20220429_124143.log.json ^
@REM     --keys loss_cls loss_bbox --legend loss_cls loss_bbox

python ../detect.py --out_path D:\ProgramData\SyncThing\Work\000000创世纪\炼体\kingcoco128\work_dir\detect ^
    D:\ProgramData\SyncThing\Work\000000创世纪\炼体\kingcoco128\test.mp4 ^
    fcos_common.py ^
    D:\ProgramData\SyncThing\Work\000000创世纪\炼体\kingcoco128\work_dir\latest.pth 