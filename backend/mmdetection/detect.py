# Copyright (c) OpenMMLab. All rights reserved.
import os
import cv2
import mmcv
import glob
from alive_progress import alive_bar
from argparse import ArgumentParser

from mmdet.apis import (inference_detector, init_detector, show_result_pyplot)


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('source', help='source')
    parser.add_argument('config', help='Config file')
    parser.add_argument('checkpoint', help='Checkpoint file')
    parser.add_argument('--out_path', default=None, help='Path to output')
    parser.add_argument('--device', default='cuda:0', help='Device used for inference')
    parser.add_argument('--palette', default='coco', choices=['coco', 'voc', 'citys', 'random'], help='Color palette used for visualization')
    parser.add_argument('--score-thr', type=float, default=0.3, help='bbox score threshold')
    args = parser.parse_args()
    return args


def single_image(args):
    model = init_detector(args.config, args.checkpoint, device=args.device)
    basename = os.path.basename(args.source)
    result = inference_detector(model, args.source)
    model.show_result(args.source, result, args.score_thr, show=False, win_name=basename, bbox_color=args.palette, text_color=args.palette, mask_color=None, out_file=os.path.join(args.out_path, "detect_" + basename))


def multi_image(args):
    model = init_detector(args.config, args.checkpoint, device=args.device)
    imgs = glob.glob(os.path.join(args.source, "*.jpg"))
    with alive_bar(len(imgs), ctrl_c=False, title=f'Detecting') as bar:
        for img in imgs:
            basename = os.path.basename(img)
            result = inference_detector(model, img)
            model.show_result(img, result, args.score_thr, show=False, win_name=basename, bbox_color=args.palette, text_color=args.palette, mask_color=None, out_file=os.path.join(args.out_path, "detect_" + basename))
            bar()


def single_video(args):
    import matplotlib
    matplotlib.use('agg')
    model = init_detector(args.config, args.checkpoint, device=args.device)
    video_reader = mmcv.VideoReader(args.source)
    video_writer = None

    basename = os.path.basename(args.source)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(os.path.join(args.out_path, "detect_" + basename), fourcc, video_reader.fps, (video_reader.width, video_reader.height))

    for frame in mmcv.track_iter_progress(video_reader):
        result = inference_detector(model, frame)
        frame = model.show_result(frame, result, score_thr=args.score_thr, bbox_color=args.palette, text_color=args.palette, mask_color=None)
        video_writer.write(frame)

    if video_writer:
        video_writer.release()
    cv2.destroyAllWindows()


def main(args):
    print(args)
    if os.path.isdir(args.source):
        multi_image(args)
    elif args.source.endswith(('.bmp', '.dib', '.png', '.jpg', '.jpeg', '.pbm', '.pgm', '.ppm', '.tif', '.tiff')):
        single_image(args)
    elif args.source.endswith(".mp4"):
        single_video(args)


if __name__ == '__main__':
    args = parse_args()
    main(args)