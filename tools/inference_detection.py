import argparse
import logging
import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path
from PIL import Image

import cv2
import sys
sys.path.append('..')

import torch
from torch.autograd import Variable
from torchvision import datasets, transforms
from tracker.utils import log_handler
from yolov3.models import Darknet
from yolov3.utils import load_classes, non_max_suppression
from datetime import datetime
from tracker.sort import SORT
from tqdm import tqdm


def detect_image(img, model, img_size=416, conf_threshold=0.8, nms_threshold=0.4):
    # resize and pad image
    ratio = min(img_size/img.size[0], img_size/img.size[1])
    imw = round(img.size[0] * ratio)
    imh = round(img.size[1] * ratio)
    img_transforms = transforms.Compose([
        transforms.Resize((imh, imw)),
        transforms.Pad((
            max(int((imh-imw)/2),0),
            max(int((imw-imh)/2),0)), fill=(128,128,128)),
        transforms.ToTensor(),
    ])

    # convert image to Tensor
    Tensor = torch.cuda.FloatTensor
    tensor = img_transforms(img).float()
    tensor = tensor.unsqueeze_(0)
    input_image = Variable(tensor.type(Tensor))

    # run inference on the model and get detections
    with torch.no_grad():
        detections = model(input_image)
        detections = non_max_suppression(detections, 80, conf_threshold, nms_threshold)
    return detections[0]


def argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config',
                        dest='config',
                        default='../cfg/yolov3.cfg',
                        help='darknet config file path')
    parser.add_argument('--checkpoint',
                        dest='checkpoint',
                        default='../darknet/cfg/yolov3.weights',
                        help='Pretrain model file path')
    parser.add_argument('--classname',
                        dest='classname',
                        default='../darknet/data/coco.names',
                        help='COCO dataset class name')
    parser.add_argument('--gpus', dest='gpus', default=1, type=int,
                        help='Number of GPUs for inference')
    parser.add_argument('--video', dest='video', required=True,
                        help='target vidoe to inference')
    parser.add_argument('--img-size', dest='img_size', default=416, type=int,
                        help='model input image size')
    parser.add_argument('--output-dir', dest='output_dir', default='../outputs',
                        help='output video directory')
    parser.add_argument('--nolog', dest='islog', action='store_false')
    parser.set_defaults(islog=True)
    return parser

def main(args: argparse.Namespace):
    # setting log and logger
    logname = ''
    if args.islog:
        logdir = Path('../logs')
        if not logdir.exists():
            logdir.mkdir(parents=True)
        now_dt = datetime.now()
        logname = '{:d}-{:d}-inference.log'.format(
            now_dt.strftime('%m%dT%H%M%S'), now_dt.microsecond)
        logname = str(logdir / logname)
    logger = logging.getLogger(__name__)
    log_handler(logger, logname=logname)
    logger.info(args)

    # prepare video IO
    cap = cv2.VideoCapture(args.video)
    video_nframe = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    video_w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    video_h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    logger.info('video h={}, w={}, fps={:3f}, nframe={}'.format(
        int(video_h), int(video_w), video_fps, int(video_nframe)))
    
    output_dir = Path(args.output_dir)
    if not output_dir.exists():
        output_dir.mkdir(parents=True)
    output_videoname = str(output_dir / '{}.avi'.format(Path(args.video).stem))
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_videoname, fourcc=fourcc, fps=int(video_fps), frameSize=(int(video_w), int(video_h)))

    # load model and weight
    logger.info('load model')
    model = Darknet(args.config, img_size=args.img_size)
    logger.info('load weight')
    model.load_weights(args.checkpoint)
    model.cuda()
    model.eval()
    classes = load_classes(args.classname)
    tracker = SORT()

    # draw setting
    cmap = plt.get_cmap('tab20b')
    bbox_palette = [cmap(i)[:3] for i in np.linspace(0, 1, 1000)]
    random.shuffle(bbox_palette)

    # loop over the video
    for frame_idx in tqdm(range(int(video_nframe))):
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(frame)

        # detection
        _start_time = datetime.now()
        detections = detect_image(image, model, img_size=args.img_size)
        _cost_time = datetime.now() - _start_time
        logger.debug('detect frame {} in {}, get detections {}'.format(
            frame_idx+1, str(_cost_time), detections.shape))

        # image and bbox transition
        frame = np.array(image)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        pad_x = max(frame.shape[0] - frame.shape[1], 0) * (args.img_size / max(frame.shape))
        pad_y = max(frame.shape[1] - frame.shape[0], 0) * (args.img_size / max(frame.shape))
        unpad_h = args.img_size - pad_y
        unpad_w = args.img_size - pad_x

        if detections is not None:
            tracked_detections = tracker.update(detections.cpu())
            unique_labels = detections[:, -1].cpu().unique()
            num_unique_labels = len(unique_labels)
            for x1, x2, y1, y2, obj_id, cls_pred in tracked_detections:
                box_h = int(((y2 - y1) / unpad_h) * frame.shape[0])
                box_w = int(((x2 - x1) / unpad_w) * frame.shape[1])
                y1 = int(((y1 - pad_y // 2) / unpad_h) * frame.shape[0])
                x1 = int(((x1 - pad_x // 2) / unpad_w) * frame.shape[1])
                label = classes[int(cls_pred)]
                color = bbox_palette[int(obj_id) % len(bbox_palette)]
                color = [i*255 for i in color]

                cv2.rectangle(frame,
                              (x1, y1),
                              (x1+box_w, y1+box_h),
                              color, 2)
                cv2.rectangle(frame,
                              (x1, y1-35),
                              (x1+len(label)*19+60, y1),
                              color, -1)
                cv2.putText(frame,
                            '{}-{}'.format(label, int(obj_id)),
                            (x1, y1-10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1, (255, 255, 255), 3)
        out.write(frame)
    cap.release()
    out.release()


if __name__ == '__main__':
    args = argparser().parse_args()
    main(args)
