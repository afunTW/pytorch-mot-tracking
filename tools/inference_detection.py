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
                        default='../darknet/cfg/yolov3.cfg',
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
        logname = '{}-{}-inference.log'.format(
            now_dt.strftime('%m%dT%H%M%S'), now_dt.microsecond)
        # logname = f"{now_dt.strftime('%m%dT%H%M%S')}-{now_dt.microsecond}-inference.log"
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

    # load model and weight
    logger.info('load model')
    model = Darknet(args.config, img_size=args.img_size)
    logger.info('load weight')
    model.load_weights(args.checkpoint)
    model.cuda()
    model.eval()
    classes = load_classes(args.classname)

    # test
    ok, frame = cap.read()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(frame)
    frame = np.array(image)
    pad_x = max(frame.shape[0] - frame.shape[1], 0) * (args.img_size / max(frame.shape))
    pad_y = max(frame.shape[1] - frame.shape[0], 0) * (args.img_size / max(frame.shape))
    unpad_h = args.img_size - pad_y
    unpad_w = args.img_size - pad_x

    _start_time = datetime.now()
    detections = detect_image(image, model, img_size=args.img_size)
    _cost_time = datetime.now() - _start_time
    logger.info(f"detect image in {_cost_time}")
    print(detections)

    # draw test
    # Get bounding-box colors
    cmap = plt.get_cmap('tab20b')
    bbox_palette = [cmap(i) for i in np.linspace(0, 1, 20)]
    out_image = image.copy()
    plt.figure()
    fig, ax = plt.subplots(1, figsize=(12, 9))
    ax.imshow(frame)

    # draw bbox and label
    if detections is not None:
        unique_labels = detections[:, -1].cpu().unique()
        num_unique_labels = len(unique_labels)
        bbox_colors = random.sample(bbox_palette, num_unique_labels)

        # browse detections and draw bbox
        for x1, y1, x2, y2, conf, cls_conf, cls_pred in detections:
            box_h = int(((y2 - y1) / unpad_h) * frame.shape[0])
            box_w = int(((x2 - x1) / unpad_w) * frame.shape[1])
            y1 = int(((y1 - pad_y // 2) / unpad_h) * frame.shape[0])
            x1 = int(((x1 - pad_x // 2) / unpad_w) * frame.shape[1])

            color = bbox_colors[int(np.where(unique_labels == int(cls_pred))[0])]
            bbox = patches.Rectangle((x1, y1), box_w, box_h,
                                     linewidth=2,
                                     edgecolor=color,
                                     facecolor='none')
            ax.add_patch(bbox)
            plt.text(x1, y1,
                     s=classes[int(cls_pred)],
                     color='white',
                     verticalalignment='top',
                     bbox={'color': color, 'pad': 0})
    plt.axis('off')
    plt.savefig('../demo/test.jpg', bbox_inches='tight', pad_inches=0.0)


if __name__ == '__main__':
    args = argparser().parse_args()
    main(args)
