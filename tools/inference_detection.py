import argparse
import logging
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


def detect_image(image, model, img_size=416, conf_threshold=0.8, nms_threshold=0.4):
    # resize and pad image
    ratio = min(img_size/image.size[0], img_size/image.size[1])
    new_w = round(image.size[0] * ratio)
    new_h = round(image.size[1] * ratio)
    image_transform = transforms.Compose([
        transforms.Resize((new_h, new_w)),
        transforms.Pad((
            max(int((new_h-new_w)/2), 0),
            max(int((new_w-new_h)/2), 0)), fill=(128, 128, 128)),
        transforms.ToTensor()        
    ])

    # convert image to Tensor
    Tensor = torch.cuda.FloatTensor
    tensor = image_transform(image).float()
    tensor.unsqueeze_(0)
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
    frame = frame[..., ::-1]
    frame = Image.fromarray(frame)
    logger.debug('detect frame %d' % 1)
    detections = detect_image(frame, model, img_size=args.img_size)
    print(detections)


if __name__ == '__main__':
    args = argparser().parse_args()
    main(args)
