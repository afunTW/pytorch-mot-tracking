# pytorch-mot-tracking

Demo the multiple object tracking.

- model: pretrain YOLOv3 model from DarkNet with COCO data
- dataset: COCO

## Results 

### YOLOv3 detection

![dets](demo/pedestrian-1-dets.gif)

### SORT

![SORT](demo/pedestrian-1-sort.gif)

## Reference

- [YOLOv3 offical website (Darknet)](https://pjreddie.com/darknet/yolo/)
- [YOLOv3 pytorch 1.0.0 implementation (eriklindernoren/PyTorch-YOLOv3)](https://github.com/eriklindernoren/PyTorch-YOLOv3)
- [SORT tracking (abewley/sort)](https://github.com/abewley/sort)