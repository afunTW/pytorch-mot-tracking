# pytorch-mot-tracking

Demo the multiple object tracking.

- model: pretrain YOLOv3 model from DarkNet with COCO data
- dataset: COCO

| YOLOv3 detection | SORT |
| --- | --- |
| ![dets](demo/pedestrian-1-dets.gif) | ![SORT](demo/pedestrian-1-ans.gif)

## Reference

- [YOLOv3 offical website (Darknet)](https://pjreddie.com/darknet/yolo/)
- [YOLOv3 pytorch 1.0.0 implementation (eriklindernoren/PyTorch-YOLOv3)](https://github.com/eriklindernoren/PyTorch-YOLOv3)
- [SORT tracking (abewley/sort)](https://github.com/abewley/sort)