# pytorch-mot-tracking

Demo the multiple object tracking.

## Results 

### SORT

**SORT = Kalman Filter + Hungarian matching**
No any Deep Learning tricks involved, if we replace to better detection model, the performance will simply enhance

- model: pretrain YOLOv3 model from DarkNet with COCO data
- dataset: COCO

![SORT](demo/pedestrian-1-sort.gif)

## Reference

- YOLOv3: An Incremental Improvement [[web]](https://pjreddie.com/darknet/yolo/) [[paper]](https://arxiv.org/abs/1804.02767) [[github]](https://github.com/pjreddie/darknet)
- Simple Online and Realtime Tracking [[paper]](https://arxiv.org/abs/1602.00763) [[github]](https://github.com/abewley/sort)
- Object detection and tracking in PyTorch (implementation) [[github]](https://github.com/cfotache/pytorch_objectdetecttrack) [[medium]](https://towardsdatascience.com/object-detection-and-tracking-in-pytorch-b3cf1a696a98)