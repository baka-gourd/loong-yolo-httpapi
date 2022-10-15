import torch
import re
from torch.backends import cudnn

from models.common import DetectMultiBackend
from utils.dataloaders import LoadImages
from utils.general import Profile, check_img_size, check_imshow, non_max_suppression, scale_boxes, \
    xyxy2xywh
from utils.torch_utils import select_device, smart_inference_mode


class Detector:
    def __init__(self, weights="./loong.pt", imgsz=(640, 640), conf_thres=0.25, iou_thres=0.25, half=False):
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres

        self.device = select_device('')
        self.model = DetectMultiBackend(weights, self.device)
        stride, names, pt = self.model.stride, self.model.names, self.model.pt
        self.imgsz = check_img_size(imgsz, s=stride)  # check image size
        half &= pt and self.device.type != 'cpu'  # half precision only supported by PyTorch on CUDA
        self.half = half

        if pt:
            self.model.model.half() if half else self.model.model.float()
        self.view_img = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        self.model.warmup(imgsz=(1, 3, *self.imgsz))

    @smart_inference_mode()
    def detect(self, image, accuracy=0.8) -> bool:
        dataset = LoadImages(image, img_size=self.imgsz, stride=self.model.stride, auto=self.model.pt, vid_stride=1)
        bs = 1
        vid_path, vid_writer = [None] * bs, [None] * bs
        seen, windows, dt = 0, [], (Profile(), Profile(), Profile())

        for path, im, im0s, vid_cap, s in dataset:
            with dt[0]:
                im = torch.from_numpy(im).to(self.model.device)
                im = im.half() if self.model.fp16 else im.float()  # uint8 to fp16/32
                im /= 255  # 0 - 255 to 0.0 - 1.0
                if len(im.shape) == 3:
                    im = im[None]  # expand for batch dim

            # Inference
            with dt[1]:
                visualize = False
                pred = self.model(im, augment=False, visualize=visualize)

            # NMS
            with dt[2]:
                pred = non_max_suppression(pred, self.conf_thres, self.iou_thres, None, False, max_det=100)

            # Second-stage classifier (optional)
            # pred = utils.general.apply_classifier(pred, classifier_model, im, im0s)

            # Process predictions
            for i, det in enumerate(pred):  # per image
                seen += 1
                p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)
                s += '%gx%g ' % im.shape[2:]  # print string
                gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
                if len(det):
                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()
                    result = []
                    # Write results
                    for *xyxy, conf, cls in reversed(det):
                        # xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        result.append(float(f'{conf:.2f}'))
                    for r in result:
                        if r > accuracy:
                            return True
                    return False
                else:
                    return False
