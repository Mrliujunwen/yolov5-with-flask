# -*- coding: UTF-8 -*-


import cv2
import numpy as np
import torch

from models.common import DetectMultiBackend
from utils.augmentations import letterbox
from utils.general import check_img_size, non_max_suppression, scale_coords
from utils.torch_utils import select_device


class YOLOv5:

	# 初始化操作，加载模型
	def __init__(self, device=''):
		# 参数设置
		self.weights = './weights/best.pt'
		self.imgsz = 640
		self.iou_thres = 0.45
		self.conf_thres = 0.25
		self.data = './data/car.yaml'
		self.classes = None

		self.device = select_device(device)
		self.half = self.device != "cpu"

		self.model = DetectMultiBackend(self.weights, device=self.device, data=self.data)  # load FP32 model
		model = self.model
		stride, self.names, pt, jit, onnx, engine = model.stride, model.names, model.pt, model.jit, model.onnx, model.engine
		self.imgsz = check_img_size(self.imgsz, s=stride)  # check image size

		# Half
		# FP16 supported on limited backends with CUDA
		self.half = (pt or jit or onnx or engine) and self.device.type != 'cpu'
		if pt or jit:
			self.model.model.half() if self.half else self.model.model.float()

	def infer(self, inImg):
		img, ratio, _ = letterbox(inImg, new_shape=self.imgsz)
		new_shape = int(inImg.shape[1] * ratio[1]), int(inImg.shape[0] * ratio[0])
		img_copy = cv2.resize(inImg.copy(), new_shape, interpolation=cv2.INTER_LINEAR)

		img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
		img = np.ascontiguousarray(img)
		img = torch.from_numpy(img).to(self.device)
		img = img.half() if self.half else img.float()  # uint8 to fp16/32
		img /= 255.0  # 0 - 255 to 0.0 - 1.0
		if img.ndimension() == 3:
			img = img.unsqueeze(0)

		pred = self.model(img)
		# NMS
		pred = non_max_suppression(pred, self.conf_thres, self.iou_thres, classes=self.classes)

		bbox_xyxy = []
		confs = []
		cls_ids = []
		class_counts = {}

		for i, det in enumerate(pred):  # detections per image
			if det is not None and len(det):
				for c in det[:, -1].unique():
					n = (det[:, -1] == c).sum()  # detections per class
					class_counts[self.names[int(c)]] = int(n)
				det[:, :4] = scale_coords(img.shape[2:], det[:, :4], inImg.shape).round()
				for *xyxy, conf, cls in reversed(det):
					bbox_xyxy.append([int(x) for x in xyxy])
					confs.append(conf.item())
					cls_ids.append(int(cls.item()))

		return bbox_xyxy, confs, cls_ids, class_counts, img_copy

	def infer_and_draw(self, frame):
		xyxys, confs, cls_ids, class_counts, frame_copy = self.infer(frame)
		for xyxy, conf, cls in zip(xyxys, confs, cls_ids):
			cv2.rectangle(frame, (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3])), (255, 0, 0), 2)
			cv2.putText(frame, f'{self.names[cls]} {conf:.2f}', (int(xyxy[0]), int(xyxy[1])),
			            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
		return xyxys, confs, cls_ids, class_counts, frame_copy


def video_detect_generator(model: YOLOv5, video_path, video_name):
	print(f'[INFO] Start video_detect_generator {video_name}')
	video = cv2.VideoCapture(video_path)
	ret, frame = video.read()
	fourcc = cv2.VideoWriter_fourcc(*'mp4v')
	fps = video.get(cv2.CAP_PROP_FPS)
	if fps == 0:
		fps = 30
	w, h = frame.shape[1], frame.shape[0]
	out = cv2.VideoWriter('./static/download/' + video_name, fourcc, fps, (w, h))
	while ret:
		xyxys, confs, cls_ids, class_counts, frame_copy = model.infer_and_draw(frame)
		out.write(frame)
		result = {
			'xyxys': xyxys,
			'confs': confs,
			'cls_ids': cls_ids,
			'class_counts': class_counts,
			'frame': frame,
			'frame_copy': frame_copy,
		}
		yield result

		ret, frame = video.read()

	print(f'[INFO] Finish video_detect_generator {video_name}')
	video.release()
	out.release()
	return 'finish'


if __name__ == '__main__':
	yolo = YOLOv5()
	video_generator = video_detect_generator(yolo, 0, '0.mp4')
	# result = next(video_generator)
	# print(result['xyxys'])
	# video_generator.close()
	# print('[INFO] Finish main')
	# result = next(video_generator)
	# print(result)
	count = 0
	while True:
		result = next(video_generator)
		print(result['frame_copy'].shape, count)
		count += 1
		cv2.imshow('frame', result['frame'])
		if cv2.waitKey(1) & 0xFF == ord('q'):
			break
	video_generator.close()
	print('[INFO] Finish main')
