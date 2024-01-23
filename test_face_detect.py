from insightface_func.face_detect_crop_multi import Face_detect_crop
import cv2
import os

face_detection = Face_detect_crop(name='antelope', root='./insightface_func/models')
img = "/mnt/sdb/liwen/wav2lip_288x288/test_data/test1.png"

img_1 = "/mnt/sdb/liwen/wav2lip_288x288/test_data/out.jpg"
root_dir = "/mnt/sdb/liwen/wav2lip_288x288/test_data"
basename ="out"

face_detection.prepare(ctx_id = 0, det_thresh=0.6, det_size=(640,640), mode = None, crop_size=384, ratio=0.8)
img = cv2.imread(img)
bboxes = face_detection.get(img)  # 会得到一个bbox列表
print(len(bboxes))
# 一个循环对应一个路径
# img = img[bboxes[1]:bboxes[3], bboxes[0]:bboxes[2],:]
        #     width = bboxes[i][2] - bboxes[i][0]
        #     height = bboxes[i][3] - bboxes[i][1]
        #     if max(width, height) < self.min_size: