import sys

if sys.version_info[0] < 3 and sys.version_info[1] < 2:
	raise Exception("Must be using >= Python 3.2")

from os import listdir, path
from insightface_func.face_detect_crop_multi import Face_detect_crop

import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
import argparse, os, cv2, traceback, subprocess
from tqdm import tqdm
from glob import glob
import audio
from hparams import hparams as hp

parser = argparse.ArgumentParser()

parser.add_argument('--ngpu', help='Number of GPUs across which to run in parallel', default=4, type=int)
parser.add_argument("--data_root", help="Root folder of the LRS2 dataset",  default="/mnt/sdb/liwen/data/Youtubev1_HR",required=False)
parser.add_argument("--preprocessed_root", help="Root folder of the preprocessed dataset", default="/mnt/sdb/liwen/data/Youtubev1_preprocessed" ,required=False)

args = parser.parse_args()

# fa = [face_detection.FaceAlignment(face_detection.LandmarksType._2D, flip_input=False, 
# 									device='cuda:{}'.format(id)) for id in range(args.ngpu)]

face_detection = [Face_detect_crop(name='antelope', root='./insightface_func/models', device='cuda:{}'.format(id)) for id in range(args.ngpu)]

template = 'ffmpeg -loglevel panic -y -i {} -strict -2 {}'
# template2 = 'ffmpeg -hide_banner -loglevel panic -threads 1 -y -i {} -async 1 -ac 1 -vn -acodec pcm_s16le -ar 16000 {}'
template2 = 'ffmpeg -i {} -vn -ar 16000 -ac 1 {}'


def process_video_file(vfile, args, gpu_id):
	video_stream = cv2.VideoCapture(vfile)
	frames = []
	while 1:
		still_reading, frame = video_stream.read()
		if not still_reading:
			video_stream.release()
			break
		frames.append(frame)
	
	vidname = os.path.basename(vfile).split('.')[0]
	dirname = vfile.split('/')[-2]

	fulldir = path.join(args.preprocessed_root, dirname, vidname)
	os.makedirs(fulldir, exist_ok=True)
	# 对每帧进行处理
	i = -1
	for fb in frames:  # 如果说frame里面有多个的人脸的话
		bboxes = face_detection[gpu_id].get(np.asarray(fb)) # 返回的是一个list,list中包含了几个人脸，每个人脸又是一个list，list里面是4个点加一个置信度
		for index, face in enumerate(bboxes):
			if len(face)!=1:
				continue
			else:
				face_crop = fb[face[1]:face[3], face[0]:face[2],:]  # H W C
				cv2.imwrite(path.join(fulldir, '{}.jpg'.format(index)), face_crop)  # 按index保存图片
				
def process_audio_file(vfile, args):
	vidname = os.path.basename(vfile).split('.')[0]
	dirname = vfile.split('/')[-2]

	fulldir = path.join(args.preprocessed_root, dirname, vidname)
	os.makedirs(fulldir, exist_ok=True)

	wavpath = path.join(fulldir, 'audio.wav')
	command = template2.format(vfile, wavpath)  # 16khz采样率
	# command += '&&'
	# command += template.format(vfile, wavpath)
	subprocess.call(command, shell=True)

	
def mp_handler(job):
	vfile, args, gpu_id = job
	try:
		process_video_file(vfile, args, gpu_id)
	except KeyboardInterrupt:
		exit(0)
	except:
		traceback.print_exc()
		
def main(args):
	print('Started processing for {} with {} GPUs'.format(args.data_root, args.ngpu))
	filelist = glob(path.join(args.data_root, '*/*.mp4'))
	jobs = [(vfile, args, i%args.ngpu) for i, vfile in enumerate(filelist)]
	p = ThreadPoolExecutor(args.ngpu)
	futures = [p.submit(mp_handler, j) for j in jobs]
	_ = [r.result() for r in tqdm(as_completed(futures), total=len(futures))]
	print('Dumping audios...')
	for vfile in tqdm(filelist):
		try:
			process_audio_file(vfile, args)
		except KeyboardInterrupt:
			exit(0)
		except:
			traceback.print_exc()
			continue

if __name__ == '__main__':
	main(args)