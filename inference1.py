from os import listdir, path
import numpy as np
import scipy, cv2, os, sys, argparse, audio
import json, subprocess, random, string
from tqdm import tqdm
from glob import glob
import torch, face_detection
from models import Wav2Lip
from gfpgan import GFPGANer
import platform
import hashlib
import time

parser = argparse.ArgumentParser(description='Inference code to lip-sync videos in the wild using Wav2Lip models')

parser.add_argument('--checkpoint_path', type=str,
					help='Name of saved checkpoint to load weights from', required=True)

parser.add_argument('--face', type=str,
					help='Filepath of video/image that contains faces to use', required=True)
parser.add_argument('--audio', type=str,
					help='Filepath of video/audio file to use as raw audio source', required=True)

parser.add_argument('--outpath', type=str, help='output file path', default='out')

parser.add_argument('--pads', nargs='+', type=int, default=[0, 10, 0, 0],
					help='Padding (top, bottom, left, right). Please adjust to include chin at least')

parser.add_argument('--wav2lip_batch_size', type=int, help='Batch size for Wav2Lip model(s)', default=128)

parser.add_argument('--nosmooth', default=False, action='store_true',
					help='Prevent smoothing face detections over a short temporal window')

starting_time = time.time()
args = parser.parse_args()
args.img_size = 96

def md5sum(f):
	m = hashlib.md5()
	n = 1024 * 8
	inp = open(f, 'rb')
	try:
		while True:
			buf = inp.read(n)
			if not buf:
				break
			m.update(buf)
	finally:
		inp.close()

	return m.hexdigest()

def Laplacian_Pyramid_Blending_with_mask(A, B, m, num_levels=6):
	# generate Gaussian pyramid for A,B and mask
	GA = A.copy()
	GB = B.copy()
	GM = m.copy()
	gpA = [GA]
	gpB = [GB]
	gpM = [GM]
	for i in range(num_levels):
		GA = cv2.pyrDown(GA)
		GB = cv2.pyrDown(GB)
		GM = cv2.pyrDown(GM)
		gpA.append(np.float32(GA))
		gpB.append(np.float32(GB))
		gpM.append(np.float32(GM))

	# generate Laplacian Pyramids for A,B and masks
	lpA = [gpA[num_levels - 1]]  # the bottom of the Lap-pyr holds the last (smallest) Gauss level
	lpB = [gpB[num_levels - 1]]
	gpMr = [gpM[num_levels - 1]]
	for i in range(num_levels - 1, 0, -1):
		# Laplacian: subtarct upscaled version of lower level from current level
		# to get the high frequencies
		LA = np.subtract(gpA[i - 1], cv2.pyrUp(gpA[i]))
		LB = np.subtract(gpB[i - 1], cv2.pyrUp(gpB[i]))
		lpA.append(LA)
		lpB.append(LB)
		gpMr.append(gpM[i - 1])  # also reverse the masks

	# Now blend images according to mask in each level
	LS = []
	for la, lb, gm in zip(lpA, lpB, gpMr):
		gm = gm[:, :, np.newaxis]
		ls = la * gm + lb * (1.0 - gm)
		LS.append(ls)

	# now reconstruct
	ls_ = LS[0]
	for i in range(1, num_levels):
		ls_ = cv2.pyrUp(ls_)
		ls_ = cv2.add(ls_, LS[i])
	return ls_


def merge_face(f, p, c):
	# f 原图， p 人脸
	y1, y2, x1, x2 = c
	w = x2 - x1
	h = y2 - y1

	pad = 512
	while pad < min(2048, max(w, h)):
		pad = pad * 2
	h_pad = int((pad - h) / 2)
	w_pad = int((pad - w) / 2)
	height, width = f.shape[:2]

	yy1 = y1 - h_pad
	if yy1 < 0:
		yy1 = 0
	yy2 = yy1 + pad
	if yy2 > height:
		yy2 = height
		yy1 = height - pad

	xx1 = x1 - w_pad
	if xx1 < 0:
		xx1 = 0
	xx2 = xx1 + pad
	if xx2 > width:
		xx2 = width
		xx1 = width - pad

	#print(y1, y2, h, h_pad, yy1, yy2)
	#print(x1, x2, w, w_pad, xx1, xx2)
	b = (yy1, yy2, xx1, xx2)
	ff = f.copy()[yy1: yy2, xx1: xx2]

	mask = np.zeros_like(ff)
	face = np.zeros_like(ff)
	face[y1-yy1: y2-yy1, x1-xx1: x2-xx1] = p
	mask[y1-yy1+15: y2-yy1-15, x1-xx1+15: x2-xx1-15] = 255
	mask = cv2.GaussianBlur(mask, (15, 15), 0)
	mask = mask / 255
	# cv2.imwrite("ff.jpg", ff)
	# cv2.imwrite("face.jpg", face)
	# cv2.imwrite("mask.jpg", mask*255)

	ff = np.float32(ff)
	face = np.float32(face)
	mask = np.float32(mask[:, :, 0])
	# noinspection PyTypeChecker
	img = Laplacian_Pyramid_Blending_with_mask(face, ff, mask, 1)
	f[yy1: yy2, xx1: xx2] = img
	return f

def get_smoothened_boxes(boxes, t):
	for i in range(len(boxes)):
		if i + t > len(boxes):
			window = boxes[len(boxes) - t:]
		else:
			window = boxes[i : i + t]
		boxes[i] = np.mean(window, axis=0)
	return boxes

def face_detect(detector, images):
	while 1:
		predictions = []
		try:
			predictions.extend(detector.get_detections_for_batch(np.array(images)))
		except RuntimeError:
			print('Recovering from OOM error')
			continue
		break

	results = []
	pady1, pady2, padx1, padx2 = args.pads
	for rect, image in zip(predictions, images):
		if rect is None:
			#cv2.imwrite('temp/faulty_frame.jpg', image) # check this frame where the face was not detected.
			#raise ValueError('Face not detected! Ensure the video contains a face in all the frames.')
			results.append([0, 0, 0, 0])
		else:
			y1 = max(0, rect[1] - pady1)
			y2 = min(image.shape[0], rect[3] + pady2)
			x1 = max(0, rect[0] - padx1)
			x2 = min(image.shape[1], rect[2] + padx2)
			results.append([x1, y1, x2, y2])

	boxes = np.array(results)
	if not args.nosmooth: boxes = get_smoothened_boxes(boxes, t=5)
	results = [[image[y1: y2, x1:x2], (y1, y2, x1, x2)] for image, (x1, y1, x2, y2) in zip(images, boxes)]

	return results

mel_step_size = 16
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Using {} for inference.'.format(device))

def _load(checkpoint_path):
	if device == 'cuda':
		checkpoint = torch.load(checkpoint_path)
	else:
		checkpoint = torch.load(checkpoint_path, map_location=lambda storage, loc: storage)
	return checkpoint

def load_model(p):
	model = Wav2Lip()
	print("Load checkpoint from: {}".format(p))
	checkpoint = _load(p)
	s = checkpoint["state_dict"]
	new_s = {}
	for k, v in s.items():
		new_s[k.replace('module.', '')] = v
	model.load_state_dict(new_s)

	model = model.to(device)
	return model.eval()

def main():
	if not os.path.isfile(args.face):
		raise ValueError('--face argument must be a valid path to video/image file')

	video_md5 = md5sum(args.face)
	args.outpath = os.path.join(args.outpath, f"{video_md5}")
	video_stream = cv2.VideoCapture(args.face)
	_, frame = video_stream.read()
	frame_h, frame_w = frame.shape[:-1]
	print(frame_h, frame_w)

	if not os.path.isdir(args.outpath):
		os.makedirs(args.outpath)

	args.resize_factor = 1
	'''
	if frame_w > frame_h:
		if frame_h > 1080:
			args.resize_factor = int(frame_h/1080)
	else:
		if frame_w > 1080:
			args.resize_factor = int(frame_w/1080)
	'''

	video_stream.release()
	video_stream = cv2.VideoCapture(args.face)
	fps = video_stream.get(cv2.CAP_PROP_FPS)
	args.fps = fps
	cost_time()

	if not args.audio.endswith('.wav'):
		print('Extracting raw audio...')
		command = 'ffmpeg -hwaccel auto -i {} -strict -2 -y {}'.format(args.audio, os.path.join(args.outpath, 'temp.wav'))
		args.audio = os.path.join(args.outpath, 'temp.wav')
		subprocess.call(command, shell=True)

	wav = audio.load_wav(args.audio, 16000)
	mel = audio.melspectrogram(wav)
	print(mel.shape)

	if np.isnan(mel.reshape(-1)).sum() > 0:
		raise ValueError('Mel contains nan! Using a TTS voice? Add a small epsilon noise to the wav file and try again')

	mel_chunks = []
	mel_idx_multiplier = 80./fps
	i = 0
	while 1:
		start_idx = int(i * mel_idx_multiplier)
		if start_idx + mel_step_size > len(mel[0]):
			mel_chunks.append(mel[:, len(mel[0]) - mel_step_size:])
			break
		mel_chunks.append(mel[:, start_idx : start_idx + mel_step_size])
		i += 1

	print("Length of mel chunks: {}".format(len(mel_chunks)))
	cost_time()

	restorer = GFPGANer(
		model_path='checkpoints/GFPGANv1.4.pth', upscale=4)

	model = load_model(args.checkpoint_path)
	print ("Model loaded")

	print('Reading video frames...')

	out = cv2.VideoWriter(os.path.join(args.outpath, 'result.mp4'), 0x7634706d, fps, ((frame.shape[1]//args.resize_factor//2)*2, (frame.shape[0]//args.resize_factor//2)*2))
	detector = face_detection.FaceAlignment(face_detection.LandmarksType._2D, flip_input=False, device=device)
	img_batch, mel_batch, frame_batch, coords_batch = [], [], [], []
	bar = tqdm(total=len(mel_chunks))
	i = 0
	while i < len(mel_chunks):
		still_reading, frame = video_stream.read()
		if not still_reading:
			video_stream.release()
			break
		if args.resize_factor > 1:
			frame = cv2.resize(frame, ((frame.shape[1]//args.resize_factor//2)*2, (frame.shape[0]//args.resize_factor//2)*2))
		face_det_result = face_detect(detector, [frame])
		face, coords = face_det_result[0]
		if coords[0]==0 and coords[1]==0 and coords[2]==0 and coords[3]==0:
			face = frame[0:args.img_size, 0:args.img_size]
		else:
			face = cv2.resize(face, (args.img_size, args.img_size))

		img_batch.append(face)
		mel_batch.append(mel_chunks[i])
		frame_batch.append(frame)
		coords_batch.append(coords)
		if len(img_batch) >= args.wav2lip_batch_size:
			img_batch, mel_batch = np.asarray(img_batch), np.asarray(mel_batch)

			img_masked = img_batch.copy()
			img_masked[:, args.img_size//2:] = 0

			img_batch = np.concatenate((img_masked, img_batch), axis=3) / 255.
			mel_batch = np.reshape(mel_batch, [len(mel_batch), mel_batch.shape[1], mel_batch.shape[2], 1])

			img_batch = torch.FloatTensor(np.transpose(img_batch, (0, 3, 1, 2))).to(device)
			mel_batch = torch.FloatTensor(np.transpose(mel_batch, (0, 3, 1, 2))).to(device)

			with torch.no_grad():
				pred = model(mel_batch, img_batch)

			pred = pred.cpu().numpy().transpose(0, 2, 3, 1) * 255.

			for p, f, c in zip(pred, frame_batch, coords_batch):
				y1, y2, x1, x2 = c
				if y1!=0 and x1!=0 and y2!=0 and x2!=0:
					_, _, p = restorer.enhance(p, only_center_face=True)
					p = cv2.resize(p.astype(np.uint8), (x2 - x1, y2 - y1))
					pp = merge_face(f, p, c)
					out.write(pp)
					#f[y1:y2, x1:x2] = p
				else:
					out.write(f)
			img_batch, mel_batch, frame_batch, coords_batch = [], [], [], []

		bar.update(1)
		i += 1

	if len(img_batch) > 0:
		img_batch, mel_batch = np.asarray(img_batch), np.asarray(mel_batch)

		img_masked = img_batch.copy()
		img_masked[:, args.img_size//2:] = 0

		img_batch = np.concatenate((img_masked, img_batch), axis=3) / 255.
		mel_batch = np.reshape(mel_batch, [len(mel_batch), mel_batch.shape[1], mel_batch.shape[2], 1])

		img_batch = torch.FloatTensor(np.transpose(img_batch, (0, 3, 1, 2))).to(device)
		mel_batch = torch.FloatTensor(np.transpose(mel_batch, (0, 3, 1, 2))).to(device)

		with torch.no_grad():
			pred = model(mel_batch, img_batch)

		pred = pred.cpu().numpy().transpose(0, 2, 3, 1) * 255.

		for p, f, c in zip(pred, frame_batch, coords_batch):
			y1, y2, x1, x2 = c
			if y1!=0 and x1!=0 and y2!=0 and x2!=0:
				_, _, p = restorer.enhance(p, only_center_face=True)
				p = cv2.resize(p.astype(np.uint8), (x2 - x1, y2 - y1))
				pp = merge_face(f, p, c)
				out.write(pp)
				#f[y1:y2, x1:x2] = p
			else:
				out.write(f)

	out.release()
	print ("Number of frames available for inference: ", i)
	cost_time()

	command = 'ffmpeg -i {} -i {} -y {}'.format(args.audio, os.path.join(args.outpath, 'result.mp4'), os.path.join(args.outpath, 'result_voice.mp4'))
	subprocess.call(command, shell=platform.system() != 'Windows')
	#os.remove(os.path.join(args.outpath, 'temp.wav'))
	#os.remove(os.path.join(args.outpath, 'result.avi'))
	cost_time()

def cost_time():
	total_seconds = time.time() - starting_time
	hours, remainder = divmod(total_seconds, 3600)
	minutes, seconds = divmod(remainder, 60)
	print('coast:%d:%d:%ds' % (hours, minutes, seconds))

if __name__ == '__main__':
	main()
