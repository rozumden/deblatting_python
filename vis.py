import cv2
import numpy as np

def get_visim(H,F,M,I=None):
	Hvis = np.repeat(H[:,:,np.newaxis], 3, axis=2)/np.max(H)
	FM = np.r_[np.repeat(M[:,:,np.newaxis], 3, axis=2),F]
	FM = cv2.resize(FM,(0,0), fx=H.shape[0]/FM.shape[0], fy=H.shape[0]/FM.shape[0])
	visim = np.concatenate((Hvis, FM),1)
	if I is not None:
		visim = np.concatenate((I, visim),1)
	return visim

def imshow_nodestroy(im, inter=1):
	if inter == 1:
		cv2.imshow('image',im), cv2.waitKey(1)
	else:
		cv2.imshow('image',cv2.resize(im, (0,0), fx=inter, fy=inter, interpolation = cv2.INTER_NEAREST)), cv2.waitKey(1)


def imshow(im, wkey=0, inter=1):
	if inter == 1:
		cv2.imshow('image',im), cv2.waitKey(int(np.round(wkey*1000))), cv2.destroyAllWindows() 
	else:
		cv2.imshow('image',cv2.resize(im, (0,0), fx=inter, fy=inter, interpolation = cv2.INTER_NEAREST)), cv2.waitKey(int(np.round(wkey*1000))), cv2.destroyAllWindows()

def imshow6(im, wkey=0):
	imshow(im, wkey=wkey, inter=6)