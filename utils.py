import numpy as np
import cv2
import copy
from collections import Iterable
from scipy.cluster.vq import vq, kmeans, whiten
from collections import defaultdict

def best_fit_slope_and_intercept(xs, ys):
	m = ((np.mean(xs)*np.mean(ys)-np.mean(xs*ys)))/(np.mean(xs)**2 - np.mean(xs**2))
	b = np.mean(ys) - m*np.mean(xs)
	return m, b

def sortchars(bboxes, thresh=5):
	d={}
	xval=[]
	yval=[]
	for left, right, top, botm, _, _,clsname in bboxes:
		cx,cy = (left+right)/2,(top+botm)/2
		d[(cx,cy)] = clsname
		xval.append(cx)
		yval.append(cy)
		m,b=best_fit_slope_and_intercept(np.array(xval),np.array(yval))	
	dlist = list(d.keys())
	err=[y-(m*x+b) for x,y in dlist]
	condn = 0
	for num_cl in range(1,5):
		dist = kmeans(err,num_cl)		
		if dist[-1]<thresh:
			break
	clcodes=vq(err,list(range(num_cl)))[0] #get class codes for the points
	if num_cl != len(set(clcodes)):
		clcodes=vq(err,list(range(len(set(clcodes)))))[0]
	combined_dict = defaultdict(list)
	for i,j in zip(clcodes,dlist):
		combined_dict[i].append(j)
	yvals=[]
	for i in range(len(combined_dict)):
		if combined_dict[i]:
			yvals.append((np.mean(combined_dict[i],axis=0)[-1],i))
	strg=''
	for _,i in yvals:	
		combined_dict[i].sort()
		for j in combined_dict[i]:
			strg+=d[j]
	return strg

def resize_and_normalise_image(reqd_size,img):
	h, w, c = img.shape	
	mindim = np.min(img.shape[:-1])	
	maxdim = np.max(img.shape[:-1])	
	num = maxdim-mindim
	idx = img.shape.index(mindim)	
	if idx == 0:
		ypad = num
		xpad = 0
		# print("Padding y = ", ypad)
		padded_image=np.pad(img,((ypad,0),(0,0),(0,0)),'constant')
	elif idx == 1:
		ypad = 0
		xpad = num
		# print("Padding x = ", xpad)
		padded_image=np.pad(img,((0,0),(xpad,0),(0,0)),'constant')
	else: 
		raise Exception('Either width or height is smaller than no. of channel') 
	img_reqd = cv2.resize(padded_image,(reqd_size,reqd_size))		
	resized_image = img_reqd / 255.
	resized_image = resized_image[:,:,::-1]
	return ((ypad, xpad), padded_image, resized_image) #,ratio,num,idx
