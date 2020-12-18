import numpy as np
import cv2
import sys
sys.path.append('../')
#ACL model load and execute implementation
from acl_model import Model
#ACL init and resource management implementation
from acl_resource import AclResource 

# Path for head pose model
model_name_head_pose = 'head_pose_estimation'
# Path for face detection model
model_name_face_det = 'face_detection'
# Test image
img_file = 'face1.jpeg'

def Main():

	# Step 1: initialize ACL and ACL runtime 
	# 1.1: one line of code to create an object of the 'AclResource' class
	### Your code here, one line ###

	# 1.2: one line of code, call the 'init' function of the AclResource object, to initilize ACL and ACL runtime 
	### Your code here, one line ###

	# Step 2: Load models for face detection and head pose estimation
	# 2.1 load offline model for face detection
	# Path for face detection model
	MODEL_PATH = model_name_face_det + ".om" 
	# one line of code to create an object of the 'Model' class with two parameters: the created 'AclResource' object and the 'MODEL_PATH'
	### Your code here, one line ###


	# 2.2 load offline model for head pose estimation
	# Path for head pose estimation model
	MODEL_PATH = model_name_head_pose + ".om" 
	# one line of code to create an object of the 'Model' class with two parameters: the created 'AclResource' object and the 'MODEL_PATH'
	### Your code here, one line ###

	# Step 3: Face Detection Inference
	# load testing image file 
	image = cv2.imread(img_file)
	# preprocessing the image file for face detection
	input_image = PreProcessing_face(image)

	# one line of code, use the 'Model' object for face detection, and call its 'execute' function with parameter '[input_image]'
	### Your code here, one line ###

	# Postprocessing to get the bounding box
	try:
		xmin, ymin, xmax, ymax = PostProcessing_face(image, resultList_face)
		bbox_list = [xmin, ymin, xmax, ymax]
	except:
		print('\n***Error***: Face detection inference result is not ready for post processing\n')
		return
		
	# Step 4: Head Pose Estimation
	# Preprocessing for head pose estimation
	input_image = PreProcessing_head(image, bbox_list)
	# head pose estimation model inference
	# one line of code, use the 'Model' object for head pose detection, and call its 'execute' function with parameter '[input_image]'
	### Your code here, one line ###

	#post processing to obtain coordinates for lines drawing
	try:
		facepointList, head_status_string = PostProcessing_head(resultList_head, bbox_list, image)
	except:
		print('\n***Error***: Head Pose estimation inference result is not ready for post processing\n')
		return
	print('Head angles:', resultList_head[2])
	print('Pose:', head_status_string)

# face detection model preprocessing
def PreProcessing_face(image):
	image = cv2.resize(image, (300,300))
	image = image.astype('float32')
	image = np.transpose(image, (2, 0, 1)).copy()
	return image

# face detection model post processing
def PostProcessing_face(image, resultList, threshold=0.9):
	detections = resultList[1]
	bbox_num = 0
	bbox_list = []
	for i in range(detections.shape[1]):
		det_conf = detections[0,i,2]
		det_xmin = detections[0,i,3]
		det_ymin = detections[0,i,4]
		det_xmax = detections[0,i,5]
		det_ymax = detections[0,i,6]
		bbox_width = det_xmax - det_xmin
		bbox_height = det_ymax - det_ymin
		if threshold <= det_conf and 1>=det_conf and bbox_width>0 and bbox_height > 0:
			bbox_num += 1
			xmin = int(round(det_xmin * image.shape[1]))
			ymin = int(round(det_ymin * image.shape[0]))
			xmax = int(round(det_xmax * image.shape[1]))
			ymax = int(round(det_ymax * image.shape[0]))
			print('BBOX:', xmin, ymin, xmax, ymax)
			cv2.rectangle(image,(xmin,ymin),(xmax,ymax),(0,255,0),1)
		else:
			continue
	print("detected bbox num:", bbox_num)
	return [xmin, ymin, xmax, ymax]


# head pose estimation preprocessing
def PreProcessing_head(image, boxList):
	# convert to float type
	image = np.asarray(image, dtype=np.float32)
	# crop out detected face
	image = image[int(boxList[1]):int(boxList[3]),int(boxList[0]):int(boxList[2])]
	# resize to required input dimensions
	image = cv2.resize(image, (224, 224))
	# switch from NHWC to NCHW format
	image = np.transpose(image, (2, 0, 1)).copy()
	return image
	

'''Step 5: Complete the following function to find the viewing direction 
from predicted head pose angles (pitch, yaw, roll)

HINT: The first, second and third element of the input list 'resultList' 
correspond to pitch, yaw and roll respectively.'''
def head_status_get(resultList):
	# initialize
	yaw = 'None'
	pitch = 'None'
	roll = 'None'
	fg_pitch = True
	fg_yaw = True
	fg_roll = True
	
	
	# Assign 'up' or 'down' values to 'head_pose' from pitch angle values. 
	if resultList[2][0] < -20:
		pitch = 'Up'
	elif resultList[2][0] > 20:
		pitch = 'Down'
	else:
		fg_pitch = False

	# Assign 'left' or 'right' values to 'head_pose' from yaw angle values. 
	### Your code here ###
	
	# Assign 'swing left' or 'swing right' values to 'head_pose' from roll angle values. 
	### Your code here ###
	
	if fg_pitch is False and fg_yaw is False and fg_roll is False:
		head_pose = 'Viewing direction: Straight ahead'
	else:
		head_pose = 'Viewing direction (pitch, yaw, roll): {}, {}, {}'.format(pitch, yaw, roll)
	return head_pose

def PostProcessing_head(resultList, boxList, image):
	resultList.append([resultList[1][0][0] * 50, resultList[1][0][1] * 50, resultList[1][0][2] * 50])
	HeadPosePoint = []
	facepointList = []
	box_width = boxList[2] - boxList[0] 
	box_height = boxList[3] - boxList[1]
	box_width = box_width
	box_height = box_height
	print('box width:', box_width)
	print('box height:', box_height)
	for j in range(136):
		if j % 2 == 0:
			HeadPosePoint.append((1+resultList[0][0][j])/2  * box_width + boxList[0])
		else:
			HeadPosePoint.append((1+resultList[0][0][j])/2  * box_height + boxList[1])
		facepointList.append(HeadPosePoint)
	for j in range(136):
		if j % 2 == 0:
			canvas = cv2.circle(image, (int(facepointList[0][j]), int(facepointList[0][j+1])), 
								radius=5, color=(255, 0, 0), thickness=2)
	head_status_string = head_status_get(resultList)
	cv2.putText(canvas,head_status_string,(10,10),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,0,0),1)
	cv2.imwrite('out/result_out.jpg', canvas)
	return facepointList, head_status_string


if __name__ == '__main__':
	Main()
			



