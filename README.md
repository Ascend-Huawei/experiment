# Head Pose Estimation
This model recognizes the head pose of a person in terms of 3 angles: *yaw*, *pitch* and *roll* in an image. 
The face detection model is used to locate the face in the image prior to inferring the head pose from the detected face.

## Model Description

#### Models:
Here we are using offline models for 1. face detection 2. head pose estimation to inference on the board. 



1. Face Detection

Download the weight and network files to your project directory 'head_pose':

- Weights: https://obs-model-ascend.obs.cn-east-2.myhuaweicloud.com/face_detection/face_detection.caffemodel
- Network: https://github.com/Ascend-Huawei/models/blob/master/computer_vision/object_detect/face_detection/face_detection.prototxt

Execute the following command from your project directory 'head_pose' to convert the pre-trained model for face detection to offline model (.om) format:

**atc --output_type=FP32 --input_shape="data:1,3,300,300" --weight="face_detection.caffemodel" --input_format=NCHW --output="face_detection" --soc_version=Ascend310 
--insert_op_conf=insert_op.cfg --framework=0 --save_original_model=false 
--model="face_detection.prototxt"**



2. Head Pose Estimation

Download the weights and network files to your project directory 'head_pose':

- Weights:
https://obs-model-ascend.obs.cn-east-2.myhuaweicloud.com/head_pose_estimation/head_pose_estimation.caffemodel
- Network: https://github.com/Ascend-Huawei/models/blob/master/computer_vision/object_detect/head_pose_estimation/head_pose_estimation.prototxt


Execute the following command from your project directory 'head_pose' to convert the pre-trained model for head pose estimation to offline model (.om) format:

**atc --output_type=FP32 --input_shape="data:1,3,224,224" --weight="head_pose_estimation.caffemodel" --input_format=NCHW --output="head_pose_estimation"
--soc_version=Ascend310 --framework=0 --save_original_model=false --model="head_pose_estimation.prototxt"**


#### Inputs
The input for face detection model are as follows:
- **Input Shape**: [1,300,300, 3]
- **Input Format** : NCHW
- **Input Type**: FLOAT32

The input for the head pose estimation model are as follows:
- **Input Shape**: [1,3, 224, 224]
- **Input Format** : NCHW
- **Input Type**: FLOAT32

#### Outputs
The outputs for the face detection model are as follows:
- The pre-trained model will detect 2 types: face and others.
- Output is a list of shape: (1, 8)
  - **0 position**: not used
  - **1 position**: label
  - **2 position**: confidence score
  - **3 position**: top left x coordinate
  - **4 position**: top left y coordinate
  - **5 position**: bottom right x coordinate
  - **6 position**: bottom right y coordinate
  - **7 position**: not used

The outputs for the head pose estimation model are as follows:
- List of numpy arrays: 
  - **List shapes**: (1, 136, 1, 1), (1, 3, 1, 1)
The first list is a set of 136 facial keypoints. The second list in the output containing the 3 values of yaw, pitch, roll angles predicted by the model, which are used to determine head pose based on some preset rules.

Output printed to terminal (sample):
```
Head angles: [array([[9.411621]], dtype=float32), array([[7.91626]], dtype=float32), array([[-1.0116577]], dtype=float32)]
Pose: Head Good posture
```
Result image with 64 keypoints plotted on detected face saved in 'out' folder.

  
## Code:

  - All code files needed to run the experiment are included in folder 'head-pose'. The script 'head_pose_estimation.py' contains all the preprocessing, model inference and post_processing methods. 
  
  **Please Note:** You would need to complete all the code as per instructions, before running the application.
  
  - Preprocessing: 
    - **Resize**: (224, 224)
    - **Image Type**: FLOAT32
    - **Input Format** : NCHW
    - Change order from **[300, 300, 3]**(HWC) to **[3, 300, 300]**(CHW)
  
  - The om model file (.om) must be downloaded to the project folder 'head_pose'
 
  - Postprocessing:
    - Infer head pose from yaw, pitch and roll angles, using fixed range thresholds.

    
  - To run code, simply using commands below in the terminal:
  
    ``` 
    cd experiments/head_pose
    python3 head_pose_estimation.py 
    ``` 












  













