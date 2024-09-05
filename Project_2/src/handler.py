__copyright__   = "Copyright 2024, VISA Lab"
__license__     = "MIT"

import os
import boto3
import cv2
from PIL import Image, ImageDraw, ImageFont
from facenet_pytorch import MTCNN, InceptionResnetV1
from shutil import rmtree
import numpy as np
import torch

temp_dir_path = '/tmp'
datapt_key = 'data.pt'
os.environ['TORCH_HOME'] = temp_dir_path + '/'
output_bucket_name = 'ID-output'
temp_bucket_name = 'ID-in-bucket'

AWS_ACCESS_KEY_ID = 'Your Access Key'
AWS_SECRET_ACCESS_KEY = 'Your Secret Access Key'
region_name = 'us-east-1'

mtcnn = MTCNN(image_size=240, margin=0, min_face_size=20) # initializing mtcnn for face detection
resnet = InceptionResnetV1(pretrained='vggface2').eval() # initializing resnet for face img to embeding conversion


s3 = boto3.client('s3',aws_access_key_id= AWS_ACCESS_KEY_ID, aws_secret_access_key=AWS_SECRET_ACCESS_KEY, region_name=region_name)



def face_recognition_function(key_path):
    # Face extraction
    img = cv2.imread(key_path, cv2.IMREAD_COLOR)
    boxes, _ = mtcnn.detect(img)

    # Face recognition
    key = os.path.splitext(os.path.basename(key_path))[0].split(".")[0]
    img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    face, prob = mtcnn(img, return_prob=True, save_path=None)
    saved_data = torch.load('/tmp/data.pt')  # loading data.pt file
    if face != None:
        emb = resnet(face.unsqueeze(0)).detach()  # detech is to make required gradient false
        embedding_list = saved_data[0]  # getting embedding data
        name_list = saved_data[1]  # getting list of names
        dist_list = []  # list of matched distances, minimum distance is used to identify the person
        for idx, emb_db in enumerate(embedding_list):
            dist = torch.dist(emb, emb_db).item()
            dist_list.append(dist)
        idx_min = dist_list.index(min(dist_list))

        # Save the result name in a file
        with open("/tmp/" + key + ".txt", 'w+') as f:
            f.write(name_list[idx_min])
        return name_list[idx_min]
    else:
        print(f"No face is detected")
    return

def recognize_face_and_pust_to_S3(image_path):
    file = face_recognition_function(image_path)
    key = os.path.splitext(os.path.basename(image_path))[0].split(".")[0]
    print(key, " key after face reco")
    txt_path = key+ ".txt"
    file_path = temp_dir_path + '/' + txt_path
    print('file_path ---> ', file_path)
    try:
        s3.upload_file(file_path, output_bucket_name, txt_path)
        print(f"File uploaded successfully to S3 : {txt_path}")
    except Exception as e:
        print(f"Error uploading file to S3: {e}")
	

def handler(event, context):
	print("event --> ",event)	
	bucket_name = event['bucket_name']
	image_file_name = event['image_file_name']
	print("bucket_name and image _file name ------> ",bucket_name, image_file_name)
	s3.download_file(temp_bucket_name, datapt_key, temp_dir_path + '/' + datapt_key)
	image_path = temp_dir_path + '/' + image_file_name
	print('image_path ---> ', image_path)
	s3.download_file(bucket_name, image_file_name, image_path)
	recognize_face_and_pust_to_S3(image_path)
	