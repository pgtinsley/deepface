#!pip install deepface
from deepface.basemodels import VGGFace, OpenFace, Facenet, FbDeepFace
from deepface.commons import functions

import numpy as np

import glob
import argparse

parser = argparse.ArgumentParser(description='DeepFace')
parser.add_argument('--model', '-m', default='fbdeepface', type=str)
parser.add_argument('--dir', '-d', required=True, type=str)
parser.add_argument('--embeddings', '-e', required=True, type=str)
args = parser.parse_args()

#----------------------------------------------
#build face recognition model

if args.model=='vgg':
    model = VGGFace.loadModel()
elif args.model=='facenet':
    model = Facenet.loadModel()
elif args.model=='openface':
    model = OpenFace.loadModel()
elif args.model=='fbdeepface':
    model = FbDeepFace.loadModel()
else:
    print('Invalid model choice. Exiting...')
    exit()
    
input_shape = model.layers[0].input_shape[0][1:3]

#----------------------------------------------
#load images and find embeddings

dir = [args.dir if args.dir[-1]=='/' else args.dir+'/'][0]
img_fnames = glob.glob(dir+'*.png') + glob.glob(dir+'*.jpg') + glob.glob(dir+'*.jpeg')

embeddings = {}
for fname in img_fnames:
    feat = model.predict(functions.preprocess_face(fname, target_size=input_shape, enforce_detection=False))[0]
    embeddings[fname] = feat

with open(args.embeddings, 'wb') as f:
    pickle.dump(embeddings, f)

print('Results saved to {}'.format(args.embeddings))
exit()


#----------------------------------------------