from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from scipy import misc
import time
import sys
import os
import numpy as np
import mxnet as mx
import random
import cv2
import sklearn
from sklearn.decomposition import PCA
from time import sleep
from easydict import EasyDict as edict
from impl.mtcnn_detector import MtcnnDetector

sys.path.append('facematch/config')
from config import Config
from skimage import transform as trans
tform = trans.SimilarityTransform()

def preprocess(img, bbox=None, landmark=None, **kwargs):
  M = None
  image_size = []
  str_image_size = kwargs.get('image_size', '')
  if len(str_image_size)>0:
    image_size = [int(x) for x in str_image_size.split(',')]
    if len(image_size)==1:
      image_size = [image_size[0], image_size[0]]
    assert len(image_size)==2
    assert image_size[0]==112
    assert image_size[0]==112 or image_size[1]==96
  if landmark is not None:
    assert len(image_size)==2
    src = np.array([
      [30.2946, 51.6963],
      [65.5318, 51.5014],
      [48.0252, 71.7366],
      [33.5493, 92.3655],
      [62.7299, 92.2041] ], dtype=np.float32 )
    if image_size[1]==112:
      src[:,0] += 8.0
    dst = landmark.astype(np.float32)

    #tform = trans.SimilarityTransform()
    tform.estimate(dst, src)
    M = tform.params[0:2,:]
    #M = cv2.estimateRigidTransform( dst.reshape(1,5,2), src.reshape(1,5,2), False)

  if M is None:
    if bbox is None: #use center crop
      det = np.zeros(4, dtype=np.int32)
      det[0] = int(img.shape[1]*0.0625)
      det[1] = int(img.shape[0]*0.0625)
      det[2] = img.shape[1] - det[0]
      det[3] = img.shape[0] - det[1]
    else:
      det = bbox
    margin = kwargs.get('margin', 44)
    bb = np.zeros(4, dtype=np.int32)
    bb[0] = np.maximum(det[0]-margin/2, 0)
    bb[1] = np.maximum(det[1]-margin/2, 0)
    bb[2] = np.minimum(det[2]+margin/2, img.shape[1])
    bb[3] = np.minimum(det[3]+margin/2, img.shape[0])
    ret = img[bb[1]:bb[3],bb[0]:bb[2],:]
    if len(image_size)>0:
      ret = cv2.resize(ret, (image_size[1], image_size[0]))
    return ret 
  else: #do align using landmark
    assert len(image_size)==2

    warped = cv2.warpAffine(img,M,(image_size[1],image_size[0]), borderValue = 0.0)

    #tform3 = trans.ProjectiveTransform()
    #tform3.estimate(src, dst)
    #warped = trans.warp(img, tform3, output_shape=_shape)
    return warped


def do_flip(data):
  for idx in range(data.shape[0]):
    data[idx,:,:] = np.fliplr(data[idx,:,:])

class FaceModel:
    def __init__(self):
        model = edict()
        _vec = '112,112'.split(',')
        assert len(_vec)==2
        image_size = (int(_vec[0]), int(_vec[1]))
        self.image_size = image_size
        _vec = './models/model-r100-ii/model,0'.split(',')
        assert len(_vec)==2
        prefix = _vec[0]
        epoch = int(_vec[1])
        print('loading',prefix, epoch)
        ctx = Config.ctx #mx.cpu() #gpu(args.gpu)
        sym, arg_params, aux_params = mx.model.load_checkpoint(prefix, epoch)
        all_layers = sym.get_internals()
        sym = all_layers['fc1_output']
        model = mx.mod.Module(symbol=sym, context=ctx, label_names = None)
        # model.bind(data_shapes=[('data', (args.batch_size, 3, image_size[0], image_size[1]))], label_shapes=[('softmax_label', (args.batch_size,))])
        model.bind(data_shapes=[('data', (1, 3, image_size[0], image_size[1]))])
        model.set_params(arg_params, aux_params)
        self.model = model

        mtcnn_path = './models/mtcnn-model/' #os.path.join(os.path.dirname(__file__), 'mtcnn-model')
        # self.det_minsize = 50
        # self.det_threshold = [0.4,0.6,0.6]
        # self.det_factor = 0.9
        self.detector = MtcnnDetector(model_folder=mtcnn_path, ctx=ctx, num_worker=Config.NUM_WORKER, accurate_landmark = Config.ACC_LM, threshold= Config.MTCNN_TH)
        # self.detector = detector


    def get_input(self, face_img):
        ret = self.detector.detect_face(face_img, det_type = 0) #self
        if ret is None:
          return None
        bbox, points = ret
        if bbox.shape[0]==0:
          return None
        bbox = bbox[0,0:4]
        points = points[0,:].reshape((2,5)).T
        nimg = preprocess(face_img, bbox, points, image_size='112,112')
        nimg = cv2.cvtColor(nimg, cv2.COLOR_BGR2RGB)
        aligned = np.transpose(nimg, (2,0,1))
        return aligned
    
    def get_feature(self, aligned):
        input_blob = np.expand_dims(aligned, axis=0)
        data = mx.nd.array(input_blob)
        db = mx.io.DataBatch(data=(data,))
        self.model.forward(db, is_train=False) #self
        embedding = self.model.get_outputs()[0].asnumpy()
        embedding = sklearn.preprocessing.normalize(embedding).flatten()
        return embedding

def predict_match(emb1, emb2):
    try:
        dist = np.sum(np.square(emb1-emb2))
    except:
        dist = -999
    return dist