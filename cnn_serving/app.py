from mxnet import gluon
import os
import mxnet as mx
from PIL import Image
from storage_helper import download_file, upload_file
import dnld_blob


current_path = "/app/pythonAction"

net = gluon.model_zoo.vision.resnet50_v1(pretrained=True, root = '/tmp/')
net.hybridize(static_alloc=True, static_shape=True)
lblPath = gluon.utils.download('https://github.com/shicai/MobileNet-Caffe/blob/master/synset.txt',path='/tmp/')
with open(lblPath, 'r') as f:
    labels = [l.rstrip() for l in f]

def lambda_handler():
    blobName = "img10.jpg"    
    # dnld_blob.download_blob_new(blobName)
    # full_blob_name = blobName.split(".")
    # proc_blob_name = full_blob_name[0] + "_" + str(os.getpid()) + "." + full_blob_name[1]
    download_file(blobName, f"{current_path}/{blobName}")
    image = Image.open(f"{current_path}/{blobName}")
    image.save(f'{current_path}/tempImage_'+str(os.getpid())+'.jpeg')

    # format image as (batch, RGB, width, height)
    img = mx.image.imread(f'{current_path}/tempImage_'+str(os.getpid())+'.jpeg')
    img = mx.image.imresize(img, 224, 224) # resize
    img = mx.image.color_normalize(img.astype(dtype='float32')/255,
                                mean=mx.nd.array([0.485, 0.456, 0.406]),
                                std=mx.nd.array([0.229, 0.224, 0.225])) # normal     ize
    img = img.transpose((2, 0, 1)) # channel first
    img = img.expand_dims(axis=0) # batchify

    prob = net(img).softmax() # predict and normalize output
    idx = prob.topk(k=5)[0] # get top 5 result
    inference = ''
    for i in idx:
        i = int(i.asscalar())
        # print('With prob = %.5f, it contains %s' % (prob[0,i].asscalar(), labels[i]))
        # inference = inference + 'With prob = %.5f, it contains %s' % (prob[0,i].asscalar(), labels[i]) + '. '
        inference = inference + 'With prob = %.5f, it contains ' % (prob[0,i].asscalar()) + '. '
    return {"result = ":inference}
