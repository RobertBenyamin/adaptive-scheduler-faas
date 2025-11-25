from mxnet import gluon
import os
import mxnet as mx
from storage_helper import download_file

TMP_DIR = "/tmp/"

net = gluon.model_zoo.vision.resnet50_v1(pretrained=True, root = TMP_DIR)
net.hybridize(static_alloc=True, static_shape=True)
lblPath = gluon.utils.download('https://raw.githubusercontent.com/shicai/MobileNet-Caffe/refs/heads/master/synset.txt',path=TMP_DIR)
with open(lblPath, 'r') as f:
    labels = [l.rstrip() for l in f]

def lambda_handler(event):
    blobName = event.get("input_file", "img10.jpg")
    
    pid = str(os.getpid())
    local_file_path = os.path.join(TMP_DIR, f"{pid}_{blobName}")

    download_file(blobName, local_file_path)

    # format image as (batch, RGB, width, height)
    img = mx.image.imread(local_file_path)
    img = mx.image.imresize(img, 224, 224) # resize
    img = mx.image.color_normalize(img.astype(dtype='float32')/255,
                                mean=mx.nd.array([0.485, 0.456, 0.406]),
                                std=mx.nd.array([0.229, 0.224, 0.225])) # normalize
    img = img.transpose((2, 0, 1)) # channel first
    img = img.expand_dims(axis=0) # batchify

    prob = net(img).softmax() # predict and normalize output
    idx = prob.topk(k=5)[0] # get top 5 result
    inference = ''
    for i in idx:
        i = int(i.asscalar())
        # print('With prob = %.5f, it contains %s' % (prob[0,i].asscalar(), labels[i]))
        inference = inference + 'With prob = %.5f, it contains %s' % (prob[0,i].asscalar(), labels[i]) + '. '
        # inference = inference + 'With prob = %.5f, it contains ' % (prob[0,i].asscalar()) + '. '
    return {"result = ":inference}
