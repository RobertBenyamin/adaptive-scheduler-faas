
cd cnn_serving
sudo docker build -t server-cnn-serving:1.1.0 .
sudo docker tag server-cnn-serving:1.1.0 kind.local/server-cnn-serving:1.1.0

cd ../img_res
sudo docker build -t server-img-res:1.1.0 .
sudo docker tag server-img-res:1.1.0 kind.local/server-img-res:1.1.0

cd ../img_rot
sudo docker build -t server-img-rot:1.1.0 .
sudo docker tag server-img-rot:1.1.0 kind.local/server-img-rot:1.1.0

cd ../ml_train
sudo docker build -t server-ml-train:1.1.0 .
sudo docker tag server-ml-train:1.1.0 kind.local/server-ml-train:1.1.0

cd ../vid_proc
sudo docker build -t server-vid-proc:1.1.0 .
sudo docker tag server-vid-proc:1.1.0 kind.local/server-vid-proc:1.1.0

cd ../web_serve
sudo docker build -t server-web-serve:1.1.0 .
sudo docker tag server-web-serve:1.1.0 kind.local/server-web-serve:1.1.0

# 1.1 for default mxfaas ; 1.2 for mxfaas + srtf ; 1.2 for mxfaas + srtf + aging prevention