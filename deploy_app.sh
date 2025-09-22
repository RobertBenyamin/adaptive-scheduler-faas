
cd cnn_serving
sudo kubectl apply -f cnn_serving.yaml

cd ../ml_train
sudo kubectl apply -f ml_train.yaml

cd ../web_serve
sudo kubectl apply -f web_serve.yaml

sleep 15

cd ../img_res
sudo kubectl apply -f img_res.yaml

cd ../img_rot
sudo kubectl apply -f img_rot.yaml

cd ../vid_proc
sudo kubectl apply -f vid_proc.yaml
