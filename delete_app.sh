cd cnn_serving
sudo kubectl delete -f cnn_serving.yaml

cd ../img_res
sudo kubectl delete -f img_res.yaml

cd ../img_rot
sudo kubectl delete -f img_rot.yaml

cd ../ml_train
sudo kubectl delete -f ml_train.yaml

cd ../vid_proc
sudo kubectl delete -f vid_proc.yaml

cd ../web_serve
sudo kubectl delete -f web_serve.yaml