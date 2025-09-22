cd cnn_serving
kubectl delete -f cnn_serving.yaml

cd ../img_res
kubectl delete -f img_res.yaml

cd ../img_rot
kubectl delete -f img_rot.yaml

cd ../ml_train
kubectl delete -f ml_train.yaml

cd ../vid_proc
kubectl delete -f vid_proc.yaml

cd ../web_serve
kubectl delete -f web_serve.yaml

cd ..

# Install the required custom resources by running the command:
kubectl delete -f https://github.com/knative/serving/releases/download/knative-v1.9.1/serving-crds.yaml

# Install the core components of Knative Serving by running the command:
kubectl delete -f https://github.com/knative/serving/releases/download/knative-v1.9.1/serving-core.yaml

# Install the Knative Kourier controller by running the command: 
kubectl delete -f https://github.com/knative/net-kourier/releases/download/knative-v1.9.1/kourier.yaml

kubectl delete -f https://github.com/knative/serving/releases/download/knative-v1.9.1/serving-default-domain.yaml

sleep 3
