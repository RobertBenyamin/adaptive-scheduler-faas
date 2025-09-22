sudo systemctl restart k3s

# Install the required custom resources by running the command:
sudo kubectl apply -f https://github.com/knative/serving/releases/download/knative-v1.9.1/serving-crds.yaml

# Install the core components of Knative Serving by running the command:
sudo kubectl apply -f https://github.com/knative/serving/releases/download/knative-v1.9.1/serving-core.yaml

# Install the Knative Kourier controller by running the command: 
sudo kubectl apply -f https://github.com/knative/net-kourier/releases/download/knative-v1.9.1/kourier.yaml

# Configure Knative Serving to use Kourier by default by running the command: 
sudo kubectl patch configmap/config-network \
  --namespace knative-serving \
  --type merge \
  --patch '{"data":{"ingress-class":"kourier.ingress.networking.knative.dev"}}'
 
# Configure DNS
sudo kubectl apply -f https://github.com/knative/serving/releases/download/knative-v1.9.1/serving-default-domain.yaml

sudo kubectl patch configmap/config-domain \
  -n knative-serving \
  --type merge \
  -p '{"data":{"172.31.30.248.sslip.io":""}}'


sleep 30

cd cnn_serving
sudo kubectl apply -f cnn_serving.yaml

cd ../ml_train
sudo kubectl apply -f ml_train.yaml

cd ../web_serve
sudo kubectl apply -f web_serve.yaml

# sleep 15

cd ../img_res
sudo kubectl apply -f img_res.yaml

cd ../img_rot
sudo kubectl apply -f img_rot.yaml

cd ../vid_proc
sudo kubectl apply -f vid_proc.yaml

cd ../

sleep 15

python3 knative.py