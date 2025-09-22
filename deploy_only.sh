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
  -p '{"data":{"52.72.211.102.sslip.io":""}}'