sudo systemctl restart k3s

# Install the required custom resources by running the command:
sudo kubectl apply -f https://github.com/knative/serving/releases/download/knative-v1.9.1/serving-crds.yaml

# Install the core components of Knative Serving by running the command:
sudo kubectl apply -f https://github.com/knative/serving/releases/download/knative-v1.9.1/serving-core.yaml

# Wait for the Knative Serving webhooks to be ready
echo "Waiting for Knative webhooks to become ready..."
sudo kubectl wait --for=condition=available --timeout=300s deployment/webhook -n knative-serving
sudo kubectl wait --for=condition=available --timeout=300s deployment/domainmapping-webhook -n knative-serving
echo "Knative webhooks are ready."

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
  -p '{"data":{"172.31.91.251.sslip.io":""}}' # Private IP Master Node

# Increase the default request timeout to 1 hours (3600s)
sudo kubectl patch configmap/config-kourier \
  -n knative-serving \
  --type merge \
  -p '{"data":{"stream-idle-timeout":"3600s"}}'

# Configure all Knative Serving timeouts
sudo kubectl patch configmap/config-defaults \
  -n knative-serving \
  --type merge \
  -p '{"data":{"max-revision-timeout-seconds":"3600","revision-timeout-seconds":"3600","revision-response-start-timeout-seconds":"3600","revision-idle-timeout-seconds":"0"}}'

echo "Restarting Kourier components to apply timeout settings..."
sudo kubectl rollout restart deployment/net-kourier-controller -n knative-serving
sudo kubectl rollout restart deployment/3scale-kourier-gateway -n kourier-system

# Wait for restarts to complete
sudo kubectl rollout status deployment/net-kourier-controller -n knative-serving --timeout=120s
sudo kubectl rollout status deployment/3scale-kourier-gateway -n kourier-system --timeout=120s

echo "Knative Serving with Kourier installation complete with 1-hour timeout configuration."

# Verify timeout configurations
echo ""
echo "=== Verifying timeout configurations ==="
echo "config-kourier:"
sudo kubectl get cm config-kourier -n knative-serving -o jsonpath='{.data.stream-idle-timeout}'
echo ""
echo "config-defaults:"
sudo kubectl get cm config-defaults -n knative-serving -o jsonpath='{.data}'
echo ""