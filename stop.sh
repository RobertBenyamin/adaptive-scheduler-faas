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

# Force delete any stuck terminating pods
echo "Force deleting any terminating pods..."
kubectl delete pods --all --force --grace-period=0
echo "Terminating pods force deleted."

# Delete the namespaces
echo "Deleting Knative namespaces..."
kubectl delete namespace knative-serving --ignore-not-found=true
kubectl delete namespace kourier-system --ignore-not-found=true
echo "Knative namespaces deletion initiated."

# Delete the CRDs
echo "Deleting Knative CRDs..."
kubectl delete -f https://github.com/knative/serving/releases/download/knative-v1.9.1/serving-crds.yaml --ignore-not-found=true
echo "Knative CRDs deletion initiated."

# Force delete any stuck terminating pods again
echo "Force deleting any remaining terminating pods..."
kubectl delete pods --all --force --grace-period=0

echo "Cleanup initiated."