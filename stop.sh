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

# Delete the namespaces which will also delete all the resources within them.
# The --wait flag ensures the command blocks until deletion is complete.
echo "Deleting Knative namespaces and waiting for completion..."
kubectl delete namespace knative-serving --wait=true --ignore-not-found=true
kubectl delete namespace kourier-system --wait=true --ignore-not-found=true
echo "Knative namespaces deleted."

# The CRDs should be deleted last.
echo "Deleting Knative CRDs..."
kubectl delete -f https://github.com/knative/serving/releases/download/knative-v1.9.1/serving-crds.yaml --wait=true --ignore-not-found=true
echo "Knative CRDs deleted."

echo "Cleanup complete."