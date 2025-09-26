# Default to 'runner.py' (v1) if $1 is not provided
RUNNER_FILE=${1:-"runner.py"}
# Default to '1.1.0' if $2 is not provided
IMAGE_TAG=${2:-"1.1.0"}

cd cnn_serving
sudo docker build --build-arg RUNNER_FILE=${RUNNER_FILE} -t server-cnn-serving:${IMAGE_TAG} .
sudo docker tag server-cnn-serving:${IMAGE_TAG} kind.local/server-cnn-serving:${IMAGE_TAG}

cd ../img_res
sudo docker build --build-arg RUNNER_FILE=${RUNNER_FILE} -t server-img-res:${IMAGE_TAG} .
sudo docker tag server-img-res:${IMAGE_TAG} kind.local/server-img-res:${IMAGE_TAG}

cd ../img_rot
sudo docker build --build-arg RUNNER_FILE=${RUNNER_FILE} -t server-img-rot:${IMAGE_TAG} .
sudo docker tag server-img-rot:${IMAGE_TAG} kind.local/server-img-rot:${IMAGE_TAG}

cd ../ml_train
sudo docker build --build-arg RUNNER_FILE=${RUNNER_FILE} -t server-ml-train:${IMAGE_TAG} .
sudo docker tag server-ml-train:${IMAGE_TAG} kind.local/server-ml-train:${IMAGE_TAG}

cd ../vid_proc
sudo docker build --build-arg RUNNER_FILE=${RUNNER_FILE} -t server-vid-proc:${IMAGE_TAG} .
sudo docker tag server-vid-proc:${IMAGE_TAG} kind.local/server-vid-proc:${IMAGE_TAG}

cd ../web_serve
sudo docker build --build-arg RUNNER_FILE=${RUNNER_FILE} -t server-web-serve:${IMAGE_TAG} .
sudo docker tag server-web-serve:${IMAGE_TAG} kind.local/server-web-serve:${IMAGE_TAG}

# 1.1.0 for default mxfaas (runner.py)
# 1.2.0 for mxfaas + srtf (runners_v2.py)
# 1.3.0 for mxfaas + srtf + aging prevention (runners_v3.py)
# 1.4.0 for ALPS (runners_v4.py)
# 1.5.0 for mxfaas + srtf + dynamic aging (runners_v5.py)