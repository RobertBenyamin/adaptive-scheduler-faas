# Script to delete all Docker images built by the build script
# Assumes IMAGE_TAG variable is set (default to 1.1.0 if not set)

IMAGE_TAG=${IMAGE_TAG:-1.1.0}

# Stop all running containers
sudo docker stop $(docker ps -aq)

# Remove all containers
sudo docker rm $(docker ps -aq)

echo "Deleting Docker images with tag: ${IMAGE_TAG}"

# Delete server images
echo "Deleting server-cnn-serving:${IMAGE_TAG}"
sudo docker rmi server-cnn-serving:${IMAGE_TAG} 2>/dev/null || echo "Image not found: server-cnn-serving:${IMAGE_TAG}"

echo "Deleting server-img-res:${IMAGE_TAG}"
sudo docker rmi server-img-res:${IMAGE_TAG} 2>/dev/null || echo "Image not found: server-img-res:${IMAGE_TAG}"

echo "Deleting server-img-rot:${IMAGE_TAG}"
sudo docker rmi server-img-rot:${IMAGE_TAG} 2>/dev/null || echo "Image not found: server-img-rot:${IMAGE_TAG}"

echo "Deleting server-ml-train:${IMAGE_TAG}"
sudo docker rmi server-ml-train:${IMAGE_TAG} 2>/dev/null || echo "Image not found: server-ml-train:${IMAGE_TAG}"

echo "Deleting server-vid-proc:${IMAGE_TAG}"
sudo docker rmi server-vid-proc:${IMAGE_TAG} 2>/dev/null || echo "Image not found: server-vid-proc:${IMAGE_TAG}"

echo "Deleting server-web-serve:${IMAGE_TAG}"
sudo docker rmi server-web-serve:${IMAGE_TAG} 2>/dev/null || echo "Image not found: server-web-serve:${IMAGE_TAG}"

# Delete kind.local tagged images
echo "Deleting kind.local/server-cnn-serving:${IMAGE_TAG}"
sudo docker rmi kind.local/server-cnn-serving:${IMAGE_TAG} 2>/dev/null || echo "Image not found: kind.local/server-cnn-serving:${IMAGE_TAG}"

echo "Deleting kind.local/server-img-res:${IMAGE_TAG}"
sudo docker rmi kind.local/server-img-res:${IMAGE_TAG} 2>/dev/null || echo "Image not found: kind.local/server-img-res:${IMAGE_TAG}"

echo "Deleting kind.local/server-img-rot:${IMAGE_TAG}"
sudo docker rmi kind.local/server-img-rot:${IMAGE_TAG} 2>/dev/null || echo "Image not found: kind.local/server-img-rot:${IMAGE_TAG}"

echo "Deleting kind.local/server-ml-train:${IMAGE_TAG}"
sudo docker rmi kind.local/server-ml-train:${IMAGE_TAG} 2>/dev/null || echo "Image not found: kind.local/server-ml-train:${IMAGE_TAG}"

echo "Deleting kind.local/server-vid-proc:${IMAGE_TAG}"
sudo docker rmi kind.local/server-vid-proc:${IMAGE_TAG} 2>/dev/null || echo "Image not found: kind.local/server-vid-proc:${IMAGE_TAG}"

echo "Deleting kind.local/server-web-serve:${IMAGE_TAG}"
sudo docker rmi kind.local/server-web-serve:${IMAGE_TAG} 2>/dev/null || echo "Image not found: kind.local/server-web-serve:${IMAGE_TAG}"

echo "Cleaning up dangling images..."
sudo docker image prune -f

echo "Image deletion completed!"

# Display remaining images
echo "Remaining Docker images:"
sudo docker images