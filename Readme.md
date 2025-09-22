This repository provides the source code and related materials for a scheduler implementation targeting Function-as-a-Service (FaaS) environments. The scheduler combines dynamic aging with the Shortest Remaining Time First (SRTF) algorithm to improve response time and resource efficiency in serverless workloads. The code includes all essential scripts, configurations, and illustrative examples to reproduce the described scheduling behavior.

The implementation is intended for research and educational purposes, and may be extended in future publications. For installation, usage guidelines, and further technical details, please refer to the inline code comments and accompanying scripts. Contributions, suggestions, or feedback are very welcome through the issue tracker or pull requests.



## Getting Started

To run this repository, you'll need a cluster with a minimum of two nodes: one **master** and one or more **slave** nodes. All nodes must have Docker installed.

### Prerequisites

  * **Docker**: Ensure Docker is installed on all nodes. For EC2 instances, you can follow the official guide [here](https://docs.aws.amazon.com/serverless-application-model/latest/developerguide/install-docker.html).
  * **Networking**: Ensure your slave nodes can communicate with the master node.

### Installation & Setup

1.  **Set up the Master Node:**
    Run the following command on your master node to install K3s with Docker as the container runtime:

    ```bash
    curl -sfL https://get.k3s.io | sh -s - --docker
    ```

2.  **Get the K3s Cluster Token:**
    Find your cluster token, which is required to add slave nodes. You can learn more about K3s tokens [here](https://docs.k3s.io/cli/token).

    ```bash
    cat /var/lib/rancher/k3s/server/token
    ```

3.  **Set up the Slave Node:**
    On each slave node, join the cluster by running this command. Be sure to replace `<master_ip_address>` and `<node_token>` with your specific values.

    ```bash
    curl -sfL https://get.k3s.io | K3S_URL=https://<master_ip_address>:6443 K3S_TOKEN=<node_token> sh -s - --docker
    ```

4.  **Install KNative at Slave Node:**
    Do following things to install KNative

    ```bash 
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
    -p '{"data":{"172.31.30.248.sslip.io":""}}' // ip master
    ```

5.  **Build Application Image on Slave Nodes**
    The application image needs to be built on each slave node. Execute the build.sh script to create the necessary image for the evaluation.

    ```bash
    ./build.sh
    ```

-----

## Usage

1.  **Prepare Remote Storage:**
    Place all files from the `storage-s3` folder into your remote storage (e.g., an S3 bucket). Make sure to configure your environment variables with the necessary credentials:

      * `AWS_ACCESS_KEY_ID`
      * `AWS_SECRET_ACCESS_KEY`
      * `AWS_DEFAULT_REGION`

2.  **Install Knative and Start Evaluation at Master:**
    Once your cluster and remote storage are ready, execute the `start.sh` script to install Knative and begin the evaluation.

    ```bash
    ./start.sh
    ```

-----

### Additional Notes

For more technical details and usage guidelines, please refer to the inline code comments and accompanying scripts within this repository.