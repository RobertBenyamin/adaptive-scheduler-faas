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

    Run the following command on your master node to install KNative CLI:
    ```bash
    wget https://github.com/knative/client/releases/download/knative-v1.9.1/kn-linux-amd64
    sudo mv kn-linux-amd64 /usr/local/bin/kn
    sudo chmod +x /usr/local/bin/kn
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

4.  **Build Application Image on Slave Nodes**
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

2.  **Install Knative infrastructure:**
    Before running the script, open `deploy_only.sh` and update it with the private IP address of your master node.
    ```bash
    ./deploy_only.sh
    ```

3.  **Deploy Applications**
    ```bash
    ./deploy_app.sh
    ```

4.  **Install Neccessary Python Package**
    ```bash
    sudo apt install python3-numpy python3-requests -y
    ```

5.  **Start Evaluation**
    ```bash
    python3 knative.py
    ```

    The result will be stored on `run-all-out.txt`

-----

### Additional Notes

For more technical details and usage guidelines, please refer to the inline code comments and accompanying scripts within this repository.