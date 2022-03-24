---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.13.7
  kernelspec:
    display_name: Python 3
    name: python3
---

```python colab={"base_uri": "https://localhost:8080/"} id="Yg1eogMBatuZ" executionInfo={"status": "ok", "timestamp": 1628957389115, "user_tz": -330, "elapsed": 531, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="e93ffbfe-2115-4280-d7c3-702ac4dbf289"
!mkdir ./code/microservice_01
%cd ./code/microservice_01
```

<!-- #region id="rVfAm-mMZ3eK" -->
## Creating a trivial Node.js app
<!-- #endregion -->

<!-- #region id="BdRwBZh5aG2K" -->
It should be clear what this code does. It starts up an HTTP server on port 8080. The server responds with an HTTP response status code 200 OK and the text "You've hit <hostname>" to every request. The request handler also logs the client’s IP address to the standard output, which you’ll need later. The returned hostname is the server’s actual hostname, not the one the client sends in the HTTP request’s Host header.
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="dXZJNKZMZoLA" executionInfo={"status": "ok", "timestamp": 1628957398830, "user_tz": -330, "elapsed": 1037, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="45d8c35b-8a22-478a-e1db-9805a729d76e"
%%writefile app.js
const http = require('http');
const os = require('os');

console.log("Kubia server starting...");

var handler = function(request, response) {
  console.log("Received request from " + request.connection.remoteAddress);
  response.writeHead(200);
  response.end("You've hit " + os.hostname() + "\n");
};

var www = http.createServer(handler);
www.listen(8080);
```

<!-- #region id="ogwAmyERadyO" -->
## Creating a Dockerfile for the image
To package your app into an image, you first need to create a file called Dockerfile, which will contain a list of instructions that Docker will perform when building the image. The Dockerfile needs to be in the same directory as the app.js file and should contain the commands shown in the following listing.
<!-- #endregion -->

```python id="wGhByw5EgfDV" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1628961861052, "user_tz": -330, "elapsed": 1027, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="8ad74e08-7292-463d-a311-fa356e6aaebf"
%%writefile Dockerfile
FROM node:7
ADD app.js /app.js
ENTRYPOINT ["node", "app.js"]
```

<!-- #region id="ZgqPU7Kng8qM" -->
## Building the container image
Now that you have your Dockerfile and the app.js file, you have everything you need to build your image. To build it, run the following Docker command:
<!-- #endregion -->

```python id="6EP_eWiKg-vW" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1628967051675, "user_tz": -330, "elapsed": 752, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="291d46be-4ef6-4ac4-fb81-3e50090cb839"
%%writefile 01_build_image.sh
docker build -t kubia .
```

<!-- #region id="GZoV7t6JhJa6" -->
You’re telling Docker to build an image called kubia based on the contents of the current directory (note the dot at the end of the build command). Docker will look for the Dockerfile in the directory and build the image based on the instructions in the file.
<!-- #endregion -->

<!-- #region id="YgssmmCnhLde" -->
<!-- #endregion -->

<!-- #region id="VpySZGINhdE-" -->
The build process isn’t performed by the Docker client. Instead, the contents of the whole directory are uploaded to the Docker daemon and the image is built there. The client and daemon don’t need to be on the same machine at all. If you’re using Docker on a non-Linux OS, the client is on your host OS, but the daemon runs inside a VM. Because all the files in the build directory are uploaded to the daemon, if it contains many large files and the daemon isn’t running locally, the upload may take longer.
<!-- #endregion -->

<!-- #region id="f6mZA0tdiwuK" -->
> Tip: You can see the list of locally-stored docker images by running ```$ docker images``` command.
<!-- #endregion -->

<!-- #region id="QvaoG9HLhJ2m" -->
> Note: Container images are composed of layers that can be shared among different images.
<!-- #endregion -->

<!-- #region id="m46SeC4xh6QI" -->
<!-- #endregion -->

<!-- #region id="mvA1wOvsh6tJ" -->
## Running the container image
You can now run your image with the following command:
<!-- #endregion -->

```python id="NjDqr2YPiZw5" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1628967053857, "user_tz": -330, "elapsed": 5, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="11a284e0-8ace-48bf-be1e-f7e468acf8fe"
%%writefile 02_run_container.sh
docker run --name kubia-container -p 8080:8080 -d kubia
```

<!-- #region id="UxTZ1AFnibRQ" -->
This tells Docker to run a new container called kubia-container from the kubia image. The container will be detached from the console (-d flag), which means it will run in the background. Port 8080 on the local machine will be mapped to port 8080 inside the container (-p 8080:8080 option), so you can access the app through http://localhost:8080.
<!-- #endregion -->

<!-- #region id="P71BlIhxwoPX" -->
## Push the image to Dockerhub
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="ZihHsG2lwq_s" executionInfo={"status": "ok", "timestamp": 1628967054548, "user_tz": -330, "elapsed": 7, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="718553e3-7e67-405e-87d5-cc9dea4212a9"
%%writefile 03_push_image.sh
docker tag kubia:latest sparshai/kubia:latest
docker push sparshai/kubia
```

<!-- #region id="CWq4z9AojJjA" -->
## Docker commands
Here are some handy commands:
- ```$ curl localhost:8080``` for accessing your app.
- ```$ docker ps``` for listing all the running containers.
- ```$ docker inspect kubia-container``` for more detailed information
- ```$ docker exec -it kubia-container bash``` for starting docker shell
- ```$ ps aux``` for listing running processes inside docker
- ```$ docker stop kubia-container``` for stopping the container
- ```$ docker rm kubia-container``` for deleting the container
- ```$ docker push <your-id>/kubia``` for pushing image to DockerHub.
<!-- #endregion -->

<!-- #region id="7nh5lo6GjeYB" -->
## Running the image on a different machine
After the push to Docker Hub is complete, the image will be available to everyone. You can now run the image on any machine running Docker by executing the following command: ```$ docker run -p 8080:8080 -d luksa/kubia```.

It doesn’t get much simpler than that. And the best thing about this is that your application will have the exact same environment every time and everywhere it’s run. If it ran fine on your machine, it should run as well on every other Linux machine. No need to worry about whether the host machine has Node.js installed or not. In fact, even if it does, your app won’t use it, because it will use the one installed inside the image.
<!-- #endregion -->

<!-- #region id="v9iWbK_mmuGO" -->
## Kubernetes commands
Here are some handy commands:
- ```$ kubectl get nodes``` for listing cluster nodes
- ```$ kubectl describe node xxx-kubia-85f6-node-0rrx``` for retrieving additional details of an object
- ```$ alias k=kubectl``` to create alias for kubectl command
- ```$ kubectl get pods``` for listing pods
- ```$ kubectl get services``` for listing services
<!-- #endregion -->

<!-- #region id="FTNW47F-x692" -->
## Start the cluster
First step is to initialize the cluster in the first terminal:
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="2xUTSsFTx_0v" executionInfo={"status": "ok", "timestamp": 1628967056065, "user_tz": -330, "elapsed": 6, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="a6796d9c-600d-4abc-a388-738490dbbbf1"
%%writefile 04_start_cluster.sh
kubeadm init --apiserver-advertise-address $(hostname -i)
```

<!-- #region id="XeLQ-iyo4OxU" -->
That means you’re almost ready to go. Last you just have to initialize your cluster networking in the first terminal:
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="KRYv_m7E4LhN" executionInfo={"status": "ok", "timestamp": 1628967056778, "user_tz": -330, "elapsed": 5, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="791067fc-5181-4df3-f058-441a65bb4aea"
%%writefile 05_enable_networking.sh
kubectl apply -n kube-system -f "https://cloud.weave.works/k8s/net?k8s-version=$(kubectl version | base64 |tr -d '\n')"
```

<!-- #region id="KqUdTnbEzSVq" -->
To join other nodes as workers to the master node, run command like the following that you will receive from the master terminal after starting the cluster.

```
kubeadm join 192.168.0.33:6443 --token 20l5u8.pekij3hq5xzux7h9 \
    --discovery-token-ca-cert-hash sha256:572d36c75cd7456302c44886f4ae99514956c99cef3db7a50be7ed78080f5a77
```
<!-- #endregion -->

<!-- #region id="mJwk5aympgA1" -->
## Deploying your Node.js app
The simplest way to deploy your app is to use the kubectl run command, which will create all the necessary components without having to deal with JSON or YAML. This way, we don’t need to dive into the structure of each object yet. Try to run the image you created and pushed to Docker Hub earlier. Here’s how to run it in Kubernetes:
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="5TltXc9vpjlK" executionInfo={"status": "ok", "timestamp": 1628967059301, "user_tz": -330, "elapsed": 450, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="88e796d4-fd32-441b-f069-56b76060f4a1"
%%writefile 06_deploy_app.sh
kubectl create deployment kubia-app --image=sparshai/kubia --port=8080
```

<!-- #region id="pmsL4rGR0DJd" -->
> Tip: Check the status by running ```$ k get pods``` or ```$ k describe pods``` for more detailed info.
<!-- #endregion -->

<!-- #region id="hF9kQ2-erJUb" -->
## Behind the scenes
<!-- #endregion -->

<!-- #region id="KI0aRzS4rOmJ" -->
<!-- #endregion -->

<!-- #region id="h2gg-qvCrNUx" -->
## Accessing your web application
With your pod running, how do you access it? We mentioned that each pod gets its own IP address, but this address is internal to the cluster and isn’t accessible from outside of it. To make the pod accessible from the outside, you’ll expose it through a Service object. You’ll create a special service of type LoadBalancer, because if you create a regular service (a ClusterIP service), like the pod, it would also only be accessible from inside the cluster. By creating a LoadBalancer-type service, an external load balancer will be created and you can connect to the pod through the load balancer’s public IP.

To create the service, you’ll tell Kubernetes to expose the ReplicationController you created earlier:
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="2IJrV6BLrk1t" executionInfo={"status": "ok", "timestamp": 1628967062753, "user_tz": -330, "elapsed": 461, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="0d6ea62f-73a0-42e5-aef3-c8a7448bae95"
%%writefile 07_run_service.sh
kubectl expose deployment kubia-app --type LoadBalancer --port 80 --target-port 8080
```

<!-- #region id="EIQ_jjoRsYeR" -->
> Tip: ```k get services``` to get list of running services.
<!-- #endregion -->
