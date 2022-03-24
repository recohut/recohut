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
    language: python
    name: python3
---

<!-- #region id="sfz7gsZ-gWWw" -->
# Books recommendations with Kubeflow Pipelines on Scaleway Kapsule
> A simple book recommender system using Kubeflow pipeline

- toc: true
- badges: true
- comments: true
- categories: [Book, Kubeflow]
- image:
<!-- #endregion -->

<!-- #region id="iAA9KHclgWW0" -->
**Code examples based on "[CPU or GPU for your recommendation engine ?](https://blog.scaleway.com/2020/cpu-or-gpu-for-your-recommendation-engine/)" blogpost by Olga Petrova**

Kubeflow Pipelines notebook, by Fabien Da Silva
<!-- #endregion -->

<!-- #region id="TD4Z-jocgWW1" -->
**Goal**: In this example we are going to learn how to:


- Create a Pipeline with 6 tasks 
  - Download the Dataset
  - Prepare the Dataset
  - Train a model using Sci-kit Learn (NearestNeighbors)
  - Train a model using Pytorch (GPU)
  
- Use a Persistent Block Storage Volume with Kubeflow Pipelines via a NFS server (store/share datasets, models, ..)

- Use GPU efficiently thanks to the Kubeflow engine and Kapsule Auto Scaling
<!-- #endregion -->

<!-- #region id="Kolahy17gWW2" -->
**Prerequisites to run this notebook**:
 - Have a Kapsule cluster (At least, with one or more GP1-M instance. See Installation guide.
 - Kubeflow is deployed on tke Kapsule cluster
 - A GPU node pool is available (you can use Auto-Scaling from 0 to n nodes)
 - A NFS server is deployed in the Kubeflow Namespace. This will provide a persistent storage on Block Storage, and that can be accessed simultaneously by several Kubernetes Pods (several pipelines)
 
 A Scaleway tutorial will be available shortly to describe the installation process (By the time, a preliminary document is provided with this code)
<!-- #endregion -->

<!-- #region id="BdoqI-krgWW3" -->
**Usefull documentation**:

- [Kubeflow home page](https://kubeflow.org)
- [Kubeflow examples](https://github.com/kubeflow/examples)
- [Scaleway Kapsule Product page](https://www.scaleway.com/en/kubernetes-kapsule)
- [Kapsule documentation](https://www.scaleway.com/en/docs/get-started-with-scaleway-kubernetes-kapsule/)
- [Kubernetes home page](https://kubernetes.io/)
- [Scaleway GPU Instances Product page](https://www.scaleway.com/en/gpu-instances/)
- [Scaleway Container Registry Product Page](https://www.scaleway.com/en/container-registry/) 
- [Scaleway Object Storage Product page](https://www.scaleway.com/en/object-storage/)
- [Scaleway Object Storage documentation](https://www.scaleway.com/en/docs/object-storage-feature/) 
- [Scaleway Block Storage Product page](https://www.scaleway.com/en/block-storage/)
- [Scaleway Block Storage documentation](https://www.scaleway.com/en/docs/block-storage-overview/)
- [Access to the Scaleway Console](https://console.scaleway.com/kapsule/)
- [Access to the Kubernetes Console](http://localhost:8001/api/v1/namespaces/kubernetes-dashboard/services/https:kubernetes-dashboard:/proxy/#/login)
- [Access to the Kubeflow Dashboard](http://localhost:8080/)
 
<!-- #endregion -->

<!-- #region id="7o8WaNs5gWW4" -->
**Kubeflow Pipeline : Main concepts**

- A pipeline is defined in a Python function, which is then compiled in a DSL format for execution.
- A pipeline is composed of tasks (components), either defined in :
    - a `dsl.ContainerOp()` object, which execute a Docker Container containing the code to run for this task (Packaged in a docker image, this pipeline component is easily shareable and reusable)
    - or in a python function that will be converted on the fly into a Docker Container, thanks to the  `kfp.components.func_to_container_op()` API function (Great to prototype pipelines)
- Building a pipeline, is as simple as using some task's output as input from other tasks
- In order to be able to link the tasks in the pipelines, Kubeflow Pipelines needs to know the type of the input and output parameters of the python functions implementing the tasks
- Tasks can be specified to access a volume storage, and/or to use GPU ressources
- In order to execute the Docker Container running the pipeline task/components, the Docker images must contain all the python libraries requied by the code
    - When converting a python function into a ContainerOp via `kfp.components.func_to_container_op()`, the python imports must be included in the body of this Python function 
<!-- #endregion -->

<!-- #region id="f-a2V74KgWW5" -->
## Additional Package Installation

(You only need to execute the following 2 cells the first time you setup your Jupyter Server environment)
<!-- #endregion -->

```python id="VoRa0JxTgWW7"
!pip install kfp-server-api=='0.5.0' --user
!pip install kfp --upgrade
```

```python id="cowZWDSmgWW8"
# Need to restart the Jupyter Kernel
import os
os._exit(00)
```

<!-- #region id="DNo4XL0IgWW9" -->
## Notebook Setup
<!-- #endregion -->

```python id="AONyFr2pgWW-"
# -------------------------------------
#    Notebook configuration 'magic'
# -------------------------------------
%load_ext autoreload
%autoreload 2
%matplotlib inline
%config InlineBackend.figure_format = 'retina'
```

```python id="ncXeuCY2gWW-"
# -------------------------------------
#     Import Kubeflow Pipelines SDK 
# -------------------------------------
import kfp
import kfp.dsl as dsl
import kfp.notebook
import kfp.components as comp
from kfp import compiler
from kfp.components import func_to_container_op, InputPath, OutputPath
from kubernetes import client as k8s_client

```

<!-- #region id="h66RYTz1gWW_" -->
## Pipeline Component : Download the dataset

In this pipeline task, we will:
- Create a python function to download the dataset
- Have Kubeflow to convert the python function into a ContainerOp (basically this package the python function into a Docker Image, here based on the Tensorflow Docker image)

In this example the dataset has been collected from BookCrossing.com, a website dedicated to the practice of "releasing books into the wild" - leaving them in public places to be picked up and read by other members of the community.  

The downloaded datasets will be stored on the NFS Server (stored on peristed Block Storage volume created during the Kubeflow cluster setup). The Data will be located in the directory `/mnt/nfs/data/datasets/`
<!-- #endregion -->

```python id="D3PtF9H_gWXA"
def download_dataset(fname: str, origin: str, extract: bool = True,
                     cachedir: str = "./", cachesubdir: str = 'datasets')-> str:
    import tensorflow as tf
    import os  
    
    try:
        # Use Keras.utils to download the dataset archive
        data_path = tf.keras.utils.get_file(fname, origin,
                          extract=extract,
                          archive_format='auto',
                          cache_dir=cachedir,
                          cache_subdir=cachesubdir)

        output_dir = os.path.dirname(data_path)
        print("Path location to the dataset is {}".format(output_dir))
        print("{} contains {}".format(output_dir, os.listdir(output_dir)))
        
    except ConnectionError:
        print('Failed to download the dataset at url {}'.format(origin))
        return None
    
    # ------------------------------
    #     Write the Output of the
    #   Kubeflow Pipeline Component
    # ------------------------------
    try:
      # This works only inside Docker containers
      with open('/output.txt', 'w') as f:
        f.write(output_dir)

    except PermissionError:
        pass
    
    return output_dir
    
# -----------------------------------------
#        Convert the Python Function
#   into a Kubeflow Pipeline ContainerOp
# -----------------------------------------     
download_op = comp.func_to_container_op(download_dataset,
                                        base_image='tensorflow/tensorflow:latest')


```

<!-- #region id="EKPlnYOpgWXB" -->
## Pipeline Component : Prepare the Dataset

In this pipeline task, we will: Clean and prepare the dataset. The output of this task will be store on the NFS server in `/mnt/nfs/data/datasets/matrix.pickle`
<!-- #endregion -->

```python id="GgBvQHdegWXC"
def prepare_dataset(datadir: str)-> str:

    import pandas as pd
    import time
    
    print("Reading data from", datadir)
    
    # Load the books and rating Datasets into a Panda Dataframes
    books = pd.read_csv(datadir+'/BX-Books.csv', sep=';', error_bad_lines=False, encoding="latin-1")
    books.columns = ['ISBN', 'bookTitle', 'bookAuthor', 'yearOfPublication', 'publisher', 'imageUrlS', 'imageUrlM', 'imageUrlL']

    ratings = pd.read_csv(datadir+'/BX-Book-Ratings.csv', sep=';', error_bad_lines=False, encoding="latin-1")
    ratings.columns = ['userID', 'ISBN', 'bookRating']
    
    # Keep only Ratings above 5:
    ratings = ratings[ratings.bookRating > 5]

    # Drop the columns that we are not going to use
    columns = ['yearOfPublication', 'publisher', 'imageUrlS', 'imageUrlM', 'imageUrlL']
    books = books.drop(columns, axis=1)
    books = books.drop_duplicates(subset='ISBN', keep="first")
    books = books.set_index('ISBN', verify_integrity=True)
    
    
    # Keep only those books, that have at least 2 ratings:
    ratings_count = ratings.groupby(by='ISBN')['bookRating'].count().reset_index().rename(columns={'bookRating':'ratingCount'})

    ratings = pd.merge(ratings, ratings_count, on='ISBN')
    ratings = ratings[ratings.ratingCount > 2]
    ratings = ratings.drop(['ratingCount'], axis=1)

    print("Rating shape", ratings.shape[0])
    start = time.time()
    matrix = ratings.pivot(index='ISBN', columns='userID', values='bookRating').fillna(0)
    end = time.time()
    print('Time it took to pivot the ratings table: ', end - start)
    
    # Save Pandas dataframe
    output=datadir+'/matrix.pickle'
    matrix.to_pickle(output)
    
    # ------------------------------
    #     Write the Output of the
    #   Kubeflow Pipeline Component
    # ------------------------------
    try:
      # This works only inside Docker containers
      with open('/output.txt', 'w') as f:
        f.write(output)

    except PermissionError:
        pass
    
    return output
    
    
# -----------------------------------------
#        Convert the Python Function
#   into a Kubeflow Pipeline ContainerOp
# -----------------------------------------    
prepare_op = comp.func_to_container_op(prepare_dataset,
                                              base_image='tensorflow/tensorflow:latest',
                                              packages_to_install=['pandas'])  
```

<!-- #region id="7yaz0VDBgWXD" -->
## Pipeline Component :  Fit a model with SciKit-Learn and make one book prediction

In this pipeline task, we will fit a ScikitLearn NearestNeighbors model as described in Olga's Petrova blogpost "[CPU or GPU for your recommendation engine ?](https://blog.scaleway.com/2020/cpu-or-gpu-for-your-recommendation-engine/)". We will also make a few predictions (using hardcoded input features), and monitor the execution time for inference, here on CPU (Scaleway GP1-M instance)
<!-- #endregion -->

```python id="LoZLtzQ4gWXD"
def recommender_scikit(picklefile: str)-> str:
    import pandas as pd
    import time
    import os
    from joblib import dump
    
    print("Reading processed dataset dataframe pickle from", picklefile)
    
    # Reload Processed dataset in a Pandas DataFrame
    matrix = pd.read_pickle(picklefile)

    from scipy.sparse import csr_matrix
    from sklearn.neighbors import NearestNeighbors

    # Fit the model
    start = time.time()
    book_matrix = csr_matrix(matrix.values)
    recommender = NearestNeighbors(metric='cosine', algorithm='brute', n_neighbors=10).fit(book_matrix)
    print('Time to fit the NearestNeighbors model {}'.format(time.time()-start))
    
    # Compute 3 Book Recommendations inference and monitor the execution time:
    start = time.time()
    _, nearestBooks = recommender.kneighbors(matrix.loc['059035342X'].values.reshape(1, -1))
    print("----------------------------------------")
    print('Time to make a recommendation for ISBN 059035342X using the CSR matrix: {}'.format(time.time()-start))

    print("----------------------------------------")
    start = time.time()
    _, nearestBooks = recommender.kneighbors(matrix.loc['0439064872'].values.reshape(1, -1))
    print('Time to make a recommendation for ISBN 0439064872 using the CSR matrix: {}'.format(time.time()-start))
    
    print("----------------------------------------")
    start = time.time()
    _, nearestBooks = recommender.kneighbors(matrix.loc['0425189058'].values.reshape(1, -1))
    print('Time to make a recommendation for ISBN 0425189058 using the CSR matrix: {}'.format(time.time()-start))


    # Save the model
    output_dir = os.path.dirname(picklefile)
    output=output_dir+'/scikit-nearestneighbors.joblib'
    dump(recommender, output) 
    
    # ------------------------------
    #     Write the Output of the
    #   Kubeflow Pipeline Component
    # ------------------------------
    try:
      # This works only inside Docker containers
      with open('/output.txt', 'w') as f:
        f.write(output)

    except PermissionError:
        pass

    return output

# -----------------------------------------
#        Convert the Python Function
#   into a Kubeflow Pipeline ContainerOp
# -----------------------------------------    
recommender_scikit_op = comp.func_to_container_op(recommender_scikit,
                                              base_image='tensorflow/tensorflow:latest',
                                              packages_to_install=['pandas', 'joblib','scikit-learn'])  
```

<!-- #region id="t7ipPORbgWXF" -->
## Pipeline Component : Fit a model with Pytorch (GPU) and make one book prediction

In this pipeline task, we will fit a NearestNeighbors model, but this time using Pytorch which runs on GPU, as described in Olga's Petrova blogpost "[CPU or GPU for your recommendation engine ?](https://blog.scaleway.com/2020/cpu-or-gpu-for-your-recommendation-engine/)". We will also make a few predictions (using hardcoded input features), and monitor the execution time for inference, here on GPU (Scaleway Render-S instance with P100)
<!-- #endregion -->

```python id="IGIKvgeIgWXF"
def recommender_pytorch(picklefile: str)-> str:

    import pandas as pd
    import time
    import os
    import torch
    
    # Reload Processed dataset in a Pandas DataFrame
    matrix = pd.read_pickle(picklefile)

    # In PyTorch, you need to explicitely specify when you want an 
    # operation to be carried out on the GPU. 
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('Running on device: ', device)

    # Now we are going to simply append .to(device) to all of our torch 
    # tensors and modules, e.g.:
    cos_sim = torch.nn.CosineSimilarity(dim=1, eps=1e-6).to(device)

    # We start by transferring our recommendation matrix to the GPU:
    torch_matrix = torch.from_numpy(matrix.values).float().to(device)

    # Compute 3 Book Recommendations inference and monitor the execution time:
    start = time.time()
    
    ind = matrix.index.get_loc('059035342X')
    HPtensor = torch_matrix[ind,:].reshape(1, -1)

    # Now we can compute the cosine similarities:
    similarities = cos_sim(HPtensor, torch_matrix)
    _, nearestBooks = torch.topk(similarities, k=10)   
    print('Time to make a recommendation for ISBN 059035342X using PyTorch: {}'.format(time.time()-start))
    
    print("----------------------------------------")
    start = time.time()
    
    ind = matrix.index.get_loc('0439064872')
    HPtensor = torch_matrix[ind,:].reshape(1, -1)

    # Now we can compute the cosine similarities:
    similarities = cos_sim(HPtensor, torch_matrix)
    _, nearestBooks = torch.topk(similarities, k=10)
    print('Time to make a recommendation for ISBN 0439064872 using PyTorch: {}'.format(time.time()-start))
    
    print("----------------------------------------")
    start = time.time()
    
    ind = matrix.index.get_loc('0425189058')
    HPtensor = torch_matrix[ind,:].reshape(1, -1)

    # Now we can compute the cosine similarities:
    similarities = cos_sim(HPtensor, torch_matrix)
    _, nearestBooks = torch.topk(similarities, k=10)
    print('Time to make a recommendation for ISBN 0425189058 using PyTorch: {}'.format(time.time()-start))
    
    # Save the model
    output_dir = os.path.dirname(picklefile)
    output = output_dir + '/recommender.pt'
    torch.save(cos_sim.state_dict(), output)
    
    # ------------------------------
    #     Write the Output of the
    #   Kubeflow Pipeline Component
    # ------------------------------
    try:
      # This works only inside Docker containers
      with open('/output.txt', 'w') as f:
        f.write(output)

    except PermissionError:
        pass

    return output

# -----------------------------------------
#        Convert the Python Function
#   into a Kubeflow Pipeline ContainerOp
# -----------------------------------------    
recommender_pytorch_op = comp.func_to_container_op(recommender_pytorch,
                                             base_image='pytorch/pytorch:latest',
                                             packages_to_install=['pandas','scikit-learn'])  
```

<!-- #region id="vIIX40TugWXG" -->
## Build the Pipeline
<!-- #endregion -->

```python id="ZAozDpE8gWXG"
@dsl.pipeline(
    name="Book Recommendation Engine ",
    description="A Basic example to build a recommendation engine using Kubeflow Pipelines"
)
def book_recommender():
    
    def mount_nfs_helper(container_op):
        ''' Helper Function to mount a NFS Volume to the ContainerOp task'''
        # NFS PVC details
        claim_name='nfs'
        name='workdir'
        mount_path='/mnt/nfs'

        # Add andd Mount the NFS volume to the ContainerOp
        nfs_pvc = k8s_client.V1PersistentVolumeClaimVolumeSource(claim_name=claim_name)
        container_op.add_volume(k8s_client.V1Volume(name=name,
                                              persistent_volume_claim=nfs_pvc))
        container_op.add_volume_mount(k8s_client.V1VolumeMount(mount_path=mount_path, name=name))
        return container_op
    
    
    
    # Pipeline's task 1 : Download dataset
    download_task = download_op(fname="BX-CSV-Dump.zip", 
                                origin="http://www2.informatik.uni-freiburg.de/~cziegler/BX/BX-CSV-Dump.zip",
                                cachedir="/mnt/nfs/data")
    download_task = mount_nfs_helper(download_task)
 
    # Pipeline's task 2 : Prepare the Dataset
    prepare_task = prepare_op(datadir=download_task.output)
    prepare_task = mount_nfs_helper(prepare_task)

    # Pipeline's task 3 : Train the Scikit-learn NearestNeighbors model
    recommender_scikit_task = recommender_scikit_op(picklefile=prepare_task.output)
    recommender_scikit_task = mount_nfs_helper(recommender_scikit_task)
 
    # Pipeline's task 3 : Fit the model and Prediction for one isbn with Pytorch on GPU (NearestNeighbors)
    recommender_pytorch_task = recommender_pytorch_op(picklefile=prepare_task.output)
    recommender_pytorch_task = mount_nfs_helper(recommender_pytorch_task)
    recommender_pytorch_task.set_gpu_limit(1)

    # Pipeline's task 4 : The goal of this task is to trigger a new GPU node to be spawned in the cluster
    # It trains the Scikit-learn NearestNeighbors model on a Render-S 
    # (slightly better execution time than on GTM-1 because the CPU on the Render-S is a higher end model)
    recommender_scikit_task2 = recommender_scikit_op(picklefile=prepare_task.output)
    recommender_scikit_task2 = mount_nfs_helper(recommender_scikit_task2)
    recommender_scikit_task2.set_gpu_limit(1)
```

<!-- #region id="xYkYJLttgWXH" -->
## Execute the Pipeline
<!-- #endregion -->

```python id="kcFzznIHgWXH"
#--------------------------------------------------
#              Compile the pipeline 
#        (composed here of 3 tasks)
#--------------------------------------------------
PACKAGE_NAME = book_recommender.__name__ + '.yaml'
kfp.compiler.Compiler().compile(pipeline_func=book_recommender, 
                                package_path=PACKAGE_NAME)

#--------------------------------------------------
#      Create/Reuse an Experiment in Kubeflow
#--------------------------------------------------
EXPERIMENT_NAME = "Tests"
client = kfp.Client()
try:
    experiment = client.get_experiment(experiment_name=EXPERIMENT_NAME)
except:
    experiment = client.create_experiment(EXPERIMENT_NAME)

#-------------------------------------------------- 
#             Submit a pipeline run
#
#    => This will create a PVC of 20Gi on 
#          a Block Storage Volume
#--------------------------------------------------
RUN_NAME = book_recommender.__name__ + ' run'
arguments = {}

run_result = client.run_pipeline(experiment_id = experiment.id, 
                                 job_name = RUN_NAME, 
                                 pipeline_package_path = PACKAGE_NAME,
                                 params = arguments
                                )

```

<!-- #region id="kPQ1uvXigWXJ" -->
## Compare some results:

**Scikit-Learn**
```
----------------------------------------
Time to make a recommendation for ISBN 059035342X using the CSR matrix: 0.023803234100341797
----------------------------------------
Time to make a recommendation for ISBN 0439064872 using the CSR matrix: 0.016010761260986328
----------------------------------------
Time to make a recommendation for ISBN 0425189058 using the CSR matrix: 0.008635520935058594
```


**Pytorch**
```
Running on device:  cuda:0
Time to make a recommendation for ISBN 059035342X using PyTorch: 0.023251771926879883
----------------------------------------
Time to make a recommendation for ISBN 0439064872 using PyTorch: 0.00032520294189453125
----------------------------------------
Time to make a recommendation for ISBN 0425189058 using PyTorch: 0.0002586841583251953
```


<!-- #endregion -->

<!-- #region id="pI_0NegUgWXJ" -->
#### Annex: How to access the data on the PVC ?

- Create a `nfs_access.yaml` file (adjust the `claimName`)
````
apiVersion: v1
kind: Pod
metadata:
  name: nfs-access
spec:
  containers:
  - name: bash
    image: bash:latest
    command: ["/bin/sh", "-ec", "while :; do echo '.'; sleep 5 ; done"]
    volumeMounts:
    - mountPath: "/mnt/nfs"
      name: workdir
  volumes:
  - name: workdir
    persistentVolumeClaim:
      claimName: nfs
```
- Create a Pod from this specifications 
```
kubectl apply -f nfs_access.yaml -n kubeflow
```
- Connect with a shell to this pods (note: there is no prompt on the command line:
```
kubectl exec -t -i -n kubeflow nfs-access -- /bin/sh
# you can explore the /mnt/nfs/data directory from here
```
- In a similar manner, you can use the kubectl cp command to copy data from/to the PVC

<!-- #endregion -->
