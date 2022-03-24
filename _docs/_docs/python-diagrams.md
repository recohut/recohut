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

<!-- #region id="bqfvKsmLl98r" -->
# Python Diagrams
<!-- #endregion -->

```python id="2htffk2wzKS4" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1637083925242, "user_tz": -330, "elapsed": 7919, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="b4222570-38ba-46b6-e4b9-dc522c73fa0a"
!pip install diagrams
```

```python id="6vRVrJP9zL3i" colab={"base_uri": "https://localhost:8080/", "height": 434} executionInfo={"status": "ok", "timestamp": 1626790674360, "user_tz": -330, "elapsed": 977, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="a3b86a34-2d16-4779-eaf4-3101bd752915"
from diagrams import Diagram
diag = Diagram("Simple Website Diagram")
diag
```

```python id="BL_8D9BjzTTO" colab={"base_uri": "https://localhost:8080/", "height": 1000} executionInfo={"status": "ok", "timestamp": 1626694676531, "user_tz": -330, "elapsed": 1872, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="8ba2a7fb-c887-432f-8377-4d4b03b56c29"
from diagrams import Diagram, Cluster
from diagrams.aws.compute import EC2
from diagrams.aws.network import ELB
from diagrams.aws.network import Route53
from diagrams.onprem.database import PostgreSQL # Would typically use RDS from aws.database
from diagrams.onprem.inmemory import Redis # Would typically use ElastiCache from aws.database

with Diagram("Simple Website Diagram", direction='LR') as diag: # It's LR by default, but you have a few options with the orientation
    dns = Route53("dns")
    load_balancer = ELB("Load Balancer")
    database = PostgreSQL("User Database")
    cache = Redis("Cache")
    with Cluster("Webserver Cluster"):
        svc_group = [EC2("Webserver 1"),
                    EC2("Webserver 2"),
                    EC2("Webserver 3")]
    dns >> load_balancer >> svc_group
    svc_group >> cache
    svc_group >> database
diag
```

```python id="U5nMJy6a0LZl" colab={"base_uri": "https://localhost:8080/", "height": 1000} executionInfo={"status": "ok", "timestamp": 1601986397828, "user_tz": -330, "elapsed": 2622, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="d41cdbd4-a90e-47ba-f450-f6b9ab63b0a2"
from diagrams import Cluster, Diagram, Edge
from diagrams.onprem.analytics import Spark
from diagrams.onprem.compute import Server
from diagrams.onprem.database import PostgreSQL
from diagrams.onprem.inmemory import Redis
from diagrams.onprem.logging import Fluentd
from diagrams.onprem.monitoring import Grafana, Prometheus
from diagrams.onprem.network import Nginx
from diagrams.onprem.queue import Kafka

with Diagram(name="Advanced Web Service with On-Premise (colored)", show=False) as diag:
    ingress = Nginx("ingress")

    metrics = Prometheus("metric")
    metrics << Edge(color="firebrick", style="dashed") << Grafana("monitoring")

    with Cluster("Service Cluster"):
        grpcsvc = [
            Server("grpc1"),
            Server("grpc2"),
            Server("grpc3")]

    with Cluster("Sessions HA"):
        master = Redis("session")
        master - Edge(color="brown", style="dashed") - Redis("replica") << Edge(label="collect") << metrics
        grpcsvc >> Edge(color="brown") >> master

    with Cluster("Database HA"):
        master = PostgreSQL("users")
        master - Edge(color="brown", style="dotted") - PostgreSQL("slave") << Edge(label="collect") << metrics
        grpcsvc >> Edge(color="black") >> master

    aggregator = Fluentd("logging")
    aggregator >> Edge(label="parse") >> Kafka("stream") >> Edge(color="black", style="bold") >> Spark("analytics")

    ingress >> Edge(color="darkgreen") << grpcsvc >> Edge(color="darkorange") >> aggregator

diag
```

```python colab={"base_uri": "https://localhost:8080/", "height": 929} id="etgZ7EfIk4q_" executionInfo={"status": "ok", "timestamp": 1626695931265, "user_tz": -330, "elapsed": 788, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="02e4d5ed-6b34-42b0-eeb0-35d95ef68f51"
from diagrams import Diagram, Cluster
from diagrams.generic.storage import Storage
from diagrams.programming.language import Python
from diagrams.programming.framework import FastAPI
from diagrams.openstack.deployment import Helm


with Diagram("Simple Website Diagram", direction='LR') as diag:
  data = Storage("Data Loading")
  code = Python("Modeling")
  serve = FastAPI("API Endpoint")

  with Cluster("MLOps"):
    mlops = [Python("Modeling"),
             FastAPI("API Endpoint")]

  data >> code >> serve
  serve >> mlops
  mlops >> data


diag
```

```python colab={"base_uri": "https://localhost:8080/"} id="9XNgxhd3WnuP" executionInfo={"status": "ok", "timestamp": 1626792150581, "user_tz": -330, "elapsed": 563, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="b1ab9425-839b-4eb2-931f-91170023c4d5"
!wget -O colab.png https://avatars.githubusercontent.com/u/33467679?s=280&v=4
```

```python id="x6ZdIfbrog-S" colab={"base_uri": "https://localhost:8080/", "height": 1000} executionInfo={"status": "ok", "timestamp": 1626792316379, "user_tz": -330, "elapsed": 1850, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="8c785656-44b0-4cea-e63e-d768f94bd4c8"
!pip install diagrams

from diagrams import Diagram, Cluster, Edge
from diagrams.generic.storage import Storage
from diagrams.programming.language import Python
from diagrams.programming.framework import FastAPI
from diagrams.onprem.queue import Kafka
from diagrams.generic.device import Mobile
from diagrams.oci.storage import DataTransfer
from diagrams.custom import Custom

!wget -O colab.png https://avatars.githubusercontent.com/u/33467679?s=280&v=4

with Diagram("Simple Website Diagram", direction='LR', filename='diagram', graph_attr={"bgcolor": "transparent"}) as diag:
    data = Storage("Data Loading")
    code = Python("Modeling")
    serve = FastAPI("API Endpoint")
    helmc = Kafka("Helm Cluster")
    email = Mobile("Email")
    datatransfer = DataTransfer("Data Transfer")
    colab = Custom("Google Colab", "colab.png")
    data >> code
    code >> Edge(color="firebrick", style="dashed") >> serve
    data >> Edge(label="parse") >> helmc
    email >> datatransfer >> colab

diag
```

```python id="8xvuw8mIS6kz" colab={"base_uri": "https://localhost:8080/", "height": 634} executionInfo={"status": "ok", "timestamp": 1637084753368, "user_tz": -330, "elapsed": 1561, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="9c21101e-4330-426f-e555-aeb9be7013f9"
from diagrams import Diagram
from diagrams.custom import Custom

!wget -q --show-progress -O circle_blue.png https://icons.iconarchive.com/icons/custom-icon-design/flatastic-6/512/Circle-icon.png
!wget -q --show-progress -O rectangle_blue.png http://www.downloadclipart.net/medium/33179-blue-rectangle-images.png

with Diagram("Simple Website Diagram", direction='LR', filename='diagram', graph_attr={"bgcolor": "transparent"}) as diag:
    yui = Custom("yui", "circle_blue.png")
    xj = Custom("xj", "rectangle_blue.png")
    xj >> yui

diag
```

```python id="tMik6Hu72MUD"

```
