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

<!-- #region id="pvcZdiGza8se" -->
# Amazon Personalize Generic Module - Cleanup Layer
<!-- #endregion -->

```python id="NYcJc82W3ivV"
import boto3
import json
import time


class personalize_cleanup:
    def __init__(self):
        self._setup_connection()

    def _setup_connection(self):
        try:
            self.personalize = boto3.client('personalize')
            print("SUCCESS | We can communicate with Personalize!")
        except:
            print("ERROR | Connection can't be established!")

    def delete_campaign(self, campaign_arn):
        self.personalize.delete_campaign(campaignArn = campaign_arn)
    
    def delete_solution(self, solution_arn):
        self.personalize.delete_solution(solutionArn = solution_arn)
    
    def delete_tracker(self, tracker_arn):
        self.personalize.delete_event_tracker(eventTrackerArn = tracker_arn)
    
    def delete_filter(self, filter_arn):
        self.personalize.delete_filter(filterArn = filter_arn)
    
    def delete_dataset(self, dataset_arn):
        self.personalize.delete_dataset(datasetArn = dataset_arn)

    def delete_schema(self, schema_arn):
        self.personalize.delete_schema(schemaArn = schema_arn)

    def delete_dataset_group(self, dataset_group_arn):
        self.personalize.delete_dataset_group(datasetGroupArn = dataset_group_arn)
```

```python colab={"base_uri": "https://localhost:8080/"} id="47lrGWcD910V" executionInfo={"status": "ok", "timestamp": 1630426082282, "user_tz": -330, "elapsed": 8, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="18088fa3-e03b-42b4-9d43-82a9c95227fb"
pc = personalize_cleanup()
```

```python id="ZS2inCBa-TCw"
pc.delete_dataset_group(dataset_group_arn = 'arn:aws:personalize:us-east-1:746888961694:dataset-group/personalize-poc-movielens')
```
