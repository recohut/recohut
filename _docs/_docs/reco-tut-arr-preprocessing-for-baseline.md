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

```python id="-UOOzCs9ukul" executionInfo={"status": "ok", "timestamp": 1628004089548, "user_tz": -330, "elapsed": 3, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
project_name = "reco-tut-arr"; branch = "main"; account = "sparsh-ai"
```

```python colab={"base_uri": "https://localhost:8080/"} id="PYvHGli8ukum" executionInfo={"status": "ok", "timestamp": 1628004100226, "user_tz": -330, "elapsed": 8075, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="78680be3-aba8-4c18-c317-0559e4622a6b"
import os

if not os.path.exists('/content/reco-tut-arr'):
    !cp /content/drive/MyDrive/mykeys.py /content
    import mykeys
    !rm /content/mykeys.py
    path = "/content/" + project_name; 
    !mkdir "{path}"
    %cd "{path}"
    import sys; sys.path.append(path)
    !git config --global user.email "arr@recohut.com"
    !git config --global user.name  "reco-tut-arr"
    !git init
    !git remote add origin https://"{mykeys.git_token}":x-oauth-basic@github.com/"{account}"/"{project_name}".git
    !git pull origin "{branch}"
    !git checkout main
else:
    %cd '/content/reco-tut-arr'
```

```python id="nz2yphhRkbhY"
!git status
!git add . && git commit -m 'commit' && git push origin main
```

```python id="J6GnSXizHC08" executionInfo={"status": "ok", "timestamp": 1628001483283, "user_tz": -330, "elapsed": 939, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score,classification_report
from sklearn.preprocessing import LabelEncoder

import gc
import datetime
from tqdm.notebook import tqdm

import warnings
warnings.filterwarnings("ignore")

np.random.seed(0)
```

```python colab={"base_uri": "https://localhost:8080/"} id="9vagQoW_RXoJ" executionInfo={"status": "ok", "timestamp": 1627900566066, "user_tz": -330, "elapsed": 609, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="9bb5c61a-0919-4a69-c191-87dbc517df39"
data_path = dict()

for dirname, _, filenames in os.walk('./data/bronze'):
    for filename in filenames:
        if filename.endswith('.parquet.gz'):
            name = filename.split('.')[0]
            data_path[name] = os.path.join(dirname, filename)

data_path
```

```python id="UCbl5-5NTcQh"
orders = pd.read_parquet(data_path['orders'])

vendors = pd.read_parquet(data_path['vendors'])
vendors = vendors.add_prefix('v_')
```

```python colab={"base_uri": "https://localhost:8080/"} id="79MpoeVsR9IS" executionInfo={"status": "ok", "timestamp": 1627900575953, "user_tz": -330, "elapsed": 8431, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="885d1d9a-24c7-4840-8458-ca7d55b54789"
test_customers = pd.read_parquet(data_path['test_customers'])
test_customers = test_customers[test_customers.duplicated('akeed_customer_id', keep='first')==False].reset_index(drop=True)
test_customers.rename(columns={'akeed_customer_id': 'customer_id'}, inplace=True)

test_locations = pd.read_parquet(data_path['test_locations'])

test_customer_detail = pd.merge(test_locations, test_customers, on='customer_id', how='left')
test_customer_detail = test_customer_detail.add_prefix('c_')
test = test_customer_detail.assign(key=1).merge(vendors.assign(key=1), on='key').drop('key', axis=1)

test_customers.shape, test_locations.shape, vendors.shape, test_customer_detail.shape, test.shape
```

```python colab={"base_uri": "https://localhost:8080/", "height": 309} id="Lg2YKcUgS241" executionInfo={"status": "ok", "timestamp": 1627900575954, "user_tz": -330, "elapsed": 39, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="1aefc158-4d61-442c-8f82-b2ebdcb8219e"
test.head()
```

```python colab={"base_uri": "https://localhost:8080/"} id="Z12sTqxTDvMB" executionInfo={"status": "ok", "timestamp": 1627900606231, "user_tz": -330, "elapsed": 30291, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="3e4fa86a-95e6-4196-c2de-5684c73f0b5a"
train_customers = pd.read_parquet(data_path['train_customers'])
train_customers = train_customers[train_customers.duplicated('akeed_customer_id', keep='first')==False].reset_index(drop=True)
train_customers.rename(columns={'akeed_customer_id': 'customer_id'}, inplace=True)

train_locations = pd.read_parquet(data_path['train_locations'])
train_customer_detail = pd.merge(train_locations, train_customers, on='customer_id', how='left')
train_customer_detail = train_customer_detail.add_prefix('c_')

train = train_customer_detail.assign(key=1).merge(vendors.assign(key=1), on='key').drop('key', axis=1)

train_customers.shape, train_locations.shape, train_customer_detail.shape, train.shape
```

```python colab={"base_uri": "https://localhost:8080/", "height": 309} id="Vpp85I5AS9iE" executionInfo={"status": "ok", "timestamp": 1627900606232, "user_tz": -330, "elapsed": 33, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="451c9c24-104c-44f4-a831-702c184e890e"
train.head()
```

```python id="kjY-hztNS-hb"
test["CID X LOC_NUM X VENDOR"] = test["c_customer_id"].astype(str)+' X '+ test["c_location_number"].astype(str)+' X '+ test["v_id"].astype(str)
train["CID X LOC_NUM X VENDOR"] = train["c_customer_id"].astype(str)+' X '+ train["c_location_number"].astype(str)+' X '+ train["v_id"].astype(str)
```

```python id="mUh4pLicTNKl"
train['target'] = 0
mask = (train["CID X LOC_NUM X VENDOR"].isin(list(set(train['CID X LOC_NUM X VENDOR']).intersection(set(orders['CID X LOC_NUM X VENDOR'])))))
train['target'][mask] = 1
```

```python colab={"base_uri": "https://localhost:8080/"} id="su7mZzbBTnJz" executionInfo={"status": "ok", "timestamp": 1627900634259, "user_tz": -330, "elapsed": 36, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="77bce956-2e07-4596-c117-814053e2f39a"
del test_customers
del test_locations
del vendors
del test_customer_detail
del train_customers
del train_locations
del train_customer_detail
del orders
del mask
gc.collect()
```

```python id="jJ-SPM2vU5_X"
test_id=test['CID X LOC_NUM X VENDOR']

cols_drop = ['v_is_akeed_delivering', 'v_open_close_flags','v_one_click_vendor',
             'v_country_id','v_city_id', 'v_display_orders',
            'c_customer_id','CID X LOC_NUM X VENDOR','v_authentication_id',
             'c_language','v_language','v_vendor_tag']

train.drop(cols_drop, axis = 1,inplace=True)
test.drop(cols_drop, axis = 1,inplace=True)
```

```python id="oFIeZyHta7p7"
train.to_parquet('./data/silver/train.parquet.gzip', compression='gzip')
test.to_parquet('./data/silver/test.parquet.gzip', compression='gzip')
```

```python id="2ycwD6eLcza3"
train = pd.read_parquet('./data/silver/train.parquet.gzip')
test = pd.read_parquet('./data/silver/test.parquet.gzip')
```

```python id="g9edE_FNiG4B" executionInfo={"status": "ok", "timestamp": 1628001562579, "user_tz": -330, "elapsed": 4261, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
train.c_gender=train.c_gender.str.strip()
test.c_gender=test.c_gender.str.strip()
train.c_gender[pd.isnull(train.c_gender)]  = 'NaN'
test.c_gender[pd.isnull(test.c_gender)]  = 'NaN'

train.replace({'c_gender': {'male': 'Male', '': 'NaN','?????':'NaN'}},inplace=True)
test.replace({'c_gender': {'male': 'Male', '': 'NaN','?????':'NaN'}},inplace=True)
```

```python id="F3DYMLMqibMU" executionInfo={"status": "ok", "timestamp": 1628001563462, "user_tz": -330, "elapsed": 909, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
train.replace({'v_OpeningTime2': {'-': 'NaN'}},inplace=True)
test.replace({'v_OpeningTime2': {'-': 'NaN'}},inplace=True)
```

```python id="9eAhP82VdwVw" executionInfo={"status": "ok", "timestamp": 1628001563464, "user_tz": -330, "elapsed": 12, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
def label_encoder(collist, old_map=None):
  map = {}
  index=0
  if old_map:
    map = old_map
    index = max(old_map.values())+1
  for x in tqdm(collist):
    if x not in map.keys():
      map[x] = index
      index+=1
  map[np.NaN] = np.NaN
  return map
```

```python colab={"base_uri": "https://localhost:8080/", "height": 1000, "referenced_widgets": ["535e841f8a654f2985ecd08b6fa5f2ea", "3f9afc094ad84024bf2d25a31784867b", "31a36948dfdd40f3a53479266e478c47", "2aea79b8861540f0b1dd75834e6fe9ad", "868bc0c52e6541cb855322b02bba3742", "444387bb67ff4f60bd33cb4deb247e0b", "6a12ed579dd14a818e25135606565729", "fcc38c1c56b34bcca4e4693286d5a4c5", "dc0e66a940c743deb887a66ef78b7ada", "214f52acf41c4bbab6b1a51896f321c7", "006ba7b1dd134dacbc0d101c4f19ee60", "23c89fdc21bd40678ec1bc8443b8923c", "02729877e2e24c8099e6be2248983c32", "bff5ea0031b24d57af7787832d5f19bc", "105f57aa103e4b7ba5821afea36ecf3e", "3fe23a6f19ff4339ac25ca10a55365f7", "ea632e2bbbf54dd7b42d59568c76ae30", "f84d10b9c80f468b8d6d11c311484e6b", "82624b74a29740bb94ad937182d6ee96", "6cdeb5e765f44b7f8a63afe680ce2e31", "3964a67d383c48839e26b4fdfac6c5fb", "c5d08d1b4dca409bbd06c502292a840f", "7625f6ef92a74826a44a1eed9732794b", "87b89af5a98b466aa1de06c828b432af", "cb5df240df114785bc4e63825738bb50", "41264d7260d7410cb274196f4464092a", "62b555ae60264ee1a2d3d4e6263cddbf", "fa4187e34208469fb97f41325e27dd57", "93ddf749a6014e7a808de59bcedc9017", "1afd4ae7ad8c46a0b76f4ffc731b855e", "8cfa66ce588a4206a6f89b85e794736a", "41169cc5b3d342b385880f26f60a63b4", "350552ba7ec646e198628bab45b30fea", "6c1c249d3ae641288810981de2dc321e", "e3fb803c7e2a44cfa495d064a4721f19", "e0ce6010920e4b9fa380961379ff3f86", "f6f4e4e05ae146f1a1b4bcd7aca18667", "f880a146be34424bb8543855a81facb8", "067fa2ca45684c239f0c6de8a92dcfd6", "704773e642c14ba6be625dcbbf0c9b47", "c3a8331410694dbeaf1449d09013006c", "7dbdd13ec5fa402a80cee6e1b53b2f3e", "a31d19a539184449b5b73182200488ce", "c4eda8ce1af449e1af9430b49aff437f", "a10529b9667d43c9be9ec251c4b9d419", "7d69829b80d7462aa2df11cc03335690", "24a060ab605d4d959ec755780642fb60", "dec6e939016b440da2bca766ded20701", "087d4a3fe0c0440b943274415094eb3e", "dd067261776940658fde9a2a164f36af", "10091969763d4283aca82fe18d682943", "8474c35200a542dca50765a8e5a0d451", "a66f37c518314949a07d4c6d98a64b04", "69c09fa0165e4e049d22b5a907533931", "96f84728dc8c4c8d810129f6436491aa", "ca603836439e4e7e9787d79348d8dd88", "cdce85b5c1f84df79105fda6ea061546", "698c3e25baa54c9485830de806bf4657", "436c5755a45a4cfbbf2502096935555f", "a56c7ed9d8ed4a34811456516a0f1e31", "d7c94f335259431e8312a3390e4d3188", "1b046715a3204aa3ac2aed5257f81368", "244b9b6795bb4aa79e8e09f0b747e452", "178fb5c9c48740eda9e4fe8a1ec759f0", "e6590474ad8c4bb78778f53cad1fe96e", "eb7e8ffa3d1c46ac8e1f5757bf6ebe0b", "f4c0c215829741b18bd26252693a1824", "8a5b7e693a1648858b8a838ca716be7c", "8cc3b4e3c8a24b79856779af72dca34b", "80465a83abee459ca00fe74b1c43d3a4", "bf2b264780c747d4a092562d8bb8e29a", "36e491a069ed4e888015816be3123472", "106215a8fcb0469a9b369d669843443f", "adfec49163db48c4bed16fbb657af690", "bba21a1c137840a78894afa1ceb6bd2e", "c70b865368264ada9f8b54f41be872d0", "9f3f5bd6a58946af937c99d91d7cad78", "5e014462c61040928e534472a623748b", "f430abf133cd47e7bcbf5f108c5f8775", "2756d318e8ec4e5db874b4a44c72308c", "65e7943f43e741e5a4450eb1e37051e8", "5ecb26a24eb6457cbffa48d2eb88424b", "9784b61d263d45f4b9f0f60a44465284", "a747a1450d314abead582f00a1e2b539", "ed15df777de04aaaba03424587876375", "4864cc2fe79a499d8950f7eb97774ea7", "c49097acdd7f48ac8bd86f3878c1cfe3", "1a1d254c09e74aaaad00fc548335a299", "22fc91a4595d44fbbc48af4769568cb2", "f0c1ea21ecb94001b4bcf3e1c13aeda4", "d3c53933004b4bb190fc736df2c9aaf5", "47b309f3e7f4401c92d58470319010e1", "ee477b998f9c4ba3951e48f715d3b5f5", "96ede690408c439f8c7b1579e90dc838", "af74470de2be4f62b2e9e2cec3506da1", "c636024ee4134815ad0edc8384ce1072", "48fc7ead6bea495b897e9d2f6cdfc327", "fc2a48a8c8294b99b33616768de00245", "e0750e8b49524f68ad633085beb3c369", "f188b33b832042e98f9a16cbcca5d099", "1901b1b0fefd496fb7cb9416d4bcca7c", "777758ac06c64e0b84b44a770131edf8", "6ed4735b486a4b6b9c4f774f754cb722", "5b47374c74a541f8b1dcddaff6020f68", "1adff13a33834708bf4217f76f4fe59d", "7d1b4647a10642ca813d77d9dbf428a8", "d94f0057ae9f40d5910b2ecbe173984e", "02a3cb51de3d4bc7918928a566259e1b", "5d7f05930d614b6aad0e989cffed7477", "968d48d291cc428295de6dcfd8052fea", "614cc1d8f32b48f6abf4bfdef4887368", "8f4a8e7c268a438e807a1181d4fc5a26", "6220c025095142a4b9f715c34c10e368", "25230ba890b840bb97f5d18c2efd943b", "4919a0a0d592460485b9a200fcba29b3", "3f61fcb0deab479fae8214efd5e24882", "f37b5dcfdc8347a0a33b9a71e6e32bb7", "c47c88079d504cde8ba88297b52d76c8", "1109dcedb1634a8596fce1ae165c81d8", "0b16438acc9d4427b438ea4d69e98f5a", "1c4acd271b0d4d938e34192d9a76d155", "f6711a8a244a4dd685ec3cad31e0a0e3", "b4eb052a9f6342a1809138613213538b", "e248b28f6f6c4339b1b1d61d7143a0a7", "ad3fb7a5c0e24cdba030ba25797b1aa7", "35c2282b14364d51b6e761cd588971a0", "31b9fdaaeea146b4b1f805b682410f1f", "d0359cdc396a4a498603a5c7832548db", "aba108ce51d84c03a5e6a2ae1980ea87", "7bb556ae92144379b41db347507a60d4", "5cb7b5cfe2eb43b2846bbb79cc39f575", "47c458caad014d4ba3e2ce69fbfa4ca5", "94e9d2a55e8841b1bad16f2af585321b", "f41b7c1940f54f0992c2f04e07cda04f", "1662a2d4845e4e4d9c1102b9cfa1025f", "364f4c9727eb4fbaa81ca79fdecef201", "41397947ff5e43bd917b5b914504acf7", "825b56d6efb746f3be62a77dc86dd229", "770b37734831463187471ec51d1bcd51", "4cdae3bf754940de829ac089dbc503df", "68b426b7cece452d9347132768a28847", "6762c36d95584d388135f17446ae7af8", "3ab2ae8ec56249daab62386371857839", "da4d35136f5c4994998a664bdd937408", "0df82768e7554c9c854febd9a9ace48e", "a9ac529f28904ac9a11a1f2ca3411555", "d330084e2d874936ab381ff7b77b406a", "acb38263ed3443618195a54cd9c67dfd", "f0fd79a34b3e4f82a8d4f0144742ddb6", "014dfef7e50341d6994dc6a0b4d89f41", "03ad9e82ea99435a9c78b7afcc4c29ef", "5d9a51e2e71245f99713791e4e63aeab", "65d197d0aeef47f8b1d415f8bda231bf", "8ce32c0efe86420d8e03580bd6aee327", "1ace933e7ea5478381b6bd482c9739b1", "07688413b9f04505bf09055c94df4062", "a8448633d6564c91b359274a925daa89", "c448563e8ef048ebae815ff29711bc51", "ad19d3e424ae49229bf14a933bff780d", "6e6759e013a945bca2bacd314b5c4ae3", "9bfd97898eb24f6d926458cec18df666", "03a726cbfaf84dd39f53ed00e4b17475", "e3bb6d497dfc41c7859fc585e43a37a7", "5b84719b5d60495bb8ff8ea5ee405dd9", "4b0c396a23f743e080d4e97830e3f592", "d3cef8e1cada49ce832e242103e70388", "bd22699fc2634a12af7831f2a40aaa48", "4be7614701ac4dfeab889304782274bb", "a375a229eb3d4f54b8c2ce6c62085149", "9ce85892461944bb9bdc55d0d7812611", "247e261f25a747d884def4fdb0e35e92", "7f7a03ed40b14bd98dcf59fd1366fe75", "a1ff81cde41845b087f32a73b1834e73", "87d4476814724528bebf36fed82e7730", "b8a09158f23b46679f5b39a8f99a21db", "9f0ba8f0a69d4d7981771f4e5d63e20c", "bad5f8d4212349b29e3b8dd52ff18bfa", "737bd8f63ab44509b33284297afdc2a9", "2c1ce92b877943fe91a880fcb431f83c", "85d2a36031144a9bb2c8388f2389fec8", "702cf4167de542a295b56d9a2456c926", "4a1bbe04eb5840feaa511f597f6f9964", "54822bbd827c4e96873f519114d8f111", "ddb591ed4d0d407c926d4832c4de06b8", "d14a27b9b3384035abd8fc94c6cdcd46", "f610ac8a81b54488bd80dd6144be191c", "e220b9dce0f6474db0bd8ed8f0bfbd68", "4084d829409546948a0f169e52ccaedd", "f7b4c057ab0347b38405c9f247ae59d9", "0c60db31de4b4ae6a7d838e8fb4046b3", "bc0df31a94fa4d2e9172c5ea33cf0271", "60dc0c7dd22a47f7a6872153a1630e4d", "f0b7279f750844749eb52492985a94a0", "4498b8d08b3847a789d9771aca96f318", "02c04f6ca8f548e680062262bc22201e", "6eea9cd88e28433d98e85cdb9362ac84", "a6188988e5fd4f9fb27f52e8101f8bb0", "7618609600ab486b9eb7cb6bb53ce989", "ac2bb13f16664dc4a7421ac85a0c496b", "73ce79b93da546a195c2974c99edbf21", "ac4ce0ed50f24688a926e4f72837836e", "7b4f923eeacf40b7bc26738266a2e1eb", "5762c4d140be4914a78a097f49840e35", "4a705a47ff31417cbcd6de7d672f0b57", "39dfa4ddbb034eaa8319da2d742b84e2", "db6f485d92eb4250842a478806a00cf8", "8322aaae4fd8406889acf64b02d57034", "281929ab38104eaca0fe23f7b7ae6d84", "5d1e342e92e4430cb0c4a15ae31963c0", "f5ed8a41ea424c12b6c247bb8793ed24", "3dacdba965ba44208f23d1c2bf9d8e26", "5a6e5004283348739edcb14eccea8405", "1ecdb0ef0117463e9106a308e78aa2d4", "00fe892170e1404b9ccca60267e73361", "fa2b8011c66940dfaf4599db0459f272", "961a99e3e18240ef84c27e4c638a1331", "e73ec1a0ae3942b288126dc07a453a9e", "89c4a76554bf4d49ae9a05f5e941da63", "47665b9ae52746dc982354735f5b0a84", "9c492b69f0ff4b0885df2d7a71ec436d", "a398e4510c7649e5b957b48f5d2d1ca7", "794c3388627749f2bcd81ebfc303b32e", "cbeb21cd1b114c4da88b72ad8f6385eb", "2840299d4e9c4b0f9a4b27ea7c3ddb37", "ea74d1ddf1304c6686d93f487e333888", "39a10ff317444e019baf132e51c4d8c3", "791a9372b87e4d439748c4d7af04db26", "41142604f2034f50ba6513dd8c73a306", "8e42e5f07ca04c949c681f2dc3d9f240", "b978ac88e9f84acd87e79ec7a8bc81c0", "2aa1f60b30b24434ac3bade3e54cdbd9", "5db6a464095f4f7f84bc3e9f23dced84", "63eaf9582b114b339bcfd448a98f8dc0", "d6ae821b892e477ea66c436c78c5e824", "8afa29dd696745d69a36c52fecb1adbd", "fadab000670b416cbdab9a8f145a8bf8", "cc7a9df3e0654187be8342e658d5e87b", "751a9e2b3a9d4c78956741e3878c0264", "6eb4c6f63def4164874d4875e021172c", "a2ffd51c9f5f4120a93f57544e0525d1", "bd1a277eaf704092ab5345c2c5ff150a", "cad7e646ebf54f0495cb829d316e21ef", "e3efd3407b7b43349040d6dc5bcc87f7", "9c6879de585543309b5aaaaf8bef5075", "8cf85743497e4f2b9fc841d2fb867e1c", "839898978e8e418ea515ae9be2202a30", "028ec9c0442f43759041e7813e28d179", "67327f6e87094aa3b833f9f127a5350c", "196ac8f1475f4618b674a674af5041a9", "80842ef0d5174fdf8343002e865dfade", "3fd5585c57b44e88ba62f0419e6eecd6", "bce529070f2541b0b4d8bdea6369755b", "7248b5ae37814e22a97490f7097d3523", "901fefb3c5af4e2d9ba80577b0f371e9", "6f61dd76c77043ab9a53e176b6b30cfc", "1fa75c7fa7d044b3a1d6e3601ef1a71c", "df444cc1aa534898b01ce108b8a08fac", "40a4fd898f3e44858d361ad58ceeb836", "d30db1b28afc4c518c8137ff8201bfb3", "5be6482436a54b55959b62766900acfb", "234df900ea3f4ce8950dbedc98434065", "95a70f152a914cae98323647d1925f9b", "3733fb9b0784440b82ffddd25e7c161f", "ec47380d4bda436aa6eb5e5e0c24d35b", "48a7dae4ebba44a28d7a76a6ce2934df", "3aae97dd06784a38a4cdd72b067b6ace", "e76b9420d5744a88843e02f2436783c7", "24a0a2013242442b885876973d9c1fca", "f03fe34073db4554b0549b3a1d58a540", "2f449fe6695b402ea1d1ff2d1c2d985e", "c15ddc31458143cca57065765cfecf3a", "a31fe2288b4a4490ab3ce5d522a42138", "ef5f9fbe4efb4f939918e9fb199aa6a3", "74e53fc6a808475a9ad7a17f89dfa35a", "9bac6ed5ff2f4bb8a9b75fe6cd6a4879", "9d9b07698c0f4b52ac70d29c0a22c450", "ad642c4ad7154577a309367012bf093a", "1503c6e758784e7d837bdfc158a762b9", "56a8388b1cf04be894a4b9068485d2e2", "6b0580ae01a24108ba08a035b743cdfa"]} id="1wD84hxigAF4" executionInfo={"status": "ok", "timestamp": 1628001683585, "user_tz": -330, "elapsed": 109405, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="620406e8-ad0b-4f7e-f265-eef01da4090c"
maps = {}
cols = ['c_location_type', 'c_gender', 'v_vendor_category_en',
        'v_OpeningTime', 'v_OpeningTime2', 'v_sunday_from_time1', 'v_sunday_to_time1',
        'v_sunday_from_time2', 'v_sunday_to_time2', 'v_monday_from_time1', 'v_monday_to_time1',
       'v_monday_from_time2', 'v_monday_to_time2', 'v_tuesday_from_time1',
       'v_tuesday_to_time1', 'v_tuesday_from_time2', 'v_tuesday_to_time2',
       'v_wednesday_from_time1', 'v_wednesday_to_time1',
       'v_wednesday_from_time2', 'v_wednesday_to_time2',
       'v_thursday_from_time1', 'v_thursday_to_time1', 'v_thursday_from_time2',
       'v_thursday_to_time2', 'v_friday_from_time1', 'v_friday_to_time1',
       'v_friday_from_time2', 'v_friday_to_time2', 'v_saturday_from_time1',
       'v_saturday_to_time1', 'v_saturday_from_time2', 'v_saturday_to_time2',
        'v_primary_tags', 'v_vendor_tag_name']

for col in cols:
    maps[col] = label_encoder(train[col].tolist())
```

```python colab={"base_uri": "https://localhost:8080/"} id="fV0z06H2do5V" executionInfo={"status": "ok", "timestamp": 1628001886092, "user_tz": -330, "elapsed": 791, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="eb92877d-3189-467f-c561-ae7c19c3de17"
for k,v in maps.items():
    print('\n{}\n'.format('='*100))
    print('Mapping for {}\n'.format(k))
    print(v)
```

```python id="IOZIiKzoghqI" executionInfo={"status": "ok", "timestamp": 1628002665650, "user_tz": -330, "elapsed": 135959, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
for col in cols:
    train.loc[:,col] = train[col].map(maps[col])
    test.loc[:,col] = test[col].map(maps[col])
    train.loc[:,col] = train.loc[:,col].astype('int')
    test.loc[:,col] = test.loc[:,col].astype('int')
    gc.collect()
```

```python id="vYrCmetLVwFA" executionInfo={"status": "ok", "timestamp": 1628002669892, "user_tz": -330, "elapsed": 4255, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
train['c_created_at'] = pd.to_datetime(train['c_created_at'], yearfirst=True)
test['c_created_at'] = pd.to_datetime(test['c_created_at'], yearfirst=True)
train['c_updated_at'] = pd.to_datetime(train['c_updated_at'], yearfirst=True)
test['c_updated_at'] = pd.to_datetime(test['c_updated_at'], yearfirst=True)
train['v_created_at'] = pd.to_datetime(train['v_created_at'], yearfirst=True)
test['v_created_at'] = pd.to_datetime(test['v_created_at'], yearfirst=True)
train['v_updated_at'] = pd.to_datetime(train['v_updated_at'], yearfirst=True)
test['v_updated_at'] = pd.to_datetime(test['v_updated_at'], yearfirst=True)
```

```python id="v96VJHyFVws1" executionInfo={"status": "ok", "timestamp": 1628002953609, "user_tz": -330, "elapsed": 279453, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
def timediff(duration):
    duration_in_s = duration.total_seconds()
    days = divmod(duration_in_s, 86400)[0]
    return days

train['c_diff_update_create']=train['c_updated_at']-train['c_created_at']
train['v_diff_update_create']=train['v_updated_at']-train['v_created_at']
train['c_v_diff_create']=train['v_created_at']-train['c_created_at']
train['c_v_diff_update']=train['v_updated_at']-train['c_updated_at']

train['c_diff_update_create']=train['c_diff_update_create'].apply(timediff)
train['v_diff_update_create']=train['v_diff_update_create'].apply(timediff)
train['c_v_diff_create']=train['c_v_diff_create'].apply(timediff)
train['c_v_diff_update']=train['c_v_diff_update'].apply(timediff)

test['c_diff_update_create']=test['c_updated_at']-test['c_created_at']
test['v_diff_update_create']=test['v_updated_at']-test['v_created_at']
test['c_v_diff_create']=test['v_created_at']-test['c_created_at']
test['c_v_diff_update']=test['v_updated_at']-test['c_updated_at']

test['c_diff_update_create']=test['c_diff_update_create'].apply(timediff)
test['v_diff_update_create']=test['v_diff_update_create'].apply(timediff)
test['c_v_diff_create']=test['c_v_diff_create'].apply(timediff)
test['c_v_diff_update']=test['c_v_diff_update'].apply(timediff)
```

```python id="fNnpTBUMhWK3" executionInfo={"status": "ok", "timestamp": 1628002983665, "user_tz": -330, "elapsed": 10189, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
train['year_c_created_at'] = train['c_created_at'].dt.year
train['month_c_created_at'] = train['c_created_at'].dt.month
train['doy_c_created_at'] = train['c_created_at'].dt.dayofyear

train['year_c_updated_at'] = train['c_updated_at'].dt.year
train['month_c_updated_at'] = train['c_updated_at'].dt.month
train['doy_c_updated_at'] = train['c_updated_at'].dt.dayofyear

train['year_v_created_at'] = train['v_created_at'].dt.year
train['month_v_created_at'] = train['v_created_at'].dt.month
train['doy_v_created_at'] = train['v_created_at'].dt.dayofyear

train['year_v_updated_at'] = train['v_updated_at'].dt.year
train['month_v_updated_at'] = train['v_updated_at'].dt.month
train['doy_v_updated_at'] = train['v_updated_at'].dt.dayofyear

test['year_c_created_at'] = test['c_created_at'].dt.year
test['month_c_created_at'] = test['c_created_at'].dt.month
test['doy_c_created_at'] = test['c_created_at'].dt.dayofyear

test['year_c_updated_at'] = test['c_updated_at'].dt.year
test['month_c_updated_at'] = test['c_updated_at'].dt.month
test['doy_c_updated_at'] = test['c_updated_at'].dt.dayofyear

test['year_v_created_at'] = test['v_created_at'].dt.year
test['month_v_created_at'] = test['v_created_at'].dt.month
test['doy_v_created_at'] = test['v_created_at'].dt.dayofyear

test['year_v_updated_at'] = test['v_updated_at'].dt.year
test['month_v_updated_at'] = test['v_updated_at'].dt.month
test['doy_v_updated_at'] = test['v_updated_at'].dt.dayofyear
```

```python id="RDvaTRquhaLj" executionInfo={"status": "ok", "timestamp": 1628002997442, "user_tz": -330, "elapsed": 10467, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
train.drop(['c_created_at','c_updated_at','v_created_at','v_updated_at'], axis = 1, inplace=True)
test.drop(['c_created_at','c_updated_at','v_created_at','v_updated_at'], axis = 1, inplace=True)
```

```python id="9T2ff-h6iXZA" executionInfo={"status": "ok", "timestamp": 1628003375695, "user_tz": -330, "elapsed": 44556, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
train.to_parquet('./data/gold/train.parquet.gzip', compression='gzip')
test.to_parquet('./data/gold/test.parquet.gzip', compression='gzip')
```

```python id="_S-C8_kIjC0g"
!git add . && git commit -m 'commit' && git push origin main
```

```python id="zeJjzqByjNc6" executionInfo={"status": "ok", "timestamp": 1628003206836, "user_tz": -330, "elapsed": 590, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
def haversine_array(lat1, lng1, lat2, lng2):
    lat1, lng1, lat2, lng2 = map(np.radians, (lat1, lng1, lat2, lng2))
    AVG_EARTH_RADIUS = 6371  # in km
    lat = lat2 - lat1
    lng = lng2 - lng1
    d = np.sin(lat * 0.5) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(lng * 0.5) ** 2
    h = 2 * AVG_EARTH_RADIUS * np.arcsin(np.sqrt(d))
    return h

def dummy_manhattan_distance(lat1, lng1, lat2, lng2):
    a = haversine_array(lat1, lng1, lat1, lng2)
    b = haversine_array(lat1, lng1, lat2, lng1)
    return a + b

def bearing_array(lat1, lng1, lat2, lng2):
    AVG_EARTH_RADIUS = 6371  # in km
    lng_delta_rad = np.radians(lng2 - lng1)
    lat1, lng1, lat2, lng2 = map(np.radians, (lat1, lng1, lat2, lng2))
    y = np.sin(lng_delta_rad) * np.cos(lat2)
    x = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(lng_delta_rad)
    return np.degrees(np.arctan2(y, x))

train['center_latitude'] = (train['c_latitude'].values + train['v_latitude'].values) / 2
train['center_longitude'] = (train['c_longitude'].values + train['v_longitude'].values) / 2
train['harvesine_dist']=haversine_array(train['c_latitude'], train['c_longitude'], train['v_latitude'], train['v_longitude'])
train['manhattan_dist']=dummy_manhattan_distance(train['c_latitude'], train['c_longitude'], train['v_latitude'], train['v_longitude'])
train['bearing']=bearing_array(train['c_latitude'], train['c_longitude'], train['v_latitude'], train['v_longitude'])

test['center_latitude'] = (test['c_latitude'].values + test['v_latitude'].values) / 2
test['center_longitude'] = (test['c_longitude'].values + test['v_longitude'].values) / 2
test['harvesine_dist']=haversine_array(test['c_latitude'], test['c_longitude'], test['v_latitude'], test['v_longitude'])
test['manhattan_dist']=dummy_manhattan_distance(test['c_latitude'], test['c_longitude'], test['v_latitude'], test['v_longitude'])
test['bearing']=bearing_array(test['c_latitude'], test['c_longitude'], test['v_latitude'], test['v_longitude'])
```

```python colab={"base_uri": "https://localhost:8080/", "height": 224} id="YTNGUZfvixny" executionInfo={"status": "ok", "timestamp": 1628003254825, "user_tz": -330, "elapsed": 615, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="9f5f64dd-6ba2-457c-9be3-2eedd1d7e5aa"
train.head()
```

```python colab={"base_uri": "https://localhost:8080/", "height": 224} id="OhkARCCdizRT" executionInfo={"status": "ok", "timestamp": 1628003262481, "user_tz": -330, "elapsed": 819, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="a8868269-36ff-4d46-ffc8-84065ceab882"
test.head()
```

```python colab={"base_uri": "https://localhost:8080/"} id="KgMgXgIBnLPt" executionInfo={"status": "ok", "timestamp": 1628004206694, "user_tz": -330, "elapsed": 567, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="83d438e5-3a4d-4f8c-bd1f-8a63884d72c9"
!git status
```

```python colab={"base_uri": "https://localhost:8080/"} id="hRx3KbrFnL6O" executionInfo={"status": "ok", "timestamp": 1628004242462, "user_tz": -330, "elapsed": 3222, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="5f31f153-75c6-4ef9-f60f-085606ba1113"
!git add . && git commit -m 'Add data layer gold' && git push origin main
```
