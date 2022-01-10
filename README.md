# recohut



<div id="top"></div>

[![Contributors][contributors-shield]][contributors-url]
[![Forks][forks-shield]][forks-url]
[![Stargazers][stars-shield]][stars-url]
[![Issues][issues-shield]][issues-url]
[![MIT License][license-shield]][license-url]



<!-- PROJECT LOGO -->
<br />
<div align="center">
  <a href="https://github.com/RecoHut-Projects/recohut">
    <img src="https://github.com/recohut/reco-static/raw/master/media/diagrams/recohut_logo.svg" alt="Logo" width="80" height="80">
  </a>

<!-- <h3 align="center">recohut</h3> -->

  <p align="center">
    a python library for building recommender systems.
    <br />
    <a href="https://recohut-projects.github.io/recohut"><strong>Explore the docs »</strong></a>
    <br />
    <br />
    <a href="https://github.com/RecoHut-Projects/recohut/tree/master/tutorials">View Demo</a>
    ·
    <a href="https://github.com/RecoHut-Projects/recohut/issues">Report Bug</a>
    ·
    <a href="https://github.com/RecoHut-Projects/recohut/issues">Request Feature</a>
  </p>
</div>



## About The Project

<img src="https://github.com/recohut/reco-static/raw/master/media/diagrams/recohut_lib_main.svg">


<p align="right">(<a href="#top">back to top</a>)</p>



### Built With

* [Python](https://www.python.org/)
* [PyTorch](https://pytorch.org/)
* [Lightning](https://www.pytorchlightning.ai/)
* [nbdev](https://github.com/fastai/nbdev)

<p align="right">(<a href="#top">back to top</a>)</p>



<!-- GETTING STARTED -->
## Getting Started

To get a local copy up and running follow these simple example steps.

### Prerequisites

* pytorch
  ```sh
  pip install torch
  ```
* lightning
  ```sh
  pip install pytorch-lightning
  ```

### Installation

```
pip install recohut
```

<p align="right">(<a href="#top">back to top</a>)</p>



<!-- USAGE EXAMPLES -->
## Usage

```python
# import the required modules
from recohut.datasets.movielens import ML1mDataModule
from recohut.models.nmf import NMF
from recohut.trainers.pl_trainer import pl_trainer

# build the dataset
class Args:
    def __init__(self):
        self.data_dir = '/content/data'
        self.min_rating = 4
        self.num_negative_samples = 99
        self.min_uc = 5
        self.min_sc = 5
        self.val_p = 0.2
        self.test_p = 0.2
        self.num_workers = 2
        self.normalize = False
        self.batch_size = 32
        self.seed = 42
        self.shuffle = True
        self.pin_memory = True
        self.drop_last = False
        self.split_type = 'stratified'

args = Args()

ds = ML1mDataModule(**args.__dict__)
ds.prepare_data()

# build the model
model = NMF(n_items=ds.data.num_items, n_users=ds.data.num_users, embedding_dim=20)

# train and evaluate the matrix factorization model
pl_trainer(model, ds, max_epochs=5)
```
Check [this](https://github.com/recohut/notebooks/blob/main/nbs/recohut_quick_tutorial.ipynb) quick tutorial.

_For more examples, please refer to the [Documentation](https://recohut-projects.github.io/recohut) and [Tutorials](https://github.com/RecoHut-Projects/recohut/tree/master/tutorials)._

<p align="right">(<a href="#top">back to top</a>)</p>



<!-- ROADMAP -->
## Roadmap

- [] RecSys Model Deployment and MLOps features
- [] RL agents and environment specific to recommender systems
- [] Visualization utilities and EDA

See the [open issues](https://github.com/RecoHut-Projects/recohut/issues) for a full list of proposed features (and known issues).

<p align="right">(<a href="#top">back to top</a>)</p>



<!-- CONTRIBUTING -->
## Contributing

Contributions are what make the open source community such an amazing place to learn, inspire, and create. Any contributions you make are **greatly appreciated**.

If you have a suggestion that would make this better, please fork the repo and create a pull request. You can also simply open an issue with the tag "enhancement".
Don't forget to give the project a star! Thanks again!

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

<p align="right">(<a href="#top">back to top</a>)</p>



<!-- LICENSE -->
## License

Distributed under the MIT License. See `LICENSE.txt` for more information.

<p align="right">(<a href="#top">back to top</a>)</p>



<!-- CONTACT -->
## Contact

Sparsh A.

[@sparsh-ai](https://github.com/RecoHut-Projects/recohut)

<p align="right">(<a href="#top">back to top</a>)</p>



<!-- ACKNOWLEDGMENTS -->
## Acknowledgments

* [nbdev team](https://nbdev.fast.ai/tutorial.html) for providing supporting tools to build this library.
* [colab team](https://colab.research.google.com/) for providing running VMs instances for development and testing.

<p align="right">(<a href="#top">back to top</a>)</p>



<!-- MARKDOWN LINKS & IMAGES -->
[contributors-shield]: https://img.shields.io/github/contributors/RecoHut-Projects/recohut.svg?style=for-the-badge
[contributors-url]: https://github.com/RecoHut-Projects/recohut/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/RecoHut-Projects/recohut.svg?style=for-the-badge
[forks-url]: https://github.com/RecoHut-Projects/recohut/network/members
[stars-shield]: https://img.shields.io/github/stars/RecoHut-Projects/recohut.svg?style=for-the-badge
[stars-url]: https://github.com/RecoHut-Projects/recohut/stargazers
[issues-shield]: https://img.shields.io/github/issues/RecoHut-Projects/recohut.svg?style=for-the-badge
[issues-url]: https://github.com/RecoHut-Projects/recohut/issues
[license-shield]: https://img.shields.io/github/license/RecoHut-Projects/recohut.svg?style=for-the-badge
[license-url]: https://github.com/RecoHut-Projects/recohut/blob/master/LICENSE.txt
[product-screenshot]: https://github.com/recohut/reco-static/raw/master/media/diagrams/recohut_lib_main.svg
