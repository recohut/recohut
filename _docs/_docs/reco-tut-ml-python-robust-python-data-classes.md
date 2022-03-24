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

<!-- #region id="3VGMOau_luZF" -->
## Data Classes
Data classes represent a heterogeneous collection of variables, all rolled into a composite type. Composite types are made up of multiple values, and should always represent some sort of relationship or logical grouping. For example, a Fraction is an excellent example of a composite type. It contains two scalar values: a numerator and a denominator.
<!-- #endregion -->

```python id="cm_Ko0BhqGsk" executionInfo={"status": "ok", "timestamp": 1630756660966, "user_tz": -330, "elapsed": 525, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
from dataclasses import dataclass

@dataclass
class MyFraction:
    numerator: int = 0
    denominator: int = 1
```

<!-- #region id="Q58pNkA0q-Im" -->
By building relationships like this, you are adding to the shared vocabulary in your codebase. Instead of developers always needing to implement each field individually, you instead provide a reusable grouping. Data classes force you to explicitly assign types to your fields, so there’s less chance of type confusion among maintainers.

Data classes and other user-defined types can be nested within the dataclass. Suppose I’m creating an automated soup maker and I need to group my soup ingredients together. Using dataclass, it looks like this:
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="VH0Wd2ipsHvb" executionInfo={"status": "ok", "timestamp": 1630756967232, "user_tz": -330, "elapsed": 1066, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="c630f137-2da0-4e6d-eb0b-fff1bfb3431d"
!python --version
```

<!-- #region id="VCxOkXTasMgP" -->
> Warning: Following code do not work in python version < 3.8
<!-- #endregion -->

```python id="mtIJ1raiq_Q1"
import datetime
from dataclasses import dataclass
from enum import auto, Enum

class ImperialMeasure(Enum):
    TEASPOON = auto()
    TABLESPOON = auto()
    CUP = auto()

class Broth(Enum):
    VEGETABLE = auto()
    CHICKEN = auto()
    BEEF = auto()
    FISH = auto()

@dataclass(frozen=True)
# Ingredients added into the broth
class Ingredient:
    name: str
    amount: float = 1
    units: ImperialMeasure = ImperialMeasure.CUP

@dataclass
class Recipe:
    aromatics: set[Ingredient]
    broth: Broth
    vegetables: set[Ingredient]
    meats: set[Ingredient]
    starches: set[Ingredient]
    garnishes: set[Ingredient]
    time_to_cook: datetime.timedelta
```

<!-- #region id="Rtj4FRI5rnSH" -->
- A soup recipe is a set of grouped information. Specifically, it can be defined by its ingredients (separated into specific categories), the broth used, and how long it takes to cook.
- Each ingredient has a name and an amount you need for the recipe.
- You have enumerations to tell you about the soup broth and measures. These are not a relationship by themselves, but they do communicate intention to the reader.
- Each grouping of ingredients is a set, rather than a tuple. This means that the user can change these after construction, but still prevent duplicates.
<!-- #endregion -->

<!-- #region id="03K3UPiHrqj3" -->
To create the dataclass, I do the following:
<!-- #endregion -->

```python id="-cUGxj6BsUdx"
pepper = Ingredient("Pepper", 1, ImperialMeasure.TABLESPOON)
garlic = Ingredient("Garlic", 2, ImperialMeasure.TEASPOON)
carrots = Ingredient("Carrots", .25, ImperialMeasure.CUP)
celery = Ingredient("Celery", .25, ImperialMeasure.CUP)
onions = Ingredient("Onions", .25, ImperialMeasure.CUP)
parsley = Ingredient("Parsley", 2, ImperialMeasure.TABLESPOON)
noodles = Ingredient("Noodles", 1.5, ImperialMeasure.CUP)
chicken = Ingredient("Chicken", 1.5, ImperialMeasure.CUP)

chicken_noodle_soup = Recipe(
    aromatics={pepper, garlic},
    broth=Broth.CHICKEN,
    vegetables={celery, onions, carrots},
    meats={chicken},
    starches={noodles},
    garnishes={parsley},
    time_to_cook=datetime.timedelta(minutes=60))
```

<!-- #region id="JoLyshFzsfoj" -->
<!-- #endregion -->

```python id="tYKHqRdxuKxX"

```
