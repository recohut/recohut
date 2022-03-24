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

<!-- #region id="017CL0EQ8ReN" -->
You can think of the data model as a description of Python as a framework. It formalizes the interfaces of the building blocks of the language itself, such as sequences, functions, iterators, coroutines, classes, context managers, and so on.

When using a framework, we spend a lot of time coding methods that are called by the framework. The same happens when we leverage the Python Data Model to build new classes. The Python interpreter invokes special methods to perform basic object operations, often triggered by special syntax. The special method names are always written with leading and trailing double underscores. For example, the syntax `obj[key]` is supported by the `__getitem__` special method.
<!-- #endregion -->

<!-- #region id="USTBJenC8c-x" -->
> **MAGIC AND DUNDER**\
> The term magic method is slang for special method, but how do we talk about a specific method like __getitem__? I learned to say “dunder-getitem” from author and teacher Steve Holden. “Dunder” is a shortcut for “double underscore before and after”. That’s why the special methods are also known as dunder methods. 
<!-- #endregion -->

```python id="sULmwgFL6Y2i" executionInfo={"status": "ok", "timestamp": 1627760932924, "user_tz": -330, "elapsed": 385, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
import collections

Card = collections.namedtuple('Card', ['rank', 'suit'])

class FrenchDeck:
    ranks = [str(n) for n in range(2, 11)] + list('JQKA')
    suits = 'spades diamonds clubs hearts'.split()

    def __init__(self):
        self._cards = [Card(rank, suit) for suit in self.suits
                                        for rank in self.ranks]

    def __repr__(self):
        return f'FrenchDeck(class)'

    def __len__(self):
        return len(self._cards)

    def __getitem__(self, position):
        return self._cards[position]
```

<!-- #region id="Z7LJhaao6m9W" -->
The first thing to note is the use of `collections.namedtuple` to construct a simple class to represent individual cards. We use `namedtuple` to build classes of objects that are just bundles of attributes with no custom methods, like a database record. In the example, we use it to provide a nice representation for the cards in the deck, as shown in the console session:
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="1K6YK4MP7JGs" executionInfo={"status": "ok", "timestamp": 1627757803445, "user_tz": -330, "elapsed": 562, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="f267feda-a64f-471d-8379-9bae88d5032a"
beer_card = Card('7', 'diamonds')
beer_card
```

```python colab={"base_uri": "https://localhost:8080/"} id="SawrlEmNGmrI" executionInfo={"status": "ok", "timestamp": 1627760940238, "user_tz": -330, "elapsed": 387, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="670058bb-5bf6-4d40-cfd3-ceb5538dea44"
FrenchDeck()
```

<!-- #region id="eqh9TjRz7SCe" -->
But the point of this example is the `FrenchDeck` class. It’s short, but it packs a punch.
1. First, like any standard Python collection, a deck responds to the `len()` function by returning the number of cards in it.
2. Reading specific cards from the deck—say, the first or the last—is easy, thanks to the `__getitem__` method.
3. How to pick a random card? Should we create a method inside `FrenchDeck` card class? No need. Python already has a function to get a random item from a sequence: `random.choice`. We can use it on a deck instance. It’s easier to benefit from the rich Python standard library and avoid reinventing the wheel, like the `random.choice` function.
4. But it gets better. Because our `__getitem__` delegates to the [] operator of `self._cards`, our deck automatically supports slicing. Here’s how we look at the top three cards from a brand-new deck, and then pick just the Aces by starting at index 12 and skipping 13 cards at a time.
5. Just by implementing the `__getitem__` special method, our deck is also iterable.
6. Iteration is often implicit. If a collection has no `__contains__` method, the `in` operator does a sequential scan. Case in point: `in` works with our FrenchDeck class because it is iterable.
7. How to sort the cards by the given ranking notion. We can sort by passing the method of ranking notion.
8. How about shuffling? As implemented so far, a `FrenchDec`k cannot be shuffled, because it is immutable: the cards, and their positions cannot be changed, except by violating encapsulation and handling the `_cards` attribute directly.
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="y-qheXPk7kvZ" executionInfo={"status": "ok", "timestamp": 1627759736609, "user_tz": -330, "elapsed": 521, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="d805bf8f-5871-4577-b937-097eddf93caf"
# we use len(deck) instead of deck.__len__()
deck = FrenchDeck()
print("\n{}\n".format("="*50))
print(len(deck))

# we use deck(idx) instead of deck.__getitem__(idx)
print("\n{}\n".format("="*50))
print(deck[0])
print(deck[-1])

# picking some random cards
from random import choice
print("\n{}\n".format("="*50))
print(choice(deck))
print(choice(deck))
print(choice(deck))

# picking cards with slicing
print("\n{}\n".format("="*50))
print(deck[:3])
print(deck[12::13])

# iterating
print("\n{}\n".format("="*50))
for card in deck[:5]:
    print(card)

# checking if a card is in deck
print("\n{}\n".format("="*50))
print(Card('Q', 'hearts') in deck)
print(Card('7', 'beasts') in deck)
```

<!-- #region id="RgEveUC-91Iy" -->
A common system of ranking cards is by rank (with aces being highest), then by suit in the order of spades (highest), hearts, diamonds, and clubs (lowest). Here is a function that ranks cards by that rule, returning 0 for the 2 of clubs and 51 for the ace of spades:
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="yNDFGmSfDn9c" executionInfo={"status": "ok", "timestamp": 1627760028021, "user_tz": -330, "elapsed": 437, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="2efaa190-1df0-4f6f-8695-77784947412a"
suit_values = dict(spades=3, hearts=2, diamonds=1, clubs=0)

def spades_high(card):
    rank_value = FrenchDeck.ranks.index(card.rank)
    return rank_value * len(suit_values) + suit_values[card.suit]

# Given spades_high, we can now list our deck in order of increasing rank
for card in sorted(deck, key=spades_high):
    print(card)
```

```python id="-lQ9rsZrDuTJ"

```
