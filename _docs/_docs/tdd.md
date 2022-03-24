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

<!-- #region id="EPR0wLGqyhHS" -->
# Test-Driven Development (TDD)
<!-- #endregion -->

<!-- #region id="QxeE8Kne7ghG" -->
Test-driven development follows a three-phase process. The three phases are:

1. **RED**. We write a failing test (including possibly compilation failures). We run the test suite to verify the failing tests.
2. **GREEN**. We write just enough production code to make the test green. We run the test suite the verify this.
3. **REFACTOR**. We remove any code-smells. These may be due to duplication, hard-coded values, or improper use of language idioms (e.g. using a verbose loop instead of a built-in iterator). If we break any tests during refactoring, we prioritize getting them back to green before exiting this phase.
<!-- #endregion -->

<!-- #region id="FufeIaSr7qgU" -->
<!-- #endregion -->

<!-- #region id="oUH9NAr87k3Y" -->
## Money Problem
<!-- #endregion -->

<!-- #region id="5JBSOdbE2Y_w" -->
Suppose we want to manage money (let's say a portfolio of different shares) like the following:
<!-- #endregion -->

<!-- #region id="-cjyEsw43LTn" -->
<!-- #endregion -->

<!-- #region id="Up3SiW0B3Ajd" -->
We will make sure that arithmatic is correct. So we will design tests:

- 5 USD x 2 = 10 USD
- 10 EUR x 2 = 20 EUR
- 4002 KRW / 4 = 1000.5 KRW
- 5 USD + 10 EUR = 17 USD
- 1 USD + 1100 KRW = 2200 KRW
<!-- #endregion -->

<!-- #region id="--S6f2vK730v" -->
## 5 USD x 2 = 10 USD
<!-- #endregion -->

<!-- #region id="JBP1v-OO4ejx" -->
### Starting with RED
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="08KNd5eV3XYw" executionInfo={"status": "ok", "timestamp": 1630139342355, "user_tz": -330, "elapsed": 425, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="8c8cf525-e5b4-46cd-c9b7-1a6cde5717da"
%%writefile test_money.py
import unittest

class TestMoney(unittest.TestCase):
  def testMultiplication(self):
    fiver = Dollar(5)
    tenner = fiver.times(2)
    self.assertEqual(10, tenner.amount)

if __name__ == '__main__':
    unittest.main()
```

```python colab={"base_uri": "https://localhost:8080/"} id="8JGtzB_e4FnL" executionInfo={"status": "ok", "timestamp": 1630139401397, "user_tz": -330, "elapsed": 492, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="e7b21056-1f95-4a35-b37c-c78c29da7f64"
!python test_money.py -v
```

<!-- #region id="05AYEHoV4OPq" -->
That’s our first failing test. Hooray!
<!-- #endregion -->

<!-- #region id="XIV8DPcG4YJl" -->
### Going for GREEN
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="FwT6uu2J44Au" executionInfo={"status": "ok", "timestamp": 1630139572312, "user_tz": -330, "elapsed": 942, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="fd3482be-b52f-406e-b475-eb35591b5480"
%%writefile test_money.py
import unittest

class Dollar:
  pass

class TestMoney(unittest.TestCase):
  def testMultiplication(self):
    fiver = Dollar(5)
    tenner = fiver.times(2)
    self.assertEqual(10, tenner.amount)

if __name__ == '__main__':
    unittest.main()
```

```python colab={"base_uri": "https://localhost:8080/"} id="wmOD_XdJ44tc" executionInfo={"status": "ok", "timestamp": 1630139572926, "user_tz": -330, "elapsed": 18, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="e4b6952d-49a8-4309-83c9-a1eb514b765c"
!python test_money.py -v
```

```python colab={"base_uri": "https://localhost:8080/"} id="6dXEoYJd5Aeb" executionInfo={"status": "ok", "timestamp": 1630139611770, "user_tz": -330, "elapsed": 425, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="529b43b6-ad6d-409e-ffd5-cdd8f5d6dc0c"
%%writefile test_money.py
import unittest

class Dollar:
  def __init__(self, amount):
    pass

class TestMoney(unittest.TestCase):
  def testMultiplication(self):
    fiver = Dollar(5)
    tenner = fiver.times(2)
    self.assertEqual(10, tenner.amount)

if __name__ == '__main__':
    unittest.main()
```

```python colab={"base_uri": "https://localhost:8080/"} id="Sc5FQPPc5Aee" executionInfo={"status": "ok", "timestamp": 1630139613080, "user_tz": -330, "elapsed": 733, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="1638e855-dbd8-4f9e-f4f8-1c55e30e6447"
!python test_money.py -v
```

<!-- #region id="Gkg73puK4j1E" -->
We see a pattern here: our test is still failing, but for slightly different reasons each time. As we define our abstractions — first Dollar and then an amount field — the error messages “improve” to the next stage. This is a hallmark of TDD: steady progress at a pace we control.

Let’s speed things up a bit by defining a times function and giving it the minimum behavior to get to green. What’s the minimum behavior necessary? Always returning a “ten dollar” object that’s required by our test, of course!
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="0D-MnEZC5e0t" executionInfo={"status": "ok", "timestamp": 1630139795036, "user_tz": -330, "elapsed": 515, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="98f3611d-32c5-4190-a287-e9ed230a663d"
%%writefile test_money.py
import unittest


class Dollar:
  def __init__(self, amount):
    self.amount = amount

  def times(self, multiplier):
    return Dollar(10)


class TestMoney(unittest.TestCase):
  def testMultiplication(self):
    fiver = Dollar(5)
    tenner = fiver.times(2)
    self.assertEqual(10, tenner.amount)

if __name__ == '__main__':
    unittest.main()
```

```python colab={"base_uri": "https://localhost:8080/"} id="VUnEKIpE5e0u" executionInfo={"status": "ok", "timestamp": 1630139795471, "user_tz": -330, "elapsed": 12, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="98188e28-cc91-4bd3-d3ed-d63fd3054861"
!python test_money.py -v
```

<!-- #region id="azyxm8Nv50DQ" -->
### Cleaning up
Refactoring is the third and final stage of the RGR cycle. We may not have many lines of code at this point, however, it’s still important to keep things tidy and compact. If we have any formatting clutter or commented out lines of code, now is the time to clean it up.

More significant is the need to remove duplication and make code readable. At first blush, it may seem that in our fewer than two dozen lines of code, there can’t be any duplication. However, there is already a subtle yet significant bit of duplication.
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="p4fe4pF_6iBo" executionInfo={"status": "ok", "timestamp": 1630140021073, "user_tz": -330, "elapsed": 740, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="9fcc9ce6-9324-4c0c-864b-d2a29b119419"
%%writefile test_money.py
import unittest


class Dollar:
  def __init__(self, amount):
    self.amount = amount

  def times(self, multiplier):
    return Dollar(5 * 2)


class TestMoney(unittest.TestCase):
  def testMultiplication(self):
    fiver = Dollar(5)
    tenner = fiver.times(2)
    self.assertEqual(10, tenner.amount)

if __name__ == '__main__':
    unittest.main()
```

```python colab={"base_uri": "https://localhost:8080/"} id="5ICUmKEj6iBq" executionInfo={"status": "ok", "timestamp": 1630140021075, "user_tz": -330, "elapsed": 20, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="d11cea8e-21ed-4269-ed48-ae6b6069fb5b"
!python test_money.py -v
```

```python colab={"base_uri": "https://localhost:8080/"} id="NkwI3KUW6sw9" executionInfo={"status": "ok", "timestamp": 1630140059490, "user_tz": -330, "elapsed": 756, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="407a1a40-f94e-427a-8200-28167de72846"
%%writefile test_money.py
import unittest


class Dollar:
  def __init__(self, amount):
    self.amount = amount

  def times(self, multiplier):
    return Dollar(self.amount * multiplier)


class TestMoney(unittest.TestCase):
  def testMultiplication(self):
    fiver = Dollar(5)
    tenner = fiver.times(2)
    self.assertEqual(10, tenner.amount)

if __name__ == '__main__':
    unittest.main()
```

```python colab={"base_uri": "https://localhost:8080/"} id="vhcQMLrl6sw_" executionInfo={"status": "ok", "timestamp": 1630140059493, "user_tz": -330, "elapsed": 22, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="f8ff3a0c-fa65-4f5a-af7a-93811a429bb2"
!python test_money.py -v
```

<!-- #region id="cLZ-maxy6rLg" -->
Hooray! The test remains green, and the duplication and the coupling are gone.
<!-- #endregion -->

<!-- #region id="Ev8zyoOW63k6" -->
The goal of test-driven development isn’t to force us to go slow. Or fast, for that matter. Its goal is to allow us to go at a pace we’re comfortable with: speeding up when we can, slowing down when we should.
<!-- #endregion -->

<!-- #region id="JzvUuwcu76d2" -->
## 10 EUR x 2 = 20 EUR
<!-- #endregion -->

<!-- #region id="QLdN5Gnj77HN" -->
This indicates that we need a more general entity than the Dollar. Something like Money, which encapsulates an amount (which we already have) and a currency (which we do not yet have). Let’s write tests to flush out this new feature.
<!-- #endregion -->

<!-- #region id="mHi-GYT68GMU" -->
Let’s add a new test in the TestMoney class. This test would verify that multiplying an object representing “10 Euros” by 2 gives us an object representing “20 Euros”:
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="fk6TMlHS8dbi" executionInfo={"status": "ok", "timestamp": 1630140701341, "user_tz": -330, "elapsed": 520, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="9b882898-bf87-46de-8c83-f646d1405e66"
%%writefile test_money.py
import unittest


class Dollar:
    def __init__(self, amount):
      self.amount = amount

    def times(self, multiplier):
        return Dollar(self.amount * multiplier)


class Money:
    def __init__(self, amount, currency):
        self.amount = amount
        self.currency = currency
        
    def times(self, multiplier):
        return Money(self.amount * multiplier, self.currency)


class TestMoney(unittest.TestCase):
    def testMultiplication(self):
        fiver = Dollar(5)
        tenner = fiver.times(2)
        self.assertEqual(10, tenner.amount)
    
    def testMultiplicationInEuros(self):
        tenEuros = Money(10, "EUR")
        twentyEuros = tenEuros.times(2)
        self.assertEqual(20, twentyEuros.amount)
        self.assertEqual("EUR", twentyEuros.currency)


if __name__ == '__main__':
    unittest.main()
```

```python colab={"base_uri": "https://localhost:8080/"} id="NdYdKGZz8dbk" executionInfo={"status": "ok", "timestamp": 1630140701776, "user_tz": -330, "elapsed": 10, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="5272d190-7085-4831-d974-b8d8fc7ee667"
!python test_money.py -v
```

<!-- #region id="XUO3Q0239RjJ" -->
Wait a minute: didn’t we just create a horrendous duplication in our code? The new entity we created to represent Money subsumes what we wrote earlier for Dollar. This can’t possibly be good. A oft-quoted rule in writing code is the DRY principle: Don’t Repeat Yourself.

Recall the RED-GREEN-REFACTOR cycle. What we did got us to green, but we haven’t done the necessary refactoring yet. Let’s remove the duplication in the code while keeping our tests green.
<!-- #endregion -->

<!-- #region id="Hz4wiL7j9nsZ" -->
The Money class’s functionality is a superset of that of the Dollar class. Which means we don’t need the latter. Let’s delete the Dollar class in its entirety.

Having done this, we get the familiar NameError: name 'Dollar' is not defined message when we run the tests. Let’s refactor the first test to use Money instead of the erstwhile Dollar:
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="I2EOWWYq9qMv" executionInfo={"status": "ok", "timestamp": 1630140876241, "user_tz": -330, "elapsed": 516, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="fc6e6e44-1097-410e-8d5b-b39376146af7"
%%writefile test_money.py
import unittest


class Money:
    def __init__(self, amount, currency):
        self.amount = amount
        self.currency = currency
        
    def times(self, multiplier):
        return Money(self.amount * multiplier, self.currency)


class TestMoney(unittest.TestCase):
    def testMultiplicationInDollars(self):
        fiver = Money(5, "USD")
        tenner = fiver.times(2)
        self.assertEqual(10, tenner.amount)
        self.assertEqual("USD", tenner.currency)
    
    def testMultiplicationInEuros(self):
        tenEuros = Money(10, "EUR")
        twentyEuros = tenEuros.times(2)
        self.assertEqual(20, twentyEuros.amount)
        self.assertEqual("EUR", twentyEuros.currency)


if __name__ == '__main__':
    unittest.main()
```

```python colab={"base_uri": "https://localhost:8080/"} id="S53AYAdo9qMx" executionInfo={"status": "ok", "timestamp": 1630140876940, "user_tz": -330, "elapsed": 14, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="19c2a2c1-7bd4-409e-fa72-452625c274d6"
!python test_money.py -v
```

<!-- #region id="Q-DC882w98TS" -->
> Note: We could delete one of the tests and still feel confident about our code. However, we also want to safeguard ourselves against accidental regression in our code. Recall that our very first implementation used hard-coded numbers (10 or 5 * 2). Having two distinct tests with different values ensures that we won’t accidentally go back to that naive implementation.
<!-- #endregion -->

<!-- #region id="6j8E0hIj-aZ3" -->
> Tip: Regression — “a return to a primitive or less developed state" — is a common theme in writing software. Having a battery of tests is a reliable way to ensure that we don’t break existing features as we build new ones.
<!-- #endregion -->

<!-- #region id="GJnylA1E-gmv" -->
## 4002 KRW / 4 = 1000.5 KRW
<!-- #endregion -->

<!-- #region id="2Ojdpd2F-wyG" -->
The next requirement is to allow division. On the surface, it looks very similar to multiplication. We know from elementary mathematics that dividing by x is the same as multiplying by 1⁠/⁠x.

Let’s test-drive this new feature and see how our code evolves. By now, we are getting into the groove of starting with a failing test. As an indicator of our growing confidence, we’ll introduce two new things in our test:

- A new currency: Korean Won (KRW), and
- Numbers with fractional parts, e.g. 1000.5
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="hpytrLT0-4XR" executionInfo={"status": "ok", "timestamp": 1630141308820, "user_tz": -330, "elapsed": 831, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="8d24239f-c405-47f3-cad7-13de4ea9b53b"
%%writefile test_money.py
import unittest


class Money:
    def __init__(self, amount, currency):
        self.amount = amount
        self.currency = currency
        
    def times(self, multiplier):
        return Money(self.amount * multiplier, self.currency)

    def divide(self, divisor):
        return Money(self.amount / divisor, self.currency)


class TestMoney(unittest.TestCase):
    def testMultiplicationInDollars(self):
        fiver = Money(5, "USD")
        tenner = fiver.times(2)
        self.assertEqual(10, tenner.amount)
        self.assertEqual("USD", tenner.currency)
    
    def testMultiplicationInEuros(self):
        tenEuros = Money(10, "EUR")
        twentyEuros = tenEuros.times(2)
        self.assertEqual(20, twentyEuros.amount)
        self.assertEqual("EUR", twentyEuros.currency)

    def testDivision(self):
        originalMoney = Money(4002, "KRW")
        actualMoneyAfterDivision = originalMoney.divide(4)
        expectedMoneyAfterDivision = Money(1000.5, "KRW")
        self.assertEqual(expectedMoneyAfterDivision.amount,
                        actualMoneyAfterDivision.amount)
        self.assertEqual(expectedMoneyAfterDivision.currency,
                        actualMoneyAfterDivision.currency)


if __name__ == '__main__':
    unittest.main()
```

```python colab={"base_uri": "https://localhost:8080/"} id="TE0WUlsO-4XT" executionInfo={"status": "ok", "timestamp": 1630141308822, "user_tz": -330, "elapsed": 19, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="d2c3ed75-514d-4efb-a2e6-41aee33a9b36"
!python test_money.py -v
```

<!-- #region id="Mgp4KNYF_luA" -->
Yay! The tests are green. Python is a dynamically (and strongly) typed language. This makes implementing this feature easier than languages with static typing.
<!-- #endregion -->

<!-- #region id="v7uGmh80_uAr" -->
### Cleaning
<!-- #endregion -->

<!-- #region id="qnWooXY0_u2M" -->
Comparing two Money objects piecemeal is verbose and tedious. In our tests, we verify that the amount and currency fields of Money objects are equal, over and over. Wouldn’t it be nice to be able to compare two Money objects directly in a single line of code?

In Python, object equality is ultimately resolved by an invocation of the __eq__ method. By default, this method returns true if the two object references being compared in fact point to the same object. This is a very strict definition of equality: it means that an object is only equal to itself, not any other object, even of the two objects have the same state.

Fortunately, it is not only possible but recommended to override the __eq__ method when needed. Let us explicitly override this method within the definition of our Money class:
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="Ga30ahrkAWbp" executionInfo={"status": "ok", "timestamp": 1630141676870, "user_tz": -330, "elapsed": 835, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="1edee9f8-acd5-4d7d-eb3f-662f7824b925"
%%writefile test_money.py
import unittest


class Money:
    def __init__(self, amount, currency):
        self.amount = amount
        self.currency = currency
        
    def times(self, multiplier):
        return Money(self.amount * multiplier, self.currency)

    def divide(self, divisor):
        return Money(self.amount / divisor, self.currency)

    def __eq__(self, other):
        return self.amount == other.amount and self.currency == other.currency


class TestMoney(unittest.TestCase):
    def testMultiplicationInDollars(self):
        fiveDollars = Money(5, "USD")
        tenDollars = Money(10, "USD")
        self.assertEqual(tenDollars, fiveDollars.times(2))
    
    def testMultiplicationInEuros(self):
        tenEuros = Money(10, "EUR")
        twentyEuros = tenEuros.times(2)
        self.assertEqual(20, twentyEuros.amount)
        self.assertEqual("EUR", twentyEuros.currency)

    def testDivision(self):
        originalMoney = Money(4002, "KRW")
        actualMoneyAfterDivision = originalMoney.divide(4)
        expectedMoneyAfterDivision = Money(1000.5, "KRW")
        self.assertEqual(expectedMoneyAfterDivision.amount,
                        actualMoneyAfterDivision.amount)
        self.assertEqual(expectedMoneyAfterDivision.currency,
                        actualMoneyAfterDivision.currency)


if __name__ == '__main__':
    unittest.main()
```

```python colab={"base_uri": "https://localhost:8080/"} id="MCmH4iMlAWbr" executionInfo={"status": "ok", "timestamp": 1630141676872, "user_tz": -330, "elapsed": 24, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="64b38040-6af9-4fd5-e3ea-44446129ba6e"
!python test_money.py -v
```

```python colab={"base_uri": "https://localhost:8080/"} id="z9SqAMKYBHSu" executionInfo={"status": "ok", "timestamp": 1630141749047, "user_tz": -330, "elapsed": 518, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="71313c4b-f821-45c5-c4a7-a7ddbb293e5e"
%%writefile test_money.py
import unittest


class Money:
    def __init__(self, amount, currency):
        self.amount = amount
        self.currency = currency
        
    def times(self, multiplier):
        return Money(self.amount * multiplier, self.currency)

    def divide(self, divisor):
        return Money(self.amount / divisor, self.currency)

    def __eq__(self, other):
        return self.amount == other.amount and self.currency == other.currency


class TestMoney(unittest.TestCase):
    def testMultiplicationInDollars(self):
        fiveDollars = Money(5, "USD")
        tenDollars = Money(10, "USD")
        self.assertEqual(tenDollars, fiveDollars.times(2))
    
    def testMultiplicationInEuros(self):
        tenEuros = Money(10, "EUR")
        twentyEuros = Money(20, "EUR")
        self.assertEqual(twentyEuros, tenEuros.times(2))

    def testDivision(self):
        originalMoney = Money(4002, "KRW")
        expectedMoneyAfterDivision = Money(1000.5, "KRW")
        self.assertEqual(expectedMoneyAfterDivision, originalMoney.divide(4))


if __name__ == '__main__':
    unittest.main()
```

```python colab={"base_uri": "https://localhost:8080/"} id="43Rtu9inBHSw" executionInfo={"status": "ok", "timestamp": 1630141749570, "user_tz": -330, "elapsed": 21, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="315d81bc-86fe-4ba2-ea3d-387b51fd13b5"
!python test_money.py -v
```

<!-- #region id="osoEz_rwBmhu" -->
## 5 USD + 10 EUR = 17 USD
<!-- #endregion -->

<!-- #region id="vkR5sSoSBnBb" -->
To test drive the next feature — 5 USD + 10 EUR = 17 USD — it’s enlightening to first sketch out how our program will evolve. TDD plays nicely with software design, contrary to prevailing myths!

The feature, as described in our feature list, says that 5 Dollars and 10 Euros should add up to 17 Dollars, assuming we get 1.2 Dollars for exchanging one Euro.

We realize that “Adding Dollars to Dollars results in Dollars” is an oversimplification. The general principle is that adding Money in different currencies gives us a Portfolio; which we can then express in any one currency (given the necessary Exchange Rates between currencies).

Did we just introduce a new entity: Portfolio? You bet! It’s vital to let our code reflect the realities of our domain. We’re writing code to represent a collection of stock holdings; for which the correct term is a Portfolio. 2

When we add two or more Money s, we should get a Portfolio. We can extend this domain model by saying that we should be able to evaluate a Portfolio in any specific currency. These nouns and verbs give us an idea about the new abstractions in our code which we’ll drive out through tests.
<!-- #endregion -->

<!-- #region id="HFBn0RHuEi77" -->
### 5 USD + 10 USD = 15 USD
<!-- #endregion -->

<!-- #region id="lp7NLbPgFMrZ" -->
**RED**
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="fj-kplEDESQ2" executionInfo={"status": "ok", "timestamp": 1630142664860, "user_tz": -330, "elapsed": 807, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="bafc5c15-095f-4c09-aa26-ff9774a7ce5b"
%%writefile test_money.py
import unittest


class Money:
    def __init__(self, amount, currency):
        self.amount = amount
        self.currency = currency
        
    def times(self, multiplier):
        return Money(self.amount * multiplier, self.currency)

    def divide(self, divisor):
        return Money(self.amount / divisor, self.currency)

    def __eq__(self, other):
        return self.amount == other.amount and self.currency == other.currency


class TestMoney(unittest.TestCase):
    def testMultiplicationInDollars(self):
        fiveDollars = Money(5, "USD")
        tenDollars = Money(10, "USD")
        self.assertEqual(tenDollars, fiveDollars.times(2))
    
    def testMultiplicationInEuros(self):
        tenEuros = Money(10, "EUR")
        twentyEuros = Money(20, "EUR")
        self.assertEqual(twentyEuros, tenEuros.times(2))

    def testDivision(self):
        originalMoney = Money(4002, "KRW")
        expectedMoneyAfterDivision = Money(1000.5, "KRW")
        self.assertEqual(expectedMoneyAfterDivision, originalMoney.divide(4))

    def testAddition(self):
        fiveDollars = Money(5, "USD")
        tenDollars = Money(10, "USD")
        fifteenDollars = Money(15, "USD")
        portfolio = Portfolio()
        portfolio.add(fiveDollars, tenDollars)
        self.assertEqual(fifteenDollars, portfolio.evaluate("USD"))


if __name__ == '__main__':
    unittest.main()
```

```python colab={"base_uri": "https://localhost:8080/"} id="Pr2F1TTwESQ4" executionInfo={"status": "ok", "timestamp": 1630142665856, "user_tz": -330, "elapsed": 16, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="f5197e5c-2133-4f4c-9131-5a020c69cf2b"
!python test_money.py -v
```

<!-- #region id="H_1k8M39FPWv" -->
**GREEN**
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="1EKZ8vb9EzyZ" executionInfo={"status": "ok", "timestamp": 1630142770142, "user_tz": -330, "elapsed": 486, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="7d7f1923-4fb0-4926-85db-8f9782010188"
%%writefile test_money.py
import unittest


class Money:
    def __init__(self, amount, currency):
        self.amount = amount
        self.currency = currency
        
    def times(self, multiplier):
        return Money(self.amount * multiplier, self.currency)

    def divide(self, divisor):
        return Money(self.amount / divisor, self.currency)

    def __eq__(self, other):
        return self.amount == other.amount and self.currency == other.currency


class Portfolio:
    def add(self, *moneys):
        pass

    def evaluate(self, currency):
        return Money(15, "USD")


class TestMoney(unittest.TestCase):
    def testMultiplicationInDollars(self):
        fiveDollars = Money(5, "USD")
        tenDollars = Money(10, "USD")
        self.assertEqual(tenDollars, fiveDollars.times(2))
    
    def testMultiplicationInEuros(self):
        tenEuros = Money(10, "EUR")
        twentyEuros = Money(20, "EUR")
        self.assertEqual(twentyEuros, tenEuros.times(2))

    def testDivision(self):
        originalMoney = Money(4002, "KRW")
        expectedMoneyAfterDivision = Money(1000.5, "KRW")
        self.assertEqual(expectedMoneyAfterDivision, originalMoney.divide(4))

    def testAddition(self):
        fiveDollars = Money(5, "USD")
        tenDollars = Money(10, "USD")
        fifteenDollars = Money(15, "USD")
        portfolio = Portfolio()
        portfolio.add(fiveDollars, tenDollars)
        self.assertEqual(fifteenDollars, portfolio.evaluate("USD"))


if __name__ == '__main__':
    unittest.main()
```

```python colab={"base_uri": "https://localhost:8080/"} id="jnjpR9E7Ezya" executionInfo={"status": "ok", "timestamp": 1630142773035, "user_tz": -330, "elapsed": 1578, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="f671bc9b-1422-43bd-be8a-1d83bbc665b4"
!python test_money.py -v
```

<!-- #region id="0fa-FrGXFKQw" -->
**REFACTOR**
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="N8EQUqO3FgEV" executionInfo={"status": "ok", "timestamp": 1630143141970, "user_tz": -330, "elapsed": 530, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="bb82080a-b921-425e-9317-72521b87367b"
%%writefile test_money.py
import unittest
import functools
import operator


class Money:
    def __init__(self, amount, currency):
        self.amount = amount
        self.currency = currency
        
    def times(self, multiplier):
        return Money(self.amount * multiplier, self.currency)

    def divide(self, divisor):
        return Money(self.amount / divisor, self.currency)

    def __eq__(self, other):
        return self.amount == other.amount and self.currency == other.currency


class Portfolio:
    def __init__(self):
        self.moneys = []

    def add(self, *moneys):
        pass

    def evaluate(self, currency):
        total = functools.reduce(
            operator.add, map(lambda m: m.amount, self.moneys)
        )
        return Money(total, currency)


class TestMoney(unittest.TestCase):
    def testMultiplicationInDollars(self):
        fiveDollars = Money(5, "USD")
        tenDollars = Money(10, "USD")
        self.assertEqual(tenDollars, fiveDollars.times(2))
    
    def testMultiplicationInEuros(self):
        tenEuros = Money(10, "EUR")
        twentyEuros = Money(20, "EUR")
        self.assertEqual(twentyEuros, tenEuros.times(2))

    def testDivision(self):
        originalMoney = Money(4002, "KRW")
        expectedMoneyAfterDivision = Money(1000.5, "KRW")
        self.assertEqual(expectedMoneyAfterDivision, originalMoney.divide(4))

    def testAddition(self):
        fiveDollars = Money(5, "USD")
        tenDollars = Money(10, "USD")
        fifteenDollars = Money(15, "USD")
        portfolio = Portfolio()
        portfolio.add(fiveDollars, tenDollars)
        self.assertEqual(fifteenDollars, portfolio.evaluate("USD"))


if __name__ == '__main__':
    unittest.main()
```

```python colab={"base_uri": "https://localhost:8080/"} id="2TqHy0JCFgEX" executionInfo={"status": "ok", "timestamp": 1630143141973, "user_tz": -330, "elapsed": 23, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="e934afdf-762d-4a12-b0fb-586bfb058684"
!python test_money.py -v
```

<!-- #region id="NZlbdAxiGlKd" -->
Ah: this gives us a new error! TypeError: reduce() of empty sequence with no initial value. We realize two things:
- The add method in Portfolio is still a no-op. That’s why our self.moneys is an empty array; and
- Notwithstanding the above problem, our code should still work with an empty array.
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="T7L8t6JjHO0X" executionInfo={"status": "ok", "timestamp": 1630143384378, "user_tz": -330, "elapsed": 441, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="d6f4ae82-a986-4271-97a5-9a2bda0f4f71"
%%writefile test_money.py
import unittest
import functools
import operator


class Money:
    def __init__(self, amount, currency):
        self.amount = amount
        self.currency = currency
        
    def times(self, multiplier):
        return Money(self.amount * multiplier, self.currency)

    def divide(self, divisor):
        return Money(self.amount / divisor, self.currency)

    def __eq__(self, other):
        return self.amount == other.amount and self.currency == other.currency


class Portfolio:
    def __init__(self):
        self.moneys = []

    def add(self, *moneys):
        self.moneys.extend(moneys)

    def evaluate(self, currency):
        total = functools.reduce(
            operator.add, map(lambda m: m.amount, self.moneys), 0)
        return Money(total, currency)


class TestMoney(unittest.TestCase):
    def testMultiplicationInDollars(self):
        fiveDollars = Money(5, "USD")
        tenDollars = Money(10, "USD")
        self.assertEqual(tenDollars, fiveDollars.times(2))
    
    def testMultiplicationInEuros(self):
        tenEuros = Money(10, "EUR")
        twentyEuros = Money(20, "EUR")
        self.assertEqual(twentyEuros, tenEuros.times(2))

    def testDivision(self):
        originalMoney = Money(4002, "KRW")
        expectedMoneyAfterDivision = Money(1000.5, "KRW")
        self.assertEqual(expectedMoneyAfterDivision, originalMoney.divide(4))

    def testAddition(self):
        fiveDollars = Money(5, "USD")
        tenDollars = Money(10, "USD")
        fifteenDollars = Money(15, "USD")
        portfolio = Portfolio()
        portfolio.add(fiveDollars, tenDollars)
        self.assertEqual(fifteenDollars, portfolio.evaluate("USD"))


if __name__ == '__main__':
    unittest.main()
```

```python colab={"base_uri": "https://localhost:8080/"} id="E9DzRoNnHO0Z" executionInfo={"status": "ok", "timestamp": 1630143384867, "user_tz": -330, "elapsed": 14, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="32053e2d-6358-4fbd-b005-cacc19ef0c4d"
!python test_money.py -v
```

<!-- #region id="R7pTPW-HG8ov" -->
### Modularization
The first thing we’ll do is to separate the test code from the production code. This will require us to solve the problem of “including”, “importing”, or “requiring” the production code in the test code. It is vital that this should always be a one-way dependency.
<!-- #endregion -->

<!-- #region id="HFaCMeU8Icex" -->
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="gnv42SqBIcv1" executionInfo={"status": "ok", "timestamp": 1630143814847, "user_tz": -330, "elapsed": 456, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="10727190-80e1-49bf-c643-0c84f424b878"
%%writefile money.py

class Money:
    def __init__(self, amount, currency):
        self.amount = amount
        self.currency = currency
        
    def times(self, multiplier):
        return Money(self.amount * multiplier, self.currency)

    def divide(self, divisor):
        return Money(self.amount / divisor, self.currency)

    def __eq__(self, other):
        return self.amount == other.amount and self.currency == other.currency
```

```python colab={"base_uri": "https://localhost:8080/"} id="mAtCSToQJJXj" executionInfo={"status": "ok", "timestamp": 1630143899724, "user_tz": -330, "elapsed": 648, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="19d54816-dbc6-48ac-ba0e-34daa0959289"
%%writefile portfolio.py
import functools
import operator

from money import Money


class Portfolio:
    def __init__(self):
        self.moneys = []

    def add(self, *moneys):
        self.moneys.extend(moneys)

    def evaluate(self, currency):
        total = functools.reduce(
            operator.add, map(lambda m: m.amount, self.moneys), 0)
        return Money(total, currency)
```

```python colab={"base_uri": "https://localhost:8080/"} id="1nbyGIeWJPOe" executionInfo={"status": "ok", "timestamp": 1630143919565, "user_tz": -330, "elapsed": 660, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="0f88da8d-1a8a-462b-ae00-a41ba76538a3"
%%writefile test_money.py
import unittest

from money import Money
from portfolio import Portfolio


class TestMoney(unittest.TestCase):
    def testMultiplicationInDollars(self):
        fiveDollars = Money(5, "USD")
        tenDollars = Money(10, "USD")
        self.assertEqual(tenDollars, fiveDollars.times(2))
    
    def testMultiplicationInEuros(self):
        tenEuros = Money(10, "EUR")
        twentyEuros = Money(20, "EUR")
        self.assertEqual(twentyEuros, tenEuros.times(2))

    def testDivision(self):
        originalMoney = Money(4002, "KRW")
        expectedMoneyAfterDivision = Money(1000.5, "KRW")
        self.assertEqual(expectedMoneyAfterDivision, originalMoney.divide(4))

    def testAddition(self):
        fiveDollars = Money(5, "USD")
        tenDollars = Money(10, "USD")
        fifteenDollars = Money(15, "USD")
        portfolio = Portfolio()
        portfolio.add(fiveDollars, tenDollars)
        self.assertEqual(fifteenDollars, portfolio.evaluate("USD"))


if __name__ == '__main__':
    unittest.main()
```

```python colab={"base_uri": "https://localhost:8080/"} id="JPaL1gnGKVHZ" executionInfo={"status": "ok", "timestamp": 1630144130465, "user_tz": -330, "elapsed": 556, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="779c7dc2-a909-41b1-e477-337c21b6fec0"
!python test_money.py -v
```

<!-- #region id="rk_zvKhfKFq3" -->
<!-- #endregion -->

<!-- #region id="SbFtfXwlKM91" -->
### Removing redundancy in tests
We currently have two tests for multiplication, and one each for division and addition. The two tests for multiplication test the same functionality in the Money class. This is a bit of duplication we can do without. Let’s delete the testMultiplicationInDollars and rename the other test to simply testMultiplication. The resulting symmetry — three tests for the three features (Multiplication, Division, and Addition) where each test uses a different currency (Euros, Wons, and Dollars respectively) — is both compact and elegant.
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="DpdgxaJZKYqZ" executionInfo={"status": "ok", "timestamp": 1630144164185, "user_tz": -330, "elapsed": 823, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="a78ddb60-dc98-4cea-fc82-cc02eddfb6b0"
%%writefile test_money.py
import unittest

from money import Money
from portfolio import Portfolio


class TestMoney(unittest.TestCase):
    def testMultiplication(self):
        fiveDollars = Money(5, "USD")
        tenDollars = Money(10, "USD")
        self.assertEqual(tenDollars, fiveDollars.times(2))

    def testDivision(self):
        originalMoney = Money(4002, "KRW")
        expectedMoneyAfterDivision = Money(1000.5, "KRW")
        self.assertEqual(expectedMoneyAfterDivision, originalMoney.divide(4))

    def testAddition(self):
        fiveDollars = Money(5, "USD")
        tenDollars = Money(10, "USD")
        fifteenDollars = Money(15, "USD")
        portfolio = Portfolio()
        portfolio.add(fiveDollars, tenDollars)
        self.assertEqual(fifteenDollars, portfolio.evaluate("USD"))


if __name__ == '__main__':
    unittest.main()
```

```python colab={"base_uri": "https://localhost:8080/"} id="rrAR8JJuKYqb" executionInfo={"status": "ok", "timestamp": 1630144164186, "user_tz": -330, "elapsed": 18, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="a455181c-d719-4369-e3bd-bdb31dff4ed0"
!python test_money.py -v
```

<!-- #region id="_4zwW5hAKe04" -->
## 5 USD + 10 EUR = 17 USD
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="EE2Xd8FLLAib" executionInfo={"status": "ok", "timestamp": 1630144389969, "user_tz": -330, "elapsed": 643, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="26fe2718-ab5c-452d-882a-c94f2256d36f"
%%writefile test_money.py
import unittest

from money import Money
from portfolio import Portfolio


class TestMoney(unittest.TestCase):
    def testMultiplication(self):
        fiveDollars = Money(5, "USD")
        tenDollars = Money(10, "USD")
        self.assertEqual(tenDollars, fiveDollars.times(2))

    def testDivision(self):
        originalMoney = Money(4002, "KRW")
        expectedMoneyAfterDivision = Money(1000.5, "KRW")
        self.assertEqual(expectedMoneyAfterDivision, originalMoney.divide(4))

    def testAddition(self):
        fiveDollars = Money(5, "USD")
        tenDollars = Money(10, "USD")
        fifteenDollars = Money(15, "USD")
        portfolio = Portfolio()
        portfolio.add(fiveDollars, tenDollars)
        self.assertEqual(fifteenDollars, portfolio.evaluate("USD"))

    def testAdditionOfDollarsAndEuros(self):
        fiveDollars = Money(5, "USD")
        tenEuros = Money(10, "EUR")
        portfolio = Portfolio()
        portfolio.add(fiveDollars, tenEuros)
        expectedValue = Money(17, "USD")
        actualValue = portfolio.evaluate("USD")
        self.assertEqual(expectedValue, actualValue)


if __name__ == '__main__':
    unittest.main()
```

```python colab={"base_uri": "https://localhost:8080/"} id="uBh7N5ZELAic" executionInfo={"status": "ok", "timestamp": 1630144390618, "user_tz": -330, "elapsed": 14, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="711751fc-c6a5-4a40-e019-063592ea5dc7"
!python test_money.py -v
```

<!-- #region id="FQvvHH-ULV6b" -->
We expect the test to fail, of course, because we are in the RED phase of our of RGR cycle. However, the error message from the assertion failure is rather cryptic:

```AssertionError: <money.Money object at 0x7f241083e1d0> != <money.Money object at 0x7f241083e2d0>```

Who on earth knows what mysterious goblins reside at those obscure memory addresses!

This is one of those times where we must slow down and write a better failing test before we attempt to get to GREEN. Can we make the assertion statement print a more helpful error message?

The assertEqual method — like most other assertion methods in the unittest package — takes an optional third parameter, which is a custom error message. Let’s provide a formatted string showing the stringified representation of expectedValue and actualValue:
<!-- #endregion -->

<!-- #region id="TGcRVHl-NlnM" -->
> Note: We are not following modular approach because we are learning and are in Jupyter environment. So its better to keep it in same file. But since we covered modularity topic, we know how to make the code modular for future projects.
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="lW51oa5eMVrM" executionInfo={"status": "ok", "timestamp": 1630145142395, "user_tz": -330, "elapsed": 409, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="dd93074b-64da-4529-d78d-9fe4abb89e65"
%%writefile test_money.py
import unittest
import functools
import operator


class Money:
    def __init__(self, amount, currency):
        self.amount = amount
        self.currency = currency
        
    def times(self, multiplier):
        return Money(self.amount * multiplier, self.currency)

    def divide(self, divisor):
        return Money(self.amount / divisor, self.currency)

    def __eq__(self, other):
        return self.amount == other.amount and self.currency == other.currency

    def __str__(self):
        return f"{self.currency} {self.amount:0.2f}"


class Portfolio:
    def __init__(self):
        self.moneys = []

    def add(self, *moneys):
        self.moneys.extend(moneys)

    def evaluate(self, currency):
        total = functools.reduce(
            operator.add, map(lambda m: m.amount, self.moneys), 0)
        return Money(total, currency)


class TestMoney(unittest.TestCase):
    def testMultiplication(self):
        fiveDollars = Money(5, "USD")
        tenDollars = Money(10, "USD")
        self.assertEqual(tenDollars, fiveDollars.times(2))

    def testDivision(self):
        originalMoney = Money(4002, "KRW")
        expectedMoneyAfterDivision = Money(1000.5, "KRW")
        self.assertEqual(expectedMoneyAfterDivision, originalMoney.divide(4))

    def testAddition(self):
        fiveDollars = Money(5, "USD")
        tenDollars = Money(10, "USD")
        fifteenDollars = Money(15, "USD")
        portfolio = Portfolio()
        portfolio.add(fiveDollars, tenDollars)
        self.assertEqual(fifteenDollars, portfolio.evaluate("USD"))

    def testAdditionOfDollarsAndEuros(self):
        fiveDollars = Money(5, "USD")
        tenEuros = Money(10, "EUR")
        portfolio = Portfolio()
        portfolio.add(fiveDollars, tenEuros)
        expectedValue = Money(17, "USD")
        actualValue = portfolio.evaluate("USD")
        self.assertEqual(expectedValue, actualValue,
                "%s != %s"%(expectedValue, actualValue))


if __name__ == '__main__':
    unittest.main()
```

```python colab={"base_uri": "https://localhost:8080/"} id="jAFZyfjIMRNP" executionInfo={"status": "ok", "timestamp": 1630145142843, "user_tz": -330, "elapsed": 17, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="8919931f-213d-4dee-ff33-21d0f0e13994"
!python test_money.py -v
```

<!-- #region id="0QyQP1uROeGu" -->
This validates our belief that the evaluate method, as currently implemented, mindlessly adds the amounts of all Money objects (5 and 10 in our test) to get the result, with no regard to the currencies (USD and EUR, respectively, in our test).

A closer examination of the evaluate method shows that the mindlessness is in the lambda expression. It maps every Money object to its amount, regardless of its currency. These amounts are then added up by the reduce function using the add operator.

What if the lambda expression mapped every Money object to its converted value? The target currency for the conversion would be the currency in which the Portfolio is being evaluated.

```python
total = functools.reduce(operator.add,
          map(lambda m: self.__convert(m, currency), self.moneys), 0)
```

How should we implement the __convert method? Converting to the same currency as that of the Money is trivial: the Money’s amount doesn’t change in this case. When converting to a different currency, we’ll multiply Money’s amount with the (for now) hard-coded exchange rate between USD and EUR:

<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="qfvWBipMOrvT" executionInfo={"status": "ok", "timestamp": 1630145386687, "user_tz": -330, "elapsed": 855, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="fe971ee1-424a-4aa8-81b7-5871678e8ad1"
%%writefile test_money.py
import unittest
import functools
import operator


class Money:
    def __init__(self, amount, currency):
        self.amount = amount
        self.currency = currency
        
    def times(self, multiplier):
        return Money(self.amount * multiplier, self.currency)

    def divide(self, divisor):
        return Money(self.amount / divisor, self.currency)

    def __eq__(self, other):
        return self.amount == other.amount and self.currency == other.currency

    def __str__(self):
        return f"{self.currency} {self.amount:0.2f}"


class Portfolio:
    def __init__(self):
        self.moneys = []

    def add(self, *moneys):
        self.moneys.extend(moneys)

    def evaluate(self, currency):
        total = functools.reduce(operator.add,
                                 map(lambda m: self.__convert(m, currency), self.moneys), 0)
        return Money(total, currency)

    def __convert(self, aMoney, aCurrency):
        if aMoney.currency == aCurrency:
            return aMoney.amount
        else:
            return aMoney.amount * 1.2


class TestMoney(unittest.TestCase):
    def testMultiplication(self):
        fiveDollars = Money(5, "USD")
        tenDollars = Money(10, "USD")
        self.assertEqual(tenDollars, fiveDollars.times(2))

    def testDivision(self):
        originalMoney = Money(4002, "KRW")
        expectedMoneyAfterDivision = Money(1000.5, "KRW")
        self.assertEqual(expectedMoneyAfterDivision, originalMoney.divide(4))

    def testAddition(self):
        fiveDollars = Money(5, "USD")
        tenDollars = Money(10, "USD")
        fifteenDollars = Money(15, "USD")
        portfolio = Portfolio()
        portfolio.add(fiveDollars, tenDollars)
        self.assertEqual(fifteenDollars, portfolio.evaluate("USD"))

    def testAdditionOfDollarsAndEuros(self):
        fiveDollars = Money(5, "USD")
        tenEuros = Money(10, "EUR")
        portfolio = Portfolio()
        portfolio.add(fiveDollars, tenEuros)
        expectedValue = Money(17, "USD")
        actualValue = portfolio.evaluate("USD")
        self.assertEqual(expectedValue, actualValue,
                "%s != %s"%(expectedValue, actualValue))


if __name__ == '__main__':
    unittest.main()
```

```python colab={"base_uri": "https://localhost:8080/"} id="7aiaupt_OrvV" executionInfo={"status": "ok", "timestamp": 1630145386689, "user_tz": -330, "elapsed": 19, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="fa18eb1a-bdac-4bfc-9a8e-d86c8051c592"
!python test_money.py -v
```

<!-- #region id="hrVw2HuCPHCZ" -->
The test is green. Yay … and hmm! We should do the refactoring to remove the ugliness of this code. Here are some problems with it:
- The exchange rate is hard-coded. It should be declared as a variable.
- The exchange rate isn’t dependent on the currency. It should be looked up based on the two currencies involved.
- The exchange rate should be modifiable.
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="Eees3qEBPhLn" executionInfo={"status": "ok", "timestamp": 1630145611646, "user_tz": -330, "elapsed": 411, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="01b1197a-beab-4925-a7fd-db9fbb25586d"
%%writefile test_money.py
import unittest
import functools
import operator


class Money:
    def __init__(self, amount, currency):
        self.amount = amount
        self.currency = currency
        
    def times(self, multiplier):
        return Money(self.amount * multiplier, self.currency)

    def divide(self, divisor):
        return Money(self.amount / divisor, self.currency)

    def __eq__(self, other):
        return self.amount == other.amount and self.currency == other.currency

    def __str__(self):
        return f"{self.currency} {self.amount:0.2f}"


class Portfolio:
    def __init__(self):
        self.moneys = []
        self._eur_to_usd = 1.2

    def add(self, *moneys):
        self.moneys.extend(moneys)

    def evaluate(self, currency):
        total = functools.reduce(operator.add,
                                 map(lambda m: self.__convert(m, currency), self.moneys), 0)
        return Money(total, currency)

    def __convert(self, aMoney, aCurrency):
        if aMoney.currency == aCurrency:
            return aMoney.amount
        else:
            return aMoney.amount * self._eur_to_usd


class TestMoney(unittest.TestCase):
    def testMultiplication(self):
        fiveDollars = Money(5, "USD")
        tenDollars = Money(10, "USD")
        self.assertEqual(tenDollars, fiveDollars.times(2))

    def testDivision(self):
        originalMoney = Money(4002, "KRW")
        expectedMoneyAfterDivision = Money(1000.5, "KRW")
        self.assertEqual(expectedMoneyAfterDivision, originalMoney.divide(4))

    def testAddition(self):
        fiveDollars = Money(5, "USD")
        tenDollars = Money(10, "USD")
        fifteenDollars = Money(15, "USD")
        portfolio = Portfolio()
        portfolio.add(fiveDollars, tenDollars)
        self.assertEqual(fifteenDollars, portfolio.evaluate("USD"))

    def testAdditionOfDollarsAndEuros(self):
        fiveDollars = Money(5, "USD")
        tenEuros = Money(10, "EUR")
        portfolio = Portfolio()
        portfolio.add(fiveDollars, tenEuros)
        expectedValue = Money(17, "USD")
        actualValue = portfolio.evaluate("USD")
        self.assertEqual(expectedValue, actualValue,
                "%s != %s"%(expectedValue, actualValue))


if __name__ == '__main__':
    unittest.main()
```

```python colab={"base_uri": "https://localhost:8080/"} id="BiXKkIHTRD_p" executionInfo={"status": "ok", "timestamp": 1630145891806, "user_tz": -330, "elapsed": 596, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="f11b5a3c-aec7-403f-9f3f-bafff071604d"
!python test_money.py -v
```

<!-- #region id="3pPqyFCEQAKf" -->
Here’s the current state of our evaluate feature vis-à-vis Money s in a Portfolio:
1. When “converting” a Money in a currency to the same currency, it returns the amount of the Money. This is correct: the exchange rate for any currency to itself is 1.
2. In all other cases, the amount of the Money is multiplied by a fixed number (1.2). This is correct in a very limited sense: this rate ensures conversions from USD to EUR only. There is no way to modify this exchange rate or specify any other rate.

Our currency conversion code does one thing correctly and another thing almost correctly. It’s time to make it work correctly in both cases. In this chapter, we’ll introduce — at long last — the conversion of money from one currency into another using currency-specific exchange rates.
<!-- #endregion -->

<!-- #region id="3okIIVv9QEWX" -->
What we need is a hashmap that allows us to look up exchange rates given a “from” currency and a “to” currency. The hashmap would be a representation of an exchange rate table we regularly see in banks and currency exchange counters at airports.
<!-- #endregion -->

<!-- #region id="-vPlRknuQJOp" -->
## 1 USD + 1100 KRW = 2200 KRW
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="lzOtlrBZRFtJ" executionInfo={"status": "ok", "timestamp": 1630145968868, "user_tz": -330, "elapsed": 689, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="16c961c3-09f8-4554-ddf8-f91d42665e7e"
%%writefile test_money.py
import unittest
import functools
import operator


class Money:
    def __init__(self, amount, currency):
        self.amount = amount
        self.currency = currency
        
    def times(self, multiplier):
        return Money(self.amount * multiplier, self.currency)

    def divide(self, divisor):
        return Money(self.amount / divisor, self.currency)

    def __eq__(self, other):
        return self.amount == other.amount and self.currency == other.currency

    def __str__(self):
        return f"{self.currency} {self.amount:0.2f}"


class Portfolio:
    def __init__(self):
        self.moneys = []
        self._eur_to_usd = 1.2

    def add(self, *moneys):
        self.moneys.extend(moneys)

    def evaluate(self, currency):
        total = functools.reduce(operator.add,
                                 map(lambda m: self.__convert(m, currency), self.moneys), 0)
        return Money(total, currency)

    def __convert(self, aMoney, aCurrency):
        if aMoney.currency == aCurrency:
            return aMoney.amount
        else:
            return aMoney.amount * self._eur_to_usd


class TestMoney(unittest.TestCase):
    def testMultiplication(self):
        fiveDollars = Money(5, "USD")
        tenDollars = Money(10, "USD")
        self.assertEqual(tenDollars, fiveDollars.times(2))

    def testDivision(self):
        originalMoney = Money(4002, "KRW")
        expectedMoneyAfterDivision = Money(1000.5, "KRW")
        self.assertEqual(expectedMoneyAfterDivision, originalMoney.divide(4))

    def testAddition(self):
        fiveDollars = Money(5, "USD")
        tenDollars = Money(10, "USD")
        fifteenDollars = Money(15, "USD")
        portfolio = Portfolio()
        portfolio.add(fiveDollars, tenDollars)
        self.assertEqual(fifteenDollars, portfolio.evaluate("USD"))

    def testAdditionOfDollarsAndEuros(self):
        fiveDollars = Money(5, "USD")
        tenEuros = Money(10, "EUR")
        portfolio = Portfolio()
        portfolio.add(fiveDollars, tenEuros)
        expectedValue = Money(17, "USD")
        actualValue = portfolio.evaluate("USD")
        self.assertEqual(expectedValue, actualValue,
                "%s != %s"%(expectedValue, actualValue))

    def testAdditionOfDollarsAndWons(self):
        oneDollar = Money(1, "USD")
        elevenHundredWon = Money(1100, "KRW")
        portfolio = Portfolio()
        portfolio.add(oneDollar, elevenHundredWon)
        expectedValue = Money(2200, "KRW")
        actualValue = portfolio.evaluate("KRW")
        self.assertEqual(expectedValue, actualValue,
        "%s != %s"%(expectedValue, actualValue))


if __name__ == '__main__':
    unittest.main()
```

```python colab={"base_uri": "https://localhost:8080/"} id="ZEYMDCpjRFtK" executionInfo={"status": "ok", "timestamp": 1630145968869, "user_tz": -330, "elapsed": 17, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="f6de18d7-0d63-4e85-9816-566598c54e6f"
!python test_money.py -v
```

<!-- #region id="bOUbgKHyRM4B" -->
The __convert method is using the rate eurToUsd, which is incorrect for this case. That’s where the peculiar amount 1101.20 comes from.

Let’s introduce a dictionary to store exchange rates. We’ll add the two entries we need currently: EUR→USD = 1.2 and USD→KRW = 1100. We’ll keep this dictionary in the __convert method to begin with:
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="cWbC3vgCRsn-" executionInfo={"status": "ok", "timestamp": 1630146238755, "user_tz": -330, "elapsed": 1677, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="68a51f51-6507-4ad3-95c1-cd689350002c"
%%writefile test_money.py
import unittest
import functools
import operator


class Money:
    def __init__(self, amount, currency):
        self.amount = amount
        self.currency = currency
        
    def times(self, multiplier):
        return Money(self.amount * multiplier, self.currency)

    def divide(self, divisor):
        return Money(self.amount / divisor, self.currency)

    def __eq__(self, other):
        return self.amount == other.amount and self.currency == other.currency

    def __str__(self):
        return f"{self.currency} {self.amount:0.2f}"


class Portfolio:
    def __init__(self):
        self.moneys = []

    def add(self, *moneys):
        self.moneys.extend(moneys)

    def evaluate(self, currency):
        total = functools.reduce(operator.add,
                                 map(lambda m: self.__convert(m, currency), self.moneys), 0)
        return Money(total, currency)

    def __convert(self, aMoney, aCurrency):
        exchangeRates = {'EUR->USD': 1.2, 'USD->KRW': 1100}
        if aMoney.currency == aCurrency:
            return aMoney.amount
        else:
            key = aMoney.currency + '->' + aCurrency
            return aMoney.amount * exchangeRates[key]


class TestMoney(unittest.TestCase):
    def testMultiplication(self):
        fiveDollars = Money(5, "USD")
        tenDollars = Money(10, "USD")
        self.assertEqual(tenDollars, fiveDollars.times(2))

    def testDivision(self):
        originalMoney = Money(4002, "KRW")
        expectedMoneyAfterDivision = Money(1000.5, "KRW")
        self.assertEqual(expectedMoneyAfterDivision, originalMoney.divide(4))

    def testAddition(self):
        fiveDollars = Money(5, "USD")
        tenDollars = Money(10, "USD")
        fifteenDollars = Money(15, "USD")
        portfolio = Portfolio()
        portfolio.add(fiveDollars, tenDollars)
        self.assertEqual(fifteenDollars, portfolio.evaluate("USD"))

    def testAdditionOfDollarsAndEuros(self):
        fiveDollars = Money(5, "USD")
        tenEuros = Money(10, "EUR")
        portfolio = Portfolio()
        portfolio.add(fiveDollars, tenEuros)
        expectedValue = Money(17, "USD")
        actualValue = portfolio.evaluate("USD")
        self.assertEqual(expectedValue, actualValue,
                "%s != %s"%(expectedValue, actualValue))

    def testAdditionOfDollarsAndWons(self):
        oneDollar = Money(1, "USD")
        elevenHundredWon = Money(1100, "KRW")
        portfolio = Portfolio()
        portfolio.add(oneDollar, elevenHundredWon)
        expectedValue = Money(2200, "KRW")
        actualValue = portfolio.evaluate("KRW")
        self.assertEqual(expectedValue, actualValue,
        "%s != %s"%(expectedValue, actualValue))


if __name__ == '__main__':
    unittest.main()
```

```python colab={"base_uri": "https://localhost:8080/"} id="oHJaBf7cRsoA" executionInfo={"status": "ok", "timestamp": 1630146238757, "user_tz": -330, "elapsed": 20, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="8597711a-7101-400d-f0d8-71ee82b16995"
!python test_money.py -v
```

<!-- #region id="s7RftzBYSYyG" -->
With these changes, all our tests turn green again.
<!-- #endregion -->

<!-- #region id="SAzVHT0mSbNA" -->
## Error Handling
<!-- #endregion -->

<!-- #region id="bn61o8YZbGUI" -->
We’d like to raise an Exception when evaluate fails due to missing exchange rates. In its message, the exception should describe all the missing exchange rate keys (ie. the “from” and “to” currencies). Let’s start with a test that validates this behavior.
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="tKFfWFLPbMfb" executionInfo={"status": "ok", "timestamp": 1630148668498, "user_tz": -330, "elapsed": 637, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="4a7031c6-7cb6-43f7-9a36-63a0447d7e03"
%%writefile test_money.py
import unittest
import functools
import operator


class Money:
    def __init__(self, amount, currency):
        self.amount = amount
        self.currency = currency
        
    def times(self, multiplier):
        return Money(self.amount * multiplier, self.currency)

    def divide(self, divisor):
        return Money(self.amount / divisor, self.currency)

    def __eq__(self, other):
        return self.amount == other.amount and self.currency == other.currency

    def __str__(self):
        return f"{self.currency} {self.amount:0.2f}"


class Portfolio:
    def __init__(self):
        self.moneys = []

    def add(self, *moneys):
        self.moneys.extend(moneys)

    def evaluate(self, currency):
        total = functools.reduce(operator.add,
                                 map(lambda m: self.__convert(m, currency), self.moneys), 0)
        return Money(total, currency)

    def __convert(self, aMoney, aCurrency):
        exchangeRates = {'EUR->USD': 1.2, 'USD->KRW': 1100}
        if aMoney.currency == aCurrency:
            return aMoney.amount
        else:
            key = aMoney.currency + '->' + aCurrency
            return aMoney.amount * exchangeRates[key]


class TestMoney(unittest.TestCase):
    def testMultiplication(self):
        fiveDollars = Money(5, "USD")
        tenDollars = Money(10, "USD")
        self.assertEqual(tenDollars, fiveDollars.times(2))

    def testDivision(self):
        originalMoney = Money(4002, "KRW")
        expectedMoneyAfterDivision = Money(1000.5, "KRW")
        self.assertEqual(expectedMoneyAfterDivision, originalMoney.divide(4))

    def testAddition(self):
        fiveDollars = Money(5, "USD")
        tenDollars = Money(10, "USD")
        fifteenDollars = Money(15, "USD")
        portfolio = Portfolio()
        portfolio.add(fiveDollars, tenDollars)
        self.assertEqual(fifteenDollars, portfolio.evaluate("USD"))

    def testAdditionOfDollarsAndEuros(self):
        fiveDollars = Money(5, "USD")
        tenEuros = Money(10, "EUR")
        portfolio = Portfolio()
        portfolio.add(fiveDollars, tenEuros)
        expectedValue = Money(17, "USD")
        actualValue = portfolio.evaluate("USD")
        self.assertEqual(expectedValue, actualValue,
                "%s != %s"%(expectedValue, actualValue))

    def testAdditionOfDollarsAndWons(self):
        oneDollar = Money(1, "USD")
        elevenHundredWon = Money(1100, "KRW")
        portfolio = Portfolio()
        portfolio.add(oneDollar, elevenHundredWon)
        expectedValue = Money(2200, "KRW")
        actualValue = portfolio.evaluate("KRW")
        self.assertEqual(expectedValue, actualValue,
        "%s != %s"%(expectedValue, actualValue))

    def testAdditionWithMultipleMissingExchangeRates(self):
        oneDollar = Money(1, "USD")
        oneEuro = Money(1, "EUR")
        oneWon = Money(1, "KRW")
        portfolio = Portfolio()
        portfolio.add(oneDollar, oneEuro, oneWon)
        with self.assertRaisesRegex(
            Exception,
            "Missing exchange rate\(s\):\[USD\->Kalganid,EUR->Kalganid,KRW->Kalganid]",
        ):
            portfolio.evaluate("Kalganid")


if __name__ == '__main__':
    unittest.main()
```

```python colab={"base_uri": "https://localhost:8080/"} id="_bjA1WA6bMfd" executionInfo={"status": "ok", "timestamp": 1630148668500, "user_tz": -330, "elapsed": 28, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="709da868-0983-4479-fddf-1a90b9b44fe7"
!python test_money.py -v
```

<!-- #region id="OTVO5MDkbqWu" -->
> Note: assertRaisesRegex is one of the many useful assertion methods defined in Python’s TestCase class. Since our exception string has several characters that have special meaning in regular expressions, we escape them using the backslash character.

This test is similar to the existing tests for addition, with a couple of differences. First: we are attempting to evaluate a Portfolio in “Kalganid”, for which no exchange rates exist. Second: we expect the evaluate method to throw an exception with a specific error message that we verify in the assertRaisesRegex statement.

The test fails with two exceptions. First, there’s the KeyError which we expect: there is no exchange rate key involving the “Kalganid” currency. The second error is the assertion failure we sought to cause. 

We need to modify our evaluate method to respond to Exceptions arising from its calls to __convert. Let’s unroll the lambda expression into a loop and add a try ... except block to capture any failures. If there are no failures, we return a new Money object as before. If there are failures, we raise an Exception whose message is a comma-separated list of the stringified KeyError exceptions that are caught:
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="IFP5picFcYbw" executionInfo={"status": "ok", "timestamp": 1630148972785, "user_tz": -330, "elapsed": 681, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="90cda089-fc53-4eed-9402-4a5e1b93e44e"
%%writefile test_money.py
import unittest
import functools
import operator


class Money:
    def __init__(self, amount, currency):
        self.amount = amount
        self.currency = currency
        
    def times(self, multiplier):
        return Money(self.amount * multiplier, self.currency)

    def divide(self, divisor):
        return Money(self.amount / divisor, self.currency)

    def __eq__(self, other):
        return self.amount == other.amount and self.currency == other.currency

    def __str__(self):
        return f"{self.currency} {self.amount:0.2f}"


class Portfolio:
    def __init__(self):
        self.moneys = []

    def add(self, *moneys):
        self.moneys.extend(moneys)

    def evaluate(self, currency):
        total = 0.0
        failures = []
        for m in self.moneys:
            try:
                total += self.__convert(m, currency)
            except KeyError as ke:
                failures.append(ke)

        if len(failures) == 0:
            return Money(total, currency)

        failureMessage = ",".join(str(f) for f in failures)
        raise Exception("Missing exchange rate(s):[" + failureMessage + "]")

    def __convert(self, aMoney, aCurrency):
        exchangeRates = {'EUR->USD': 1.2, 'USD->KRW': 1100}
        if aMoney.currency == aCurrency:
            return aMoney.amount
        else:
            key = aMoney.currency + '->' + aCurrency
            return aMoney.amount * exchangeRates[key]


class TestMoney(unittest.TestCase):
    def testMultiplication(self):
        fiveDollars = Money(5, "USD")
        tenDollars = Money(10, "USD")
        self.assertEqual(tenDollars, fiveDollars.times(2))

    def testDivision(self):
        originalMoney = Money(4002, "KRW")
        expectedMoneyAfterDivision = Money(1000.5, "KRW")
        self.assertEqual(expectedMoneyAfterDivision, originalMoney.divide(4))

    def testAddition(self):
        fiveDollars = Money(5, "USD")
        tenDollars = Money(10, "USD")
        fifteenDollars = Money(15, "USD")
        portfolio = Portfolio()
        portfolio.add(fiveDollars, tenDollars)
        self.assertEqual(fifteenDollars, portfolio.evaluate("USD"))

    def testAdditionOfDollarsAndEuros(self):
        fiveDollars = Money(5, "USD")
        tenEuros = Money(10, "EUR")
        portfolio = Portfolio()
        portfolio.add(fiveDollars, tenEuros)
        expectedValue = Money(17, "USD")
        actualValue = portfolio.evaluate("USD")
        self.assertEqual(expectedValue, actualValue,
                "%s != %s"%(expectedValue, actualValue))

    def testAdditionOfDollarsAndWons(self):
        oneDollar = Money(1, "USD")
        elevenHundredWon = Money(1100, "KRW")
        portfolio = Portfolio()
        portfolio.add(oneDollar, elevenHundredWon)
        expectedValue = Money(2200, "KRW")
        actualValue = portfolio.evaluate("KRW")
        self.assertEqual(expectedValue, actualValue,
        "%s != %s"%(expectedValue, actualValue))

    def testAdditionWithMultipleMissingExchangeRates(self):
        oneDollar = Money(1, "USD")
        oneEuro = Money(1, "EUR")
        oneWon = Money(1, "KRW")
        portfolio = Portfolio()
        portfolio.add(oneDollar, oneEuro, oneWon)
        with self.assertRaisesRegex(
            Exception,
            "Missing exchange rate\(s\):\[USD\->Kalganid,EUR->Kalganid,KRW->Kalganid]",
        ):
            portfolio.evaluate("Kalganid")


if __name__ == '__main__':
    unittest.main()
```

```python colab={"base_uri": "https://localhost:8080/"} id="6Kh6De_UcYby" executionInfo={"status": "ok", "timestamp": 1630148973350, "user_tz": -330, "elapsed": 21, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="34f4f9a2-44f6-4d28-f10f-ef2decdf26d8"
!python test_money.py -v
```

<!-- #region id="NmA4PAMzc0px" -->
A simple change to the way we assemble our failureMessage can fix our problem:
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="ZAh9V0o5dWvk" executionInfo={"status": "ok", "timestamp": 1630149141972, "user_tz": -330, "elapsed": 872, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="d3295d6a-e86a-45fc-dbbe-76dc058b3219"
%%writefile test_money.py
import unittest
import functools
import operator


class Money:
    def __init__(self, amount, currency):
        self.amount = amount
        self.currency = currency
        
    def times(self, multiplier):
        return Money(self.amount * multiplier, self.currency)

    def divide(self, divisor):
        return Money(self.amount / divisor, self.currency)

    def __eq__(self, other):
        return self.amount == other.amount and self.currency == other.currency

    def __str__(self):
        return f"{self.currency} {self.amount:0.2f}"


class Portfolio:
    def __init__(self):
        self.moneys = []

    def add(self, *moneys):
        self.moneys.extend(moneys)

    def evaluate(self, currency):
        total = 0.0
        failures = []
        for m in self.moneys:
            try:
                total += self.__convert(m, currency)
            except KeyError as ke:
                failures.append(ke)

        if len(failures) == 0:
            return Money(total, currency)

        failureMessage = ",".join(f.args[0] for f in failures)
        raise Exception("Missing exchange rate(s):[" + failureMessage + "]")

    def __convert(self, aMoney, aCurrency):
        exchangeRates = {'EUR->USD': 1.2, 'USD->KRW': 1100}
        if aMoney.currency == aCurrency:
            return aMoney.amount
        else:
            key = aMoney.currency + '->' + aCurrency
            return aMoney.amount * exchangeRates[key]


class TestMoney(unittest.TestCase):
    def testMultiplication(self):
        fiveDollars = Money(5, "USD")
        tenDollars = Money(10, "USD")
        self.assertEqual(tenDollars, fiveDollars.times(2))

    def testDivision(self):
        originalMoney = Money(4002, "KRW")
        expectedMoneyAfterDivision = Money(1000.5, "KRW")
        self.assertEqual(expectedMoneyAfterDivision, originalMoney.divide(4))

    def testAddition(self):
        fiveDollars = Money(5, "USD")
        tenDollars = Money(10, "USD")
        fifteenDollars = Money(15, "USD")
        portfolio = Portfolio()
        portfolio.add(fiveDollars, tenDollars)
        self.assertEqual(fifteenDollars, portfolio.evaluate("USD"))

    def testAdditionOfDollarsAndEuros(self):
        fiveDollars = Money(5, "USD")
        tenEuros = Money(10, "EUR")
        portfolio = Portfolio()
        portfolio.add(fiveDollars, tenEuros)
        expectedValue = Money(17, "USD")
        actualValue = portfolio.evaluate("USD")
        self.assertEqual(expectedValue, actualValue,
                "%s != %s"%(expectedValue, actualValue))

    def testAdditionOfDollarsAndWons(self):
        oneDollar = Money(1, "USD")
        elevenHundredWon = Money(1100, "KRW")
        portfolio = Portfolio()
        portfolio.add(oneDollar, elevenHundredWon)
        expectedValue = Money(2200, "KRW")
        actualValue = portfolio.evaluate("KRW")
        self.assertEqual(expectedValue, actualValue,
        "%s != %s"%(expectedValue, actualValue))

    def testAdditionWithMultipleMissingExchangeRates(self):
        oneDollar = Money(1, "USD")
        oneEuro = Money(1, "EUR")
        oneWon = Money(1, "KRW")
        portfolio = Portfolio()
        portfolio.add(oneDollar, oneEuro, oneWon)
        with self.assertRaisesRegex(
            Exception,
            "Missing exchange rate\(s\):\[USD\->Kalganid,EUR->Kalganid,KRW->Kalganid]",
        ):
            portfolio.evaluate("Kalganid")


if __name__ == '__main__':
    unittest.main()
```

```python colab={"base_uri": "https://localhost:8080/"} id="aLyUzJK0dWvl" executionInfo={"status": "ok", "timestamp": 1630149142436, "user_tz": -330, "elapsed": 19, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="edbde693-e9d2-417e-82d4-93e7b25d4259"
!python test_money.py -v
```

<!-- #region id="-FeLjYFAdd-x" -->
## Banking
<!-- #endregion -->

<!-- #region id="RL7cLxvneOby" -->
What is the name of the real world institution that helps us exchange money? A bank. Or a Currency Exchange. Often, a domain will have multiple similar entities that are indistinguishable from the perspective of our model. Learning which differences are salient and which are insignificant is vital to effective domain modeling.

We’ll select the name Bank to represent this missing entity. What should be the responsibilities of the Bank? It should hold exchange rates, for one thing. And it should be able to convert moneys between currencies based on the exchange rate from one currency to another. The Bank should allow asymmetric exchange rates, because that is true in the real world. Finally, the bank should clearly inform us when it cannot exchange money in one currency into a another currency because of a missing exchange rate.

Having identified the need for this new entity, the next question is: how should the dependencies between Bank and the other two existing entities — Money and Portfolio — look?

Clearly, Bank needs Money to operate. Portfolio would need both Money and Bank; the former association is one of aggregation and the latter is an interface dependency: Portfolio uses the convert method in Bank.
<!-- #endregion -->

<!-- #region id="q8ec-_EGe7oM" -->
<!-- #endregion -->

<!-- #region id="H90n9NGMfAOT" -->
> Tip: The dependency of Portfolio to Bank is kept to a minimum: it is provided as a parameter to the Evaluate method. This type of dependency injection is called “Method Injection”, because we are “injecting” the dependency directly into the method that needs it.
<!-- #endregion -->

<!-- #region id="2-40O1QPfRSh" -->
The approach we’ll take will be a combination of writing new unit tests — which is the heart of TDD and what we’ve done thus far — and refactoring existing unit tests. We know that the existing tests provide a valuable safeguard: they verify that the features we’ve built, all the crossed-out lines on our list, work as expected. We’ll continue to run these tests, modifying their implementation as needed while keeping their purpose intact. This two-pronged approach of writing new tests and refactoring existing ones will give us the assurance we need as we heal our code of its ills.
<!-- #endregion -->

<!-- #region id="Iadjv9EZePxf" -->
Our first goal is to write a test to convert one Money object into another, using the as-yet-undefined Bank abstraction:
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="KLs1lQRVfkKQ" executionInfo={"status": "ok", "timestamp": 1630149747643, "user_tz": -330, "elapsed": 891, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="fc70f29b-f5c8-4314-d8d9-f75360f85410"
%%writefile test_money.py
import unittest
import functools
import operator


class Money:
    def __init__(self, amount, currency):
        self.amount = amount
        self.currency = currency
        
    def times(self, multiplier):
        return Money(self.amount * multiplier, self.currency)

    def divide(self, divisor):
        return Money(self.amount / divisor, self.currency)

    def __eq__(self, other):
        return self.amount == other.amount and self.currency == other.currency

    def __str__(self):
        return f"{self.currency} {self.amount:0.2f}"


class Portfolio:
    def __init__(self):
        self.moneys = []

    def add(self, *moneys):
        self.moneys.extend(moneys)

    def evaluate(self, currency):
        total = 0.0
        failures = []
        for m in self.moneys:
            try:
                total += self.__convert(m, currency)
            except KeyError as ke:
                failures.append(ke)

        if len(failures) == 0:
            return Money(total, currency)

        failureMessage = ",".join(f.args[0] for f in failures)
        raise Exception("Missing exchange rate(s):[" + failureMessage + "]")

    def __convert(self, aMoney, aCurrency):
        exchangeRates = {'EUR->USD': 1.2, 'USD->KRW': 1100}
        if aMoney.currency == aCurrency:
            return aMoney.amount
        else:
            key = aMoney.currency + '->' + aCurrency
            return aMoney.amount * exchangeRates[key]


class TestMoney(unittest.TestCase):
    def testMultiplication(self):
        fiveDollars = Money(5, "USD")
        tenDollars = Money(10, "USD")
        self.assertEqual(tenDollars, fiveDollars.times(2))

    def testDivision(self):
        originalMoney = Money(4002, "KRW")
        expectedMoneyAfterDivision = Money(1000.5, "KRW")
        self.assertEqual(expectedMoneyAfterDivision, originalMoney.divide(4))

    def testAddition(self):
        fiveDollars = Money(5, "USD")
        tenDollars = Money(10, "USD")
        fifteenDollars = Money(15, "USD")
        portfolio = Portfolio()
        portfolio.add(fiveDollars, tenDollars)
        self.assertEqual(fifteenDollars, portfolio.evaluate("USD"))

    def testAdditionOfDollarsAndEuros(self):
        fiveDollars = Money(5, "USD")
        tenEuros = Money(10, "EUR")
        portfolio = Portfolio()
        portfolio.add(fiveDollars, tenEuros)
        expectedValue = Money(17, "USD")
        actualValue = portfolio.evaluate("USD")
        self.assertEqual(expectedValue, actualValue,
                "%s != %s"%(expectedValue, actualValue))

    def testAdditionOfDollarsAndWons(self):
        oneDollar = Money(1, "USD")
        elevenHundredWon = Money(1100, "KRW")
        portfolio = Portfolio()
        portfolio.add(oneDollar, elevenHundredWon)
        expectedValue = Money(2200, "KRW")
        actualValue = portfolio.evaluate("KRW")
        self.assertEqual(expectedValue, actualValue,
        "%s != %s"%(expectedValue, actualValue))

    def testAdditionWithMultipleMissingExchangeRates(self):
        oneDollar = Money(1, "USD")
        oneEuro = Money(1, "EUR")
        oneWon = Money(1, "KRW")
        portfolio = Portfolio()
        portfolio.add(oneDollar, oneEuro, oneWon)
        with self.assertRaisesRegex(
            Exception,
            "Missing exchange rate\(s\):\[USD\->Kalganid,EUR->Kalganid,KRW->Kalganid]",
        ):
            portfolio.evaluate("Kalganid")

    def testConversion(self):
        bank = Bank()
        bank.addExchangeRate("EUR", "USD", 1.2)
        tenEuros = Money(10, "EUR")
        self.assertEqual(bank.convert(tenEuros, "USD"), Money(12, "USD"))


if __name__ == '__main__':
    unittest.main()
```

```python colab={"base_uri": "https://localhost:8080/"} id="emzpcE_CfkKR" executionInfo={"status": "ok", "timestamp": 1630149747645, "user_tz": -330, "elapsed": 29, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="dd3a26af-1366-49a0-b667-f5c8d6e4cf7e"
!python test_money.py -v
```

```python colab={"base_uri": "https://localhost:8080/"} id="g2Q3KR7ef3mr" executionInfo={"status": "ok", "timestamp": 1630149854255, "user_tz": -330, "elapsed": 502, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="f3d344e5-4e9f-4bfd-c25d-829c1cba8d19"
%%writefile test_money.py
import unittest
import functools
import operator


class Money:
    def __init__(self, amount, currency):
        self.amount = amount
        self.currency = currency
        
    def times(self, multiplier):
        return Money(self.amount * multiplier, self.currency)

    def divide(self, divisor):
        return Money(self.amount / divisor, self.currency)

    def __eq__(self, other):
        return self.amount == other.amount and self.currency == other.currency

    def __str__(self):
        return f"{self.currency} {self.amount:0.2f}"


class Portfolio:
    def __init__(self):
        self.moneys = []

    def add(self, *moneys):
        self.moneys.extend(moneys)

    def evaluate(self, currency):
        total = 0.0
        failures = []
        for m in self.moneys:
            try:
                total += self.__convert(m, currency)
            except KeyError as ke:
                failures.append(ke)

        if len(failures) == 0:
            return Money(total, currency)

        failureMessage = ",".join(f.args[0] for f in failures)
        raise Exception("Missing exchange rate(s):[" + failureMessage + "]")

    def __convert(self, aMoney, aCurrency):
        exchangeRates = {'EUR->USD': 1.2, 'USD->KRW': 1100}
        if aMoney.currency == aCurrency:
            return aMoney.amount
        else:
            key = aMoney.currency + '->' + aCurrency
            return aMoney.amount * exchangeRates[key]

class Bank:
    def __init__(self):
        self.exchangeRates = {}

    def addExchangeRate(self, currencyFrom, currencyTo, rate):
        key = currencyFrom + "->" + currencyTo
        self.exchangeRates[key] = rate

    def convert(self, aMoney, aCurrency):
        if aMoney.currency == aCurrency:
            return Money(aMoney.amount, aCurrency)

        key = aMoney.currency + "->" + aCurrency
        if key in self.exchangeRates:
            return Money(aMoney.amount * self.exchangeRates[key], aCurrency)

        raise Exception("Failed")


class TestMoney(unittest.TestCase):
    def testMultiplication(self):
        fiveDollars = Money(5, "USD")
        tenDollars = Money(10, "USD")
        self.assertEqual(tenDollars, fiveDollars.times(2))

    def testDivision(self):
        originalMoney = Money(4002, "KRW")
        expectedMoneyAfterDivision = Money(1000.5, "KRW")
        self.assertEqual(expectedMoneyAfterDivision, originalMoney.divide(4))

    def testAddition(self):
        fiveDollars = Money(5, "USD")
        tenDollars = Money(10, "USD")
        fifteenDollars = Money(15, "USD")
        portfolio = Portfolio()
        portfolio.add(fiveDollars, tenDollars)
        self.assertEqual(fifteenDollars, portfolio.evaluate("USD"))

    def testAdditionOfDollarsAndEuros(self):
        fiveDollars = Money(5, "USD")
        tenEuros = Money(10, "EUR")
        portfolio = Portfolio()
        portfolio.add(fiveDollars, tenEuros)
        expectedValue = Money(17, "USD")
        actualValue = portfolio.evaluate("USD")
        self.assertEqual(expectedValue, actualValue,
                "%s != %s"%(expectedValue, actualValue))

    def testAdditionOfDollarsAndWons(self):
        oneDollar = Money(1, "USD")
        elevenHundredWon = Money(1100, "KRW")
        portfolio = Portfolio()
        portfolio.add(oneDollar, elevenHundredWon)
        expectedValue = Money(2200, "KRW")
        actualValue = portfolio.evaluate("KRW")
        self.assertEqual(expectedValue, actualValue,
        "%s != %s"%(expectedValue, actualValue))

    def testAdditionWithMultipleMissingExchangeRates(self):
        oneDollar = Money(1, "USD")
        oneEuro = Money(1, "EUR")
        oneWon = Money(1, "KRW")
        portfolio = Portfolio()
        portfolio.add(oneDollar, oneEuro, oneWon)
        with self.assertRaisesRegex(
            Exception,
            "Missing exchange rate\(s\):\[USD\->Kalganid,EUR->Kalganid,KRW->Kalganid]",
        ):
            portfolio.evaluate("Kalganid")

    def testConversion(self):
        bank = Bank()
        bank.addExchangeRate("EUR", "USD", 1.2)
        tenEuros = Money(10, "EUR")
        self.assertEqual(bank.convert(tenEuros, "USD"), Money(12, "USD"))


if __name__ == '__main__':
    unittest.main()
```

```python colab={"base_uri": "https://localhost:8080/"} id="tGQjiec9f3ms" executionInfo={"status": "ok", "timestamp": 1630149857143, "user_tz": -330, "elapsed": 1600, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="fd73c432-02c6-4457-f99e-5b142c726c52"
!python test_money.py -v
```

<!-- #region id="NhLm49VcgLyC" -->
We write a new test that expects an Exception with a specific message from the convert method:
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="Q5vqjM8ngZMf" executionInfo={"status": "ok", "timestamp": 1630149937881, "user_tz": -330, "elapsed": 465, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="cefbdcb5-93c6-422d-997a-c80952ea003c"
%%writefile test_money.py
import unittest
import functools
import operator


class Money:
    def __init__(self, amount, currency):
        self.amount = amount
        self.currency = currency
        
    def times(self, multiplier):
        return Money(self.amount * multiplier, self.currency)

    def divide(self, divisor):
        return Money(self.amount / divisor, self.currency)

    def __eq__(self, other):
        return self.amount == other.amount and self.currency == other.currency

    def __str__(self):
        return f"{self.currency} {self.amount:0.2f}"


class Portfolio:
    def __init__(self):
        self.moneys = []

    def add(self, *moneys):
        self.moneys.extend(moneys)

    def evaluate(self, currency):
        total = 0.0
        failures = []
        for m in self.moneys:
            try:
                total += self.__convert(m, currency)
            except KeyError as ke:
                failures.append(ke)

        if len(failures) == 0:
            return Money(total, currency)

        failureMessage = ",".join(f.args[0] for f in failures)
        raise Exception("Missing exchange rate(s):[" + failureMessage + "]")

    def __convert(self, aMoney, aCurrency):
        exchangeRates = {'EUR->USD': 1.2, 'USD->KRW': 1100}
        if aMoney.currency == aCurrency:
            return aMoney.amount
        else:
            key = aMoney.currency + '->' + aCurrency
            return aMoney.amount * exchangeRates[key]

class Bank:
    def __init__(self):
        self.exchangeRates = {}

    def addExchangeRate(self, currencyFrom, currencyTo, rate):
        key = currencyFrom + "->" + currencyTo
        self.exchangeRates[key] = rate

    def convert(self, aMoney, aCurrency):
        if aMoney.currency == aCurrency:
            return Money(aMoney.amount, aCurrency)

        key = aMoney.currency + "->" + aCurrency
        if key in self.exchangeRates:
            return Money(aMoney.amount * self.exchangeRates[key], aCurrency)

        raise Exception("Failed")


class TestMoney(unittest.TestCase):
    def testMultiplication(self):
        fiveDollars = Money(5, "USD")
        tenDollars = Money(10, "USD")
        self.assertEqual(tenDollars, fiveDollars.times(2))

    def testDivision(self):
        originalMoney = Money(4002, "KRW")
        expectedMoneyAfterDivision = Money(1000.5, "KRW")
        self.assertEqual(expectedMoneyAfterDivision, originalMoney.divide(4))

    def testAddition(self):
        fiveDollars = Money(5, "USD")
        tenDollars = Money(10, "USD")
        fifteenDollars = Money(15, "USD")
        portfolio = Portfolio()
        portfolio.add(fiveDollars, tenDollars)
        self.assertEqual(fifteenDollars, portfolio.evaluate("USD"))

    def testAdditionOfDollarsAndEuros(self):
        fiveDollars = Money(5, "USD")
        tenEuros = Money(10, "EUR")
        portfolio = Portfolio()
        portfolio.add(fiveDollars, tenEuros)
        expectedValue = Money(17, "USD")
        actualValue = portfolio.evaluate("USD")
        self.assertEqual(expectedValue, actualValue,
                "%s != %s"%(expectedValue, actualValue))

    def testAdditionOfDollarsAndWons(self):
        oneDollar = Money(1, "USD")
        elevenHundredWon = Money(1100, "KRW")
        portfolio = Portfolio()
        portfolio.add(oneDollar, elevenHundredWon)
        expectedValue = Money(2200, "KRW")
        actualValue = portfolio.evaluate("KRW")
        self.assertEqual(expectedValue, actualValue,
        "%s != %s"%(expectedValue, actualValue))

    def testAdditionWithMultipleMissingExchangeRates(self):
        oneDollar = Money(1, "USD")
        oneEuro = Money(1, "EUR")
        oneWon = Money(1, "KRW")
        portfolio = Portfolio()
        portfolio.add(oneDollar, oneEuro, oneWon)
        with self.assertRaisesRegex(
            Exception,
            "Missing exchange rate\(s\):\[USD\->Kalganid,EUR->Kalganid,KRW->Kalganid]",
        ):
            portfolio.evaluate("Kalganid")

    def testConversion(self):
        bank = Bank()
        bank.addExchangeRate("EUR", "USD", 1.2)
        tenEuros = Money(10, "EUR")
        self.assertEqual(bank.convert(tenEuros, "USD"), Money(12, "USD"))

    def testConversionWithMissingExchangeRate(self):
        bank = Bank()
        tenEuros = Money(10, "EUR")
        with self.assertRaisesRegex(Exception, "EUR->Kalganid"):
            bank.convert(tenEuros, "Kalganid")


if __name__ == '__main__':
    unittest.main()
```

```python colab={"base_uri": "https://localhost:8080/"} id="0PpXrXhSgZMh" executionInfo={"status": "ok", "timestamp": 1630149941057, "user_tz": -330, "elapsed": 1710, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="7e38524a-ce97-4089-ad29-1345ace11aae"
!python test_money.py -v
```

<!-- #region id="cRvgtkvSggJO" -->
To fix this, we use key to create the Exception that’s raised from convert:
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="eWt4011YgoeQ" executionInfo={"status": "ok", "timestamp": 1630150829104, "user_tz": -330, "elapsed": 1009, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="c050a1c9-6917-4240-f08a-74dfd356d563"
%%writefile test_money.py
import unittest
import functools
import operator


class Money:
    def __init__(self, amount, currency):
        self.amount = amount
        self.currency = currency
        
    def times(self, multiplier):
        return Money(self.amount * multiplier, self.currency)

    def divide(self, divisor):
        return Money(self.amount / divisor, self.currency)

    def __eq__(self, other):
        return self.amount == other.amount and self.currency == other.currency

    def __str__(self):
        return f"{self.currency} {self.amount:0.2f}"


class Portfolio:
    def __init__(self):
        self.moneys = []

    def add(self, *moneys):
        self.moneys.extend(moneys)

    def evaluate(self, currency):
        total = 0.0
        failures = []
        for m in self.moneys:
            try:
                total += self.__convert(m, currency)
            except KeyError as ke:
                failures.append(ke)

        if len(failures) == 0:
            return Money(total, currency)

        failureMessage = ",".join(f.args[0] for f in failures)
        raise Exception("Missing exchange rate(s):[" + failureMessage + "]")

    def __convert(self, aMoney, aCurrency):
        exchangeRates = {'EUR->USD': 1.2, 'USD->KRW': 1100}
        if aMoney.currency == aCurrency:
            return aMoney.amount
        else:
            key = aMoney.currency + '->' + aCurrency
            return aMoney.amount * exchangeRates[key]

class Bank:
    def __init__(self):
        self.exchangeRates = {}

    def addExchangeRate(self, currencyFrom, currencyTo, rate):
        key = currencyFrom + "->" + currencyTo
        self.exchangeRates[key] = rate

    def convert(self, aMoney, aCurrency):
        if aMoney.currency == aCurrency:
            return Money(aMoney.amount, aCurrency)

        key = aMoney.currency + "->" + aCurrency
        if key in self.exchangeRates:
            return Money(aMoney.amount * self.exchangeRates[key], aCurrency)

        raise Exception(key)


class TestMoney(unittest.TestCase):
    def testMultiplication(self):
        fiveDollars = Money(5, "USD")
        tenDollars = Money(10, "USD")
        self.assertEqual(tenDollars, fiveDollars.times(2))

    def testDivision(self):
        originalMoney = Money(4002, "KRW")
        expectedMoneyAfterDivision = Money(1000.5, "KRW")
        self.assertEqual(expectedMoneyAfterDivision, originalMoney.divide(4))

    def testAddition(self):
        fiveDollars = Money(5, "USD")
        tenDollars = Money(10, "USD")
        fifteenDollars = Money(15, "USD")
        portfolio = Portfolio()
        portfolio.add(fiveDollars, tenDollars)
        self.assertEqual(fifteenDollars, portfolio.evaluate("USD"))

    def testAdditionOfDollarsAndEuros(self):
        fiveDollars = Money(5, "USD")
        tenEuros = Money(10, "EUR")
        portfolio = Portfolio()
        portfolio.add(fiveDollars, tenEuros)
        expectedValue = Money(17, "USD")
        actualValue = portfolio.evaluate("USD")
        self.assertEqual(expectedValue, actualValue,
                "%s != %s"%(expectedValue, actualValue))

    def testAdditionOfDollarsAndWons(self):
        oneDollar = Money(1, "USD")
        elevenHundredWon = Money(1100, "KRW")
        portfolio = Portfolio()
        portfolio.add(oneDollar, elevenHundredWon)
        expectedValue = Money(2200, "KRW")
        actualValue = portfolio.evaluate("KRW")
        self.assertEqual(expectedValue, actualValue,
        "%s != %s"%(expectedValue, actualValue))

    def testAdditionWithMultipleMissingExchangeRates(self):
        oneDollar = Money(1, "USD")
        oneEuro = Money(1, "EUR")
        oneWon = Money(1, "KRW")
        portfolio = Portfolio()
        portfolio.add(oneDollar, oneEuro, oneWon)
        with self.assertRaisesRegex(
            Exception,
            "Missing exchange rate\(s\):\[USD\->Kalganid,EUR->Kalganid,KRW->Kalganid]",
        ):
            portfolio.evaluate("Kalganid")

    def testConversion(self):
        bank = Bank()
        bank.addExchangeRate("EUR", "USD", 1.2)
        tenEuros = Money(10, "EUR")
        self.assertEqual(bank.convert(tenEuros, "USD"), Money(12, "USD"))

    def testConversionWithMissingExchangeRate(self):
        bank = Bank()
        tenEuros = Money(10, "EUR")
        with self.assertRaisesRegex(Exception, "EUR->Kalganid"):
            bank.convert(tenEuros, "Kalganid")


if __name__ == '__main__':
    unittest.main()
```

```python colab={"base_uri": "https://localhost:8080/"} id="ZqAIkaZegoeS" executionInfo={"status": "ok", "timestamp": 1630150829105, "user_tz": -330, "elapsed": 23, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="d227af55-7b2c-40ae-c88f-54a080a4a83d"
!python test_money.py -v
```

<!-- #region id="wyxu5r9ug644" -->
All tests are green. With the new Bank class in place, we’re ready to change the evaluate method in Portfolio to accept a Bank object as a dependency. We have no fewer than four tests for addition of Money s which exercise the evaluate method. We fully expect these tests to fail, thereby keeping us firmly on the RGR track.
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="PdIywPbJhD6B" executionInfo={"status": "ok", "timestamp": 1630150165522, "user_tz": -330, "elapsed": 1215, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="db196650-600e-4c50-e7d8-309bc75b8abf"
%%writefile test_money.py
import unittest
import functools
import operator


class Money:
    def __init__(self, amount, currency):
        self.amount = amount
        self.currency = currency
        
    def times(self, multiplier):
        return Money(self.amount * multiplier, self.currency)

    def divide(self, divisor):
        return Money(self.amount / divisor, self.currency)

    def __eq__(self, other):
        return self.amount == other.amount and self.currency == other.currency

    def __str__(self):
        return f"{self.currency} {self.amount:0.2f}"


class Portfolio:
    def __init__(self):
        self.moneys = []

    def add(self, *moneys):
        self.moneys.extend(moneys)

    def evaluate(self, bank, currency):
        total = 0.0
        failures = []
        for m in self.moneys:
            try:
                total += self.convert(m, currency).amount
            except KeyError as ke:
                failures.append(ke)

        if len(failures) == 0:
            return Money(total, currency)

        failureMessage = ",".join(f.args[0] for f in failures)
        raise Exception("Missing exchange rate(s):[" + failureMessage + "]")
        

class Bank:
    def __init__(self):
        self.exchangeRates = {}

    def addExchangeRate(self, currencyFrom, currencyTo, rate):
        key = currencyFrom + "->" + currencyTo
        self.exchangeRates[key] = rate

    def convert(self, aMoney, aCurrency):
        if aMoney.currency == aCurrency:
            return Money(aMoney.amount, aCurrency)

        key = aMoney.currency + "->" + aCurrency
        if key in self.exchangeRates:
            return Money(aMoney.amount * self.exchangeRates[key], aCurrency)

        raise Exception(key)


class TestMoney(unittest.TestCase):
    def testMultiplication(self):
        fiveDollars = Money(5, "USD")
        tenDollars = Money(10, "USD")
        self.assertEqual(tenDollars, fiveDollars.times(2))

    def testDivision(self):
        originalMoney = Money(4002, "KRW")
        expectedMoneyAfterDivision = Money(1000.5, "KRW")
        self.assertEqual(expectedMoneyAfterDivision, originalMoney.divide(4))

    def testAddition(self):
        fiveDollars = Money(5, "USD")
        tenDollars = Money(10, "USD")
        fifteenDollars = Money(15, "USD")
        portfolio = Portfolio()
        portfolio.add(fiveDollars, tenDollars)
        self.assertEqual(fifteenDollars, portfolio.evaluate("USD"))

    def testAdditionOfDollarsAndEuros(self):
        fiveDollars = Money(5, "USD")
        tenEuros = Money(10, "EUR")
        portfolio = Portfolio()
        portfolio.add(fiveDollars, tenEuros)
        expectedValue = Money(17, "USD")
        actualValue = portfolio.evaluate("USD")
        self.assertEqual(expectedValue, actualValue,
                "%s != %s"%(expectedValue, actualValue))

    def testAdditionOfDollarsAndWons(self):
        oneDollar = Money(1, "USD")
        elevenHundredWon = Money(1100, "KRW")
        portfolio = Portfolio()
        portfolio.add(oneDollar, elevenHundredWon)
        expectedValue = Money(2200, "KRW")
        actualValue = portfolio.evaluate("KRW")
        self.assertEqual(expectedValue, actualValue,
        "%s != %s"%(expectedValue, actualValue))

    def testAdditionWithMultipleMissingExchangeRates(self):
        oneDollar = Money(1, "USD")
        oneEuro = Money(1, "EUR")
        oneWon = Money(1, "KRW")
        portfolio = Portfolio()
        portfolio.add(oneDollar, oneEuro, oneWon)
        with self.assertRaisesRegex(
            Exception,
            "Missing exchange rate\(s\):\[USD\->Kalganid,EUR->Kalganid,KRW->Kalganid]",
        ):
            portfolio.evaluate("Kalganid")

    def testConversion(self):
        bank = Bank()
        bank.addExchangeRate("EUR", "USD", 1.2)
        tenEuros = Money(10, "EUR")
        self.assertEqual(bank.convert(tenEuros, "USD"), Money(12, "USD"))

    def testConversionWithMissingExchangeRate(self):
        bank = Bank()
        tenEuros = Money(10, "EUR")
        with self.assertRaisesRegex(Exception, "EUR->Kalganid"):
            bank.convert(tenEuros, "Kalganid")


if __name__ == '__main__':
    unittest.main()
```

```python colab={"base_uri": "https://localhost:8080/"} id="_JAvnTFPhD6D" executionInfo={"status": "ok", "timestamp": 1630150165968, "user_tz": -330, "elapsed": 21, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="ff305449-282e-4983-e964-3361b94c148c"
!python test_money.py -v
```

<!-- #region id="cWG3DPbthXqH" -->
In the test methods, we can simply use self.bank as the first argument to each call to evaluate.

Also, it’d be nice if we could declare this initialization code once, rather than in each test. There is a way to do this. Our test class, by virtue of subclassing from unittest.TestCase, inherits its behavior. One aspect of this inherited behavior is that if there is a setUp method in the class, it’ll be called before each test. We can define our Bank object in this setUp method:
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="6jYqWDUOh3EJ" executionInfo={"status": "ok", "timestamp": 1630152162956, "user_tz": -330, "elapsed": 749, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="18058801-93f6-4efc-9c1a-296caf2c8d13"
%%writefile test_money.py
import unittest
import functools
import operator


class Money:
    def __init__(self, amount, currency):
        self.amount = amount
        self.currency = currency
        
    def times(self, multiplier):
        return Money(self.amount * multiplier, self.currency)

    def divide(self, divisor):
        return Money(self.amount / divisor, self.currency)

    def __eq__(self, other):
        return self.amount == other.amount and self.currency == other.currency

    def __str__(self):
        return f"{self.currency} {self.amount:0.2f}"


class Portfolio:
    def __init__(self):
        self.moneys = []

    def add(self, *moneys):
        self.moneys.extend(moneys)

    def evaluate(self, bank, currency):
        total = 0.0
        failures = []
        for m in self.moneys:
            try:
                total += bank.convert(m, currency).amount
            except Exception as ex:
                failures.append(ex)

        if len(failures) == 0:
            return Money(total, currency)

        failureMessage = ",".join(f.args[0] for f in failures)
        raise Exception("Missing exchange rate(s):[" + failureMessage + "]")
        

class Bank:
    def __init__(self):
        self.exchangeRates = {}

    def addExchangeRate(self, currencyFrom, currencyTo, rate):
        key = currencyFrom + "->" + currencyTo
        self.exchangeRates[key] = rate

    def convert(self, aMoney, aCurrency):
        if aMoney.currency == aCurrency:
            return Money(aMoney.amount, aCurrency)

        key = aMoney.currency + "->" + aCurrency
        if key in self.exchangeRates:
            return Money(aMoney.amount * self.exchangeRates[key], aCurrency)

        raise Exception(key)


class TestMoney(unittest.TestCase):
    def setUp(self):
        self.bank = Bank()
        self.bank.addExchangeRate("EUR", "USD", 1.2)
        self.bank.addExchangeRate("USD", "KRW", 1100)

    def testMultiplication(self):
        fiveDollars = Money(5, "USD")
        tenDollars = Money(10, "USD")
        self.assertEqual(tenDollars, fiveDollars.times(2))

    def testDivision(self):
        originalMoney = Money(4002, "KRW")
        expectedMoneyAfterDivision = Money(1000.5, "KRW")
        self.assertEqual(expectedMoneyAfterDivision, originalMoney.divide(4))

    def testAddition(self):
        fiveDollars = Money(5, "USD")
        tenDollars = Money(10, "USD")
        fifteenDollars = Money(15, "USD")
        portfolio = Portfolio()
        portfolio.add(fiveDollars, tenDollars)
        self.assertEqual(fifteenDollars, portfolio.evaluate(self.bank, "USD"))

    def testAdditionOfDollarsAndEuros(self):
        fiveDollars = Money(5, "USD")
        tenEuros = Money(10, "EUR")
        portfolio = Portfolio()
        portfolio.add(fiveDollars, tenEuros)
        expectedValue = Money(17, "USD")
        actualValue = portfolio.evaluate(self.bank, "USD")
        self.assertEqual(expectedValue, actualValue,
                "%s != %s"%(expectedValue, actualValue))

    def testAdditionOfDollarsAndWons(self):
        oneDollar = Money(1, "USD")
        elevenHundredWon = Money(1100, "KRW")
        portfolio = Portfolio()
        portfolio.add(oneDollar, elevenHundredWon)
        expectedValue = Money(2200, "KRW")
        actualValue = portfolio.evaluate(self.bank, "KRW")
        self.assertEqual(expectedValue, actualValue,
        "%s != %s"%(expectedValue, actualValue))

    def testAdditionWithMultipleMissingExchangeRates(self):
        oneDollar = Money(1, "USD")
        oneEuro = Money(1, "EUR")
        oneWon = Money(1, "KRW")
        portfolio = Portfolio()
        portfolio.add(oneDollar, oneEuro, oneWon)
        with self.assertRaisesRegex(
            Exception,
            "Missing exchange rate\(s\):\[USD\->Kalganid,EUR->Kalganid,KRW->Kalganid]",
        ):
            portfolio.evaluate(self.bank, "Kalganid")

    def testConversion(self):
        tenEuros = Money(10, "EUR")
        self.assertEqual(self.bank.convert(tenEuros, "USD"), Money(12, "USD"))

    def testConversionWithMissingExchangeRate(self):
        tenEuros = Money(10, "EUR")
        with self.assertRaisesRegex(Exception, "EUR->Kalganid"):
            self.bank.convert(tenEuros, "Kalganid")


if __name__ == '__main__':
    unittest.main()
```

```python colab={"base_uri": "https://localhost:8080/"} id="TYE7LGPBh3EL" executionInfo={"status": "ok", "timestamp": 1630152162958, "user_tz": -330, "elapsed": 22, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="c57d7260-6f66-4857-e2b2-23633e6b70d5"
!python test_money.py -v
```

<!-- #region id="TgMk6X_RinyA" -->
## Allow exchange rates to be modified
<!-- #endregion -->

<!-- #region id="2WUqULnFqbzq" -->
We start by adding a few lines to the end of testConversion. We’ll vary the exchange rate between EUR and USD to 1.3 and assert that this new rate is used for a second conversion between the two currencies. Here’s the test method in its entirety:
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="Q6Ura4-1qjsi" executionInfo={"status": "ok", "timestamp": 1630152611994, "user_tz": -330, "elapsed": 686, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="bdc80a84-d8e2-49d0-e872-636e22b2aacf"
%%writefile test_money.py
import unittest
import functools
import operator


class Money:
    def __init__(self, amount, currency):
        self.amount = amount
        self.currency = currency
        
    def times(self, multiplier):
        return Money(self.amount * multiplier, self.currency)

    def divide(self, divisor):
        return Money(self.amount / divisor, self.currency)

    def __eq__(self, other):
        return self.amount == other.amount and self.currency == other.currency

    def __str__(self):
        return f"{self.currency} {self.amount:0.2f}"


class Portfolio:
    def __init__(self):
        self.moneys = []

    def add(self, *moneys):
        self.moneys.extend(moneys)

    def evaluate(self, bank, currency):
        total = 0.0
        failures = []
        for m in self.moneys:
            try:
                total += bank.convert(m, currency).amount
            except Exception as ex:
                failures.append(ex)

        if len(failures) == 0:
            return Money(total, currency)

        failureMessage = ",".join(f.args[0] for f in failures)
        raise Exception("Missing exchange rate(s):[" + failureMessage + "]")
        

class Bank:
    def __init__(self):
        self.exchangeRates = {}

    def addExchangeRate(self, currencyFrom, currencyTo, rate):
        key = currencyFrom + "->" + currencyTo
        self.exchangeRates[key] = rate

    def convert(self, aMoney, aCurrency):
        if aMoney.currency == aCurrency:
            return Money(aMoney.amount, aCurrency)

        key = aMoney.currency + "->" + aCurrency
        if key in self.exchangeRates:
            return Money(aMoney.amount * self.exchangeRates[key], aCurrency)

        raise Exception(key)


class TestMoney(unittest.TestCase):
    def setUp(self):
        self.bank = Bank()
        self.bank.addExchangeRate("EUR", "USD", 1.2)
        self.bank.addExchangeRate("USD", "KRW", 1100)

    def testMultiplication(self):
        fiveDollars = Money(5, "USD")
        tenDollars = Money(10, "USD")
        self.assertEqual(tenDollars, fiveDollars.times(2))

    def testDivision(self):
        originalMoney = Money(4002, "KRW")
        expectedMoneyAfterDivision = Money(1000.5, "KRW")
        self.assertEqual(expectedMoneyAfterDivision, originalMoney.divide(4))

    def testAddition(self):
        fiveDollars = Money(5, "USD")
        tenDollars = Money(10, "USD")
        fifteenDollars = Money(15, "USD")
        portfolio = Portfolio()
        portfolio.add(fiveDollars, tenDollars)
        self.assertEqual(fifteenDollars, portfolio.evaluate(self.bank, "USD"))

    def testAdditionOfDollarsAndEuros(self):
        fiveDollars = Money(5, "USD")
        tenEuros = Money(10, "EUR")
        portfolio = Portfolio()
        portfolio.add(fiveDollars, tenEuros)
        expectedValue = Money(17, "USD")
        actualValue = portfolio.evaluate(self.bank, "USD")
        self.assertEqual(expectedValue, actualValue,
                "%s != %s"%(expectedValue, actualValue))

    def testAdditionOfDollarsAndWons(self):
        oneDollar = Money(1, "USD")
        elevenHundredWon = Money(1100, "KRW")
        portfolio = Portfolio()
        portfolio.add(oneDollar, elevenHundredWon)
        expectedValue = Money(2200, "KRW")
        actualValue = portfolio.evaluate(self.bank, "KRW")
        self.assertEqual(expectedValue, actualValue,
        "%s != %s"%(expectedValue, actualValue))

    def testAdditionWithMultipleMissingExchangeRates(self):
        oneDollar = Money(1, "USD")
        oneEuro = Money(1, "EUR")
        oneWon = Money(1, "KRW")
        portfolio = Portfolio()
        portfolio.add(oneDollar, oneEuro, oneWon)
        with self.assertRaisesRegex(
            Exception,
            "Missing exchange rate\(s\):\[USD\->Kalganid,EUR->Kalganid,KRW->Kalganid]",
        ):
            portfolio.evaluate(self.bank, "Kalganid")

    def testConversion(self):
        tenEuros = Money(10, "EUR")
        self.assertEqual(self.bank.convert(tenEuros, "USD"), Money(12, "USD"))
        self.bank.addExchangeRate("EUR", "USD", 1.3)
        self.assertEqual(self.bank.convert(tenEuros, "USD"), Money(13, "USD"))

    def testConversionWithMissingExchangeRate(self):
        tenEuros = Money(10, "EUR")
        with self.assertRaisesRegex(Exception, "EUR->Kalganid"):
            self.bank.convert(tenEuros, "Kalganid")


if __name__ == '__main__':
    unittest.main()
```

```python colab={"base_uri": "https://localhost:8080/"} id="GmqLVemRqjsk" executionInfo={"status": "ok", "timestamp": 1630152612822, "user_tz": -330, "elapsed": 28, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="82096580-5eff-4a07-8689-a41051fe7a53"
!python test_money.py -v
```

<!-- #region id="0r-SWYTuscud" -->
## CI with Github workflows
<!-- #endregion -->

```python id="2gWYspZusoKW"
!mkdir -p .github/workflows
```

```python colab={"base_uri": "https://localhost:8080/"} id="bfp1fLpPsfTl" executionInfo={"status": "ok", "timestamp": 1630153179196, "user_tz": -330, "elapsed": 447, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="2314001d-6b88-46af-b81e-58caf0c6ee5a"
%%writefile .github/workflows/py.yml
name: Python CI
on:
  push:
    branches: [ main ]
jobs:
  build:
    name: Build
    strategy:
      matrix:
        python-version: [3.9.x, 3.8.x]
        platform: [ubuntu-latest, macos-latest, windows-latest]
    runs-on: ${{matrix.platform}}
    steps:
    - name: Set up Python ${{ matrix.python-version }}
      uses: actions/setup-python@v2
      with:
        python-version: ${{ matrix.python-version }}
    - name: Checkout code
      uses: actions/checkout@v2
    - name: Test
      run: python test_money.py -v
      shell: bash
```

```python id="a5cp2GbSs3oG"

```
