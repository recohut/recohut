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

<!-- #region id="fvGSDj8S1dyw" -->
## Acceptance Testing
<!-- #endregion -->

<!-- #region id="7Pstcfzczz1c" -->
This is where acceptance testing comes in. Acceptance tests check that you are building the correct product. While unit tests and integration tests are a form of verification, acceptance tests are validation. They validate that you are building what the user expects.
<!-- #endregion -->

<!-- #region id="rS9Zs3Ir0B5U" -->
### Behavior-Driven Development
<!-- #endregion -->

<!-- #region id="aNzKo3sQ0UxH" -->
Behavior-driven development, first pioneered by Daniel Terhorst-North, is a practice that focuses on defining the behaviors in your system. BDD focuses on clarifying communications; you iterate over the requirements with the end user, defining the behaviors they want.

Before you write a single lick of code, you make sure that you have agreement on what the right thing to build is. The set of defined behaviors will drive what code you write. You work with the end user (or their proxy, such as a business analyst or product manager) to define your requirements as a specification. These specifications follow a formal language, to introduce a bit more rigidity in their definition. One of the most common languages for specifying requirements is Gherkin.

Gherkin is a specification that follows the Given-When-Then (GWT) format. Every requirement is organized as follows:
<!-- #endregion -->

```python id="vpwE3Wxw0IUA"
Feature: Name of test suite

  Scenario: A test case
    Given some precondition
    When I take some action
    Then I expect this result
```

<!-- #region id="Yc6JtYX00wQO" -->
For instance, if I wanted to capture a requirement that checks for vegan substitution of a dish, I would write it as follows:
<!-- #endregion -->

```python id="nGGSIEcH0hyx"
Feature: Vegan-friendly menu

  Scenario: Can substitute for vegan alternative
    Given an order containing a Cheeseburger with Fries
    When I ask for vegan substitutions
    Then I receive the meal with no animal products
```

<!-- #region id="_-06S_Mf06u8" -->
Another requirement might be that certain dishes can’t be made vegan:
<!-- #endregion -->

```python id="Us2T6jEY0798"
  Scenario: Cannot substitute vegan alternatives for certain meals
    Given an order containing Meatloaf
    When I ask for vegan substitutions
    Then an error shows up stating the meal is not vegan substitutable
```

<!-- #region id="nNUcW4qS1CrJ" -->
With behave, I can write Python code that maps to each of these GWT statements:
<!-- #endregion -->
