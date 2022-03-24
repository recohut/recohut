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

<!-- #region id="TbNwA9zx08_-" -->
# Linear Optimization with OR-Tools
<!-- #endregion -->

```python id="5arhRP8Hp7Lx"
!pip install ortools
```

<!-- #region id="yO4JhlNbvc3b" -->
## Solving an LP Problem

The following sections present an example of an LP problem and show how to solve it. Here's the problem:

Maximize 3x + 4y subject to the following constraints:
- x + 2y	≤	14
- 3x – y	≥	0
- x – y	≤	2


Both the objective function, 3x + 4y, and the constraints are given by linear expressions, which makes this a linear problem.

The constraints define the feasible region, which is the triangle shown below, including its interior.

<img src='https://developers.google.com/optimization/images/lp/feasible_region.png'>

Basic steps for solving an LP problem
To solve a LP problem, your program should include the following steps :

- Import the linear solver wrapper,
- declare the LP solver,
- define the variables,
- define the constraints,
- define the objective,
- call the LP solver; and
- display the solution
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="pY17nxs1uzmf" executionInfo={"status": "ok", "timestamp": 1634717214336, "user_tz": -330, "elapsed": 523, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="af35d3d4-c5c1-4954-9957-398c397de6b0"
from ortools.linear_solver import pywraplp


def LinearProgrammingExample():
    """Linear programming sample."""
    # Instantiate a Glop solver, naming it LinearExample.
    solver = pywraplp.Solver.CreateSolver('GLOP')

    # Create the two variables and let them take on any non-negative value.
    x = solver.NumVar(0, solver.infinity(), 'x')
    y = solver.NumVar(0, solver.infinity(), 'y')

    print('Number of variables =', solver.NumVariables())

    # Constraint 0: x + 2y <= 14.
    solver.Add(x + 2 * y <= 14.0)

    # Constraint 1: 3x - y >= 0.
    solver.Add(3 * x - y >= 0.0)

    # Constraint 2: x - y <= 2.
    solver.Add(x - y <= 2.0)

    print('Number of constraints =', solver.NumConstraints())

    # Objective function: 3x + 4y.
    solver.Maximize(3 * x + 4 * y)

    # Solve the system.
    status = solver.Solve()

    if status == pywraplp.Solver.OPTIMAL:
        print('Solution:')
        print('Objective value =', solver.Objective().Value())
        print('x =', x.solution_value())
        print('y =', y.solution_value())
    else:
        print('The problem does not have an optimal solution.')

    print('\nAdvanced usage:')
    print('Problem solved in %f milliseconds' % solver.wall_time())
    print('Problem solved in %d iterations' % solver.iterations())


LinearProgrammingExample()
```

<!-- #region id="pK1iSpRmvBfS" -->
Here is a graph showing the solution:

<img src='https://developers.google.com/optimization/images/lp/feasible_region_solution.png'>

The dashed green line is defined by setting the objective function equal to its optimal value of 34. Any line whose equation has the form 3x + 4y = c is parallel to the dashed line, and 34 is the largest value of c for which the line intersects the feasible region.

If you think about the geometry in the above graph, in any linear optimization problem at least one vertex of the feasible region must be an optimal solution. As a result, you can find an optimal solution by traversing the vertices of the feasible region until there is no more improvement in the objective function. This is the idea behind simplex algorithm, the most widely-used method for solving linear optimization problems.
<!-- #endregion -->

<!-- #region id="aMg5IfCDwLCW" -->
## The Stigler Diet Problem

In this section, we show how to solve a classic problem called the Stigler diet, named for economics Nobel laureate George Stigler, who computed an inexpensive way to fulfill basic nutritional needs given a set of foods. He posed this as a mathematical exercise, not as eating recommendations, although the notion of computing optimal nutrition has of come into vogue recently.

The Stigler diet mandated that these minimums be met:

| Nutrient |	Daily Recommended Intake |
| -------- | --------------------------- |
| Calories |	3,000 Calories |
| Protein |	70 grams |
| Calcium |	.8 grams |
| Iron |	12 milligrams |
| Vitamin A |	5,000 IU |
| Thiamine (Vitamin B1)	| 1.8 milligrams |
| Riboflavin (Vitamin B2) |	2.7 milligrams |
| Niacin |	18 milligrams |
| Ascorbic Acid (Vitamin C)	| 75 milligrams |

The set of foods Stigler evaluated was a reflection of the time (1944). The nutritional data below is per dollar, not per unit, so the objective is to determine how many dollars to spend on each foodstuff.

**Commodities list**

| Commodity               | Unit     | 1939 price (cents) | Calories (kcal) | Protein (g) | Calcium (g) | Iron (mg) | Vitamin A (KIU) | Thiamine (mg) | Riboflavin (mg) | Niacin (mg) | Ascorbic Acid (mg) |
| ----------------------- | -------- | ------------------ | --------------- | ----------- | ----------- | --------- | --------------- | ------------- | --------------- | ----------- | ------------------ |
| Wheat Flour (Enriched)  | 10 lb.   | 36                 | 44.7            | 1411        | 2           | 365       | 0               | 55.4          | 33.3            | 441         | 0                  |
| Macaroni                | 1 lb.    | 14.1               | 11.6            | 418         | 0.7         | 54        | 0               | 3.2           | 1.9             | 68          | 0                  |
| Wheat Cereal (Enriched) | 28 oz.   | 24.2               | 11.8            | 377         | 14.4        | 175       | 0               | 14.4          | 8.8             | 114         | 0                  |
| Corn Flakes             | 8 oz.    | 7.1                | 11.4            | 252         | 0.1         | 56        | 0               | 13.5          | 2.3             | 68          | 0                  |
| Corn Meal               | 1 lb.    | 4.6                | 36.0            | 897         | 1.7         | 99        | 30.9            | 17.4          | 7.9             | 106         | 0                  |
| Hominy Grits            | 24 oz.   | 8.5                | 28.6            | 680         | 0.8         | 80        | 0               | 10.6          | 1.6             | 110         | 0                  |
| Rice                    | 1 lb.    | 7.5                | 21.2            | 460         | 0.6         | 41        | 0               | 2             | 4.8             | 60          | 0                  |
| Rolled Oats             | 1 lb.    | 7.1                | 25.3            | 907         | 5.1         | 341       | 0               | 37.1          | 8.9             | 64          | 0                  |
| White Bread (Enriched)  | 1 lb.    | 7.9                | 15.0            | 488         | 2.5         | 115       | 0               | 13.8          | 8.5             | 126         | 0                  |
| Whole Wheat Bread       | 1 lb.    | 9.1                | 12.2            | 484         | 2.7         | 125       | 0               | 13.9          | 6.4             | 160         | 0                  |
| Rye Bread               | 1 lb.    | 9.1                | 12.4            | 439         | 1.1         | 82        | 0               | 9.9           | 3               | 66          | 0                  |
| Pound Cake              | 1 lb.    | 24.8               | 8.0             | 130         | 0.4         | 31        | 18.9            | 2.8           | 3               | 17          | 0                  |
| Soda Crackers           | 1 lb.    | 15.1               | 12.5            | 288         | 0.5         | 50        | 0               | 0             | 0               | 0           | 0                  |
| Milk                    | 1 qt.    | 11                 | 6.1             | 310         | 10.5        | 18        | 16.8            | 4             | 16              | 7           | 177                |
| Evaporated Milk (can)   | 14.5 oz. | 6.7                | 8.4             | 422         | 15.1        | 9         | 26              | 3             | 23.5            | 11          | 60                 |
| Butter                  | 1 lb.    | 30.8               | 10.8            | 9           | 0.2         | 3         | 44.2            | 0             | 0.2             | 2           | 0                  |
| Oleomargarine           | 1 lb.    | 16.1               | 20.6            | 17          | 0.6         | 6         | 55.8            | 0.2           | 0               | 0           | 0                  |
| Eggs                    | 1 doz.   | 32.6               | 2.9             | 238         | 1.0         | 52        | 18.6            | 2.8           | 6.5             | 1           | 0                  |
| Cheese (Cheddar)        | 1 lb.    | 24.2               | 7.4             | 448         | 16.4        | 19        | 28.1            | 0.8           | 10.3            | 4           | 0                  |
| Cream                   | 1/2 pt.  | 14.1               | 3.5             | 49          | 1.7         | 3         | 16.9            | 0.6           | 2.5             | 0           | 17                 |
| Peanut Butter           | 1 lb.    | 17.9               | 15.7            | 661         | 1.0         | 48        | 0               | 9.6           | 8.1             | 471         | 0                  |
| Mayonnaise              | 1/2 pt.  | 16.7               | 8.6             | 18          | 0.2         | 8         | 2.7             | 0.4           | 0.5             | 0           | 0                  |
| Crisco                  | 1 lb.    | 20.3               | 20.1            | 0           | 0           | 0         | 0               | 0             | 0               | 0           | 0                  |
| Lard                    | 1 lb.    | 9.8                | 41.7            | 0           | 0           | 0         | 0.2             | 0             | 0.5             | 5           | 0                  |
| Sirloin Steak           | 1 lb.    | 39.6               | 2.9             | 166         | 0.1         | 34        | 0.2             | 2.1           | 2.9             | 69          | 0                  |
| Round Steak             | 1 lb.    | 36.4               | 2.2             | 214         | 0.1         | 32        | 0.4             | 2.5           | 2.4             | 87          | 0                  |
| Rib Roast               | 1 lb.    | 29.2               | 3.4             | 213         | 0.1         | 33        | 0               | 0             | 2               | 0           | 0                  |
| Chuck Roast             | 1 lb.    | 22.6               | 3.6             | 309         | 0.2         | 46        | 0.4             | 1             | 4               | 120         | 0                  |
| Plate                | 1 lb.      | 14.6 | 8.5  | 404  | 0.2  | 62  | 0     | 0.9  | 0    | 0   | 0    |
| Liver (Beef)         | 1 lb.      | 26.8 | 2.2  | 333  | 0.2  | 139 | 169.2 | 6.4  | 50.8 | 316 | 525  |
| Leg of Lamb          | 1 lb.      | 27.6 | 3.1  | 245  | 0.1  | 20  | 0     | 2.8  | 3.9  | 86  | 0    |
| Lamb Chops (Rib)     | 1 lb.      | 36.6 | 3.3  | 140  | 0.1  | 15  | 0     | 1.7  | 2.7  | 54  | 0    |
| Pork Chops           | 1 lb.      | 30.7 | 3.5  | 196  | 0.2  | 30  | 0     | 17.4 | 2.7  | 60  | 0    |
| Pork Loin Roast      | 1 lb.      | 24.2 | 4.4  | 249  | 0.3  | 37  | 0     | 18.2 | 3.6  | 79  | 0    |
| Bacon                | 1 lb.      | 25.6 | 10.4 | 152  | 0.2  | 23  | 0     | 1.8  | 1.8  | 71  | 0    |
| Ham, smoked          | 1 lb.      | 27.4 | 6.7  | 212  | 0.2  | 31  | 0     | 9.9  | 3.3  | 50  | 0    |
| Salt Pork            | 1 lb.      | 16   | 18.8 | 164  | 0.1  | 26  | 0     | 1.4  | 1.8  | 0   | 0    |
| Roasting Chicken     | 1 lb.      | 30.3 | 1.8  | 184  | 0.1  | 30  | 0.1   | 0.9  | 1.8  | 68  | 46   |
| Veal Cutlets         | 1 lb.      | 42.3 | 1.7  | 156  | 0.1  | 24  | 0     | 1.4  | 2.4  | 57  | 0    |
| Salmon, Pink (can)   | 16 oz.     | 13   | 5.8  | 705  | 6.8  | 45  | 3.5   | 1    | 4.9  | 209 | 0    |
| Apples               | 1 lb.      | 4.4  | 5.8  | 27   | 0.5  | 36  | 7.3   | 3.6  | 2.7  | 5   | 544  |
| Bananas              | 1 lb.      | 6.1  | 4.9  | 60   | 0.4  | 30  | 17.4  | 2.5  | 3.5  | 28  | 498  |
| Lemons               | 1 doz.     | 26   | 1.0  | 21   | 0.5  | 14  | 0     | 0.5  | 0    | 4   | 952  |
| Oranges              | 1 doz.     | 30.9 | 2.2  | 40   | 1.1  | 18  | 11.1  | 3.6  | 1.3  | 10  | 1998 |
| Green Beans          | 1 lb.      | 7.1  | 2.4  | 138  | 3.7  | 80  | 69    | 4.3  | 5.8  | 37  | 862  |
| Cabbage              | 1 lb.      | 3.7  | 2.6  | 125  | 4.0  | 36  | 7.2   | 9    | 4.5  | 26  | 5369 |
| Carrots              | 1 bunch    | 4.7  | 2.7  | 73   | 2.8  | 43  | 188.5 | 6.1  | 4.3  | 89  | 608  |
| Celery               | 1 stalk    | 7.3  | 0.9  | 51   | 3.0  | 23  | 0.9   | 1.4  | 1.4  | 9   | 313  |
| Lettuce              | 1 head     | 8.2  | 0.4  | 27   | 1.1  | 22  | 112.4 | 1.8  | 3.4  | 11  | 449  |
| Onions               | 1 lb.      | 3.6  | 5.8  | 166  | 3.8  | 59  | 16.6  | 4.7  | 5.9  | 21  | 1184 |
| Potatoes             | 15 lb.     | 34   | 14.3 | 336  | 1.8  | 118 | 6.7   | 29.4 | 7.1  | 198 | 2522 |
| Spinach              | 1 lb.      | 8.1  | 1.1  | 106  | 0    | 138 | 918.4 | 5.7  | 13.8 | 33  | 2755 |
| Sweet Potatoes       | 1 lb.      | 5.1  | 9.6  | 138  | 2.7  | 54  | 290.7 | 8.4  | 5.4  | 83  | 1912 |
| Peaches (can)        | No. 2 1/2  | 16.8 | 3.7  | 20   | 0.4  | 10  | 21.5  | 0.5  | 1    | 31  | 196  |
| Pears (can)          | No. 2 1/2  | 20.4 | 3.0  | 8    | 0.3  | 8   | 0.8   | 0.8  | 0.8  | 5   | 81   |
| Pineapple (can)      | No. 2 1/2  | 21.3 | 2.4  | 16   | 0.4  | 8   | 2     | 2.8  | 0.8  | 7   | 399  |
| Asparagus (can)      | No. 2      | 27.7 | 0.4  | 33   | 0.3  | 12  | 16.3  | 1.4  | 2.1  | 17  | 272  |
| Green Beans (can)    | No. 2      | 10   | 1.0  | 54   | 2    | 65  | 53.9  | 1.6  | 4.3  | 32  | 431  |
| Pork and Beans (can) | 16 oz.     | 7.1  | 7.5  | 364  | 4    | 134 | 3.5   | 8.3  | 7.7  | 56  | 0    |
| Corn (can)           | No. 2      | 10.4 | 5.2  | 136  | 0.2  | 16  | 12    | 1.6  | 2.7  | 42  | 218  |
| Peas (can)           | No. 2      | 13.8 | 2.3  | 136  | 0.6  | 45  | 34.9  | 4.9  | 2.5  | 37  | 370  |
| Tomatoes (can)       | No. 2      | 8.6  | 1.3  | 63   | 0.7  | 38  | 53.2  | 3.4  | 2.5  | 36  | 1253 |
| Tomato Soup (can)    | 10 1/2 oz. | 7.6  | 1.6  | 71   | 0.6  | 43  | 57.9  | 3.5  | 2.4  | 67  | 862  |
| Peaches, Dried       | 1 lb.      | 15.7 | 8.5  | 87   | 1.7  | 173 | 86.8  | 1.2  | 4.3  | 55  | 57   |
| Prunes, Dried        | 1 lb.      | 9    | 12.8 | 99   | 2.5  | 154 | 85.7  | 3.9  | 4.3  | 65  | 257  |
| Raisins, Dried       | 15 oz.     | 9.4  | 13.5 | 104  | 2.5  | 136 | 4.5   | 6.3  | 1.4  | 24  | 136  |
| Peas, Dried          | 1 lb.      | 7.9  | 20.0 | 1367 | 4.2  | 345 | 2.9   | 28.7 | 18.4 | 162 | 0    |
| Lima Beans, Dried    | 1 lb.      | 8.9  | 17.4 | 1055 | 3.7  | 459 | 5.1   | 26.9 | 38.2 | 93  | 0    |
| Navy Beans, Dried    | 1 lb.      | 5.9  | 26.9 | 1691 | 11.4 | 792 | 0     | 38.4 | 24.6 | 217 | 0    |
| Coffee               | 1 lb.      | 22.4 | 0    | 0    | 0    | 0   | 0     | 4    | 5.1  | 50  | 0    |
| Tea                  | 1/4 lb.    | 17.4 | 0    | 0    | 0    | 0   | 0     | 0    | 2.3  | 42  | 0    |
| Cocoa                | 8 oz.      | 8.6  | 8.7  | 237  | 3    | 72  | 0     | 2    | 11.9 | 40  | 0    |
| Chocolate            | 8 oz.      | 16.2 | 8.0  | 77   | 1.3  | 39  | 0     | 0.9  | 3.4  | 14  | 0    |
| Sugar                | 10 lb.     | 51.7 | 34.9 | 0    | 0    | 0   | 0     | 0    | 0    | 0   | 0    |
| Corn Syrup           | 24 oz.     | 13.7 | 14.7 | 0    | 0.5  | 74  | 0     | 0    | 0    | 5   | 0    |
| Molasses             | 18 oz.     | 13.6 | 9.0  | 0    | 10.3 | 244 | 0     | 1.9  | 7.5  | 146 | 0    |
| Strawberry Preserves | 1 lb.      | 20.5 | 6.4  | 11   | 0.4  | 7   | 0.2   | 0.2  | 0.4  | 3   | 0    |
<!-- #endregion -->

<!-- #region id="I96hBu9yybU8" -->
Since the nutrients have all been normalized by price, our objective is simply minimizing the sum of foods.

In 1944, Stigler calculated the best answer he could, noting with sadness:

"...there does not appear to be any direct method of finding the minimum of a linear function subject to linear conditions."

He found a diet that cost $39.93 per year, in 1939 dollars. In 1947, Jack Laderman used the simplex method (then, a recent invention!) to determine the optimal solution. It took 120 man days of nine clerks on desk calculators to arrive at the answer.

**Solution using the linear solver**

The following sections present a program that solves the Stigler diet problem.
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="LrHQ9eJqyuku" executionInfo={"status": "ok", "timestamp": 1634718152111, "user_tz": -330, "elapsed": 1341, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="5cd7b30a-d7d4-43eb-d629-e6983de279f3"
"""The Stigler diet problem.

A description of the problem can be found here:
https://en.wikipedia.org/wiki/Stigler_diet.
"""
from ortools.linear_solver import pywraplp


def main():
    """Entry point of the program."""
    # Instantiate the data problem.
    # Nutrient minimums.
    nutrients = [
        ['Calories (kcal)', 3],
        ['Protein (g)', 70],
        ['Calcium (g)', 0.8],
        ['Iron (mg)', 12],
        ['Vitamin A (KIU)', 5],
        ['Vitamin B1 (mg)', 1.8],
        ['Vitamin B2 (mg)', 2.7],
        ['Niacin (mg)', 18],
        ['Vitamin C (mg)', 75],
    ]

    # Commodity, Unit, 1939 price (cents), Calories (kcal), Protein (g),
    # Calcium (g), Iron (mg), Vitamin A (KIU), Vitamin B1 (mg), Vitamin B2 (mg),
    # Niacin (mg), Vitamin C (mg)
    data = [
        [
            'Wheat Flour (Enriched)', '10 lb.', 36, 44.7, 1411, 2, 365, 0, 55.4,
            33.3, 441, 0
        ],
        ['Macaroni', '1 lb.', 14.1, 11.6, 418, 0.7, 54, 0, 3.2, 1.9, 68, 0],
        [
            'Wheat Cereal (Enriched)', '28 oz.', 24.2, 11.8, 377, 14.4, 175, 0,
            14.4, 8.8, 114, 0
        ],
        ['Corn Flakes', '8 oz.', 7.1, 11.4, 252, 0.1, 56, 0, 13.5, 2.3, 68, 0],
        [
            'Corn Meal', '1 lb.', 4.6, 36.0, 897, 1.7, 99, 30.9, 17.4, 7.9, 106,
            0
        ],
        [
            'Hominy Grits', '24 oz.', 8.5, 28.6, 680, 0.8, 80, 0, 10.6, 1.6,
            110, 0
        ],
        ['Rice', '1 lb.', 7.5, 21.2, 460, 0.6, 41, 0, 2, 4.8, 60, 0],
        ['Rolled Oats', '1 lb.', 7.1, 25.3, 907, 5.1, 341, 0, 37.1, 8.9, 64, 0],
        [
            'White Bread (Enriched)', '1 lb.', 7.9, 15.0, 488, 2.5, 115, 0,
            13.8, 8.5, 126, 0
        ],
        [
            'Whole Wheat Bread', '1 lb.', 9.1, 12.2, 484, 2.7, 125, 0, 13.9,
            6.4, 160, 0
        ],
        ['Rye Bread', '1 lb.', 9.1, 12.4, 439, 1.1, 82, 0, 9.9, 3, 66, 0],
        ['Pound Cake', '1 lb.', 24.8, 8.0, 130, 0.4, 31, 18.9, 2.8, 3, 17, 0],
        ['Soda Crackers', '1 lb.', 15.1, 12.5, 288, 0.5, 50, 0, 0, 0, 0, 0],
        ['Milk', '1 qt.', 11, 6.1, 310, 10.5, 18, 16.8, 4, 16, 7, 177],
        [
            'Evaporated Milk (can)', '14.5 oz.', 6.7, 8.4, 422, 15.1, 9, 26, 3,
            23.5, 11, 60
        ],
        ['Butter', '1 lb.', 30.8, 10.8, 9, 0.2, 3, 44.2, 0, 0.2, 2, 0],
        ['Oleomargarine', '1 lb.', 16.1, 20.6, 17, 0.6, 6, 55.8, 0.2, 0, 0, 0],
        ['Eggs', '1 doz.', 32.6, 2.9, 238, 1.0, 52, 18.6, 2.8, 6.5, 1, 0],
        [
            'Cheese (Cheddar)', '1 lb.', 24.2, 7.4, 448, 16.4, 19, 28.1, 0.8,
            10.3, 4, 0
        ],
        ['Cream', '1/2 pt.', 14.1, 3.5, 49, 1.7, 3, 16.9, 0.6, 2.5, 0, 17],
        [
            'Peanut Butter', '1 lb.', 17.9, 15.7, 661, 1.0, 48, 0, 9.6, 8.1,
            471, 0
        ],
        ['Mayonnaise', '1/2 pt.', 16.7, 8.6, 18, 0.2, 8, 2.7, 0.4, 0.5, 0, 0],
        ['Crisco', '1 lb.', 20.3, 20.1, 0, 0, 0, 0, 0, 0, 0, 0],
        ['Lard', '1 lb.', 9.8, 41.7, 0, 0, 0, 0.2, 0, 0.5, 5, 0],
        [
            'Sirloin Steak', '1 lb.', 39.6, 2.9, 166, 0.1, 34, 0.2, 2.1, 2.9,
            69, 0
        ],
        ['Round Steak', '1 lb.', 36.4, 2.2, 214, 0.1, 32, 0.4, 2.5, 2.4, 87, 0],
        ['Rib Roast', '1 lb.', 29.2, 3.4, 213, 0.1, 33, 0, 0, 2, 0, 0],
        ['Chuck Roast', '1 lb.', 22.6, 3.6, 309, 0.2, 46, 0.4, 1, 4, 120, 0],
        ['Plate', '1 lb.', 14.6, 8.5, 404, 0.2, 62, 0, 0.9, 0, 0, 0],
        [
            'Liver (Beef)', '1 lb.', 26.8, 2.2, 333, 0.2, 139, 169.2, 6.4, 50.8,
            316, 525
        ],
        ['Leg of Lamb', '1 lb.', 27.6, 3.1, 245, 0.1, 20, 0, 2.8, 3.9, 86, 0],
        [
            'Lamb Chops (Rib)', '1 lb.', 36.6, 3.3, 140, 0.1, 15, 0, 1.7, 2.7,
            54, 0
        ],
        ['Pork Chops', '1 lb.', 30.7, 3.5, 196, 0.2, 30, 0, 17.4, 2.7, 60, 0],
        [
            'Pork Loin Roast', '1 lb.', 24.2, 4.4, 249, 0.3, 37, 0, 18.2, 3.6,
            79, 0
        ],
        ['Bacon', '1 lb.', 25.6, 10.4, 152, 0.2, 23, 0, 1.8, 1.8, 71, 0],
        ['Ham, smoked', '1 lb.', 27.4, 6.7, 212, 0.2, 31, 0, 9.9, 3.3, 50, 0],
        ['Salt Pork', '1 lb.', 16, 18.8, 164, 0.1, 26, 0, 1.4, 1.8, 0, 0],
        [
            'Roasting Chicken', '1 lb.', 30.3, 1.8, 184, 0.1, 30, 0.1, 0.9, 1.8,
            68, 46
        ],
        ['Veal Cutlets', '1 lb.', 42.3, 1.7, 156, 0.1, 24, 0, 1.4, 2.4, 57, 0],
        [
            'Salmon, Pink (can)', '16 oz.', 13, 5.8, 705, 6.8, 45, 3.5, 1, 4.9,
            209, 0
        ],
        ['Apples', '1 lb.', 4.4, 5.8, 27, 0.5, 36, 7.3, 3.6, 2.7, 5, 544],
        ['Bananas', '1 lb.', 6.1, 4.9, 60, 0.4, 30, 17.4, 2.5, 3.5, 28, 498],
        ['Lemons', '1 doz.', 26, 1.0, 21, 0.5, 14, 0, 0.5, 0, 4, 952],
        ['Oranges', '1 doz.', 30.9, 2.2, 40, 1.1, 18, 11.1, 3.6, 1.3, 10, 1998],
        ['Green Beans', '1 lb.', 7.1, 2.4, 138, 3.7, 80, 69, 4.3, 5.8, 37, 862],
        ['Cabbage', '1 lb.', 3.7, 2.6, 125, 4.0, 36, 7.2, 9, 4.5, 26, 5369],
        ['Carrots', '1 bunch', 4.7, 2.7, 73, 2.8, 43, 188.5, 6.1, 4.3, 89, 608],
        ['Celery', '1 stalk', 7.3, 0.9, 51, 3.0, 23, 0.9, 1.4, 1.4, 9, 313],
        ['Lettuce', '1 head', 8.2, 0.4, 27, 1.1, 22, 112.4, 1.8, 3.4, 11, 449],
        ['Onions', '1 lb.', 3.6, 5.8, 166, 3.8, 59, 16.6, 4.7, 5.9, 21, 1184],
        [
            'Potatoes', '15 lb.', 34, 14.3, 336, 1.8, 118, 6.7, 29.4, 7.1, 198,
            2522
        ],
        ['Spinach', '1 lb.', 8.1, 1.1, 106, 0, 138, 918.4, 5.7, 13.8, 33, 2755],
        [
            'Sweet Potatoes', '1 lb.', 5.1, 9.6, 138, 2.7, 54, 290.7, 8.4, 5.4,
            83, 1912
        ],
        [
            'Peaches (can)', 'No. 2 1/2', 16.8, 3.7, 20, 0.4, 10, 21.5, 0.5, 1,
            31, 196
        ],
        [
            'Pears (can)', 'No. 2 1/2', 20.4, 3.0, 8, 0.3, 8, 0.8, 0.8, 0.8, 5,
            81
        ],
        [
            'Pineapple (can)', 'No. 2 1/2', 21.3, 2.4, 16, 0.4, 8, 2, 2.8, 0.8,
            7, 399
        ],
        [
            'Asparagus (can)', 'No. 2', 27.7, 0.4, 33, 0.3, 12, 16.3, 1.4, 2.1,
            17, 272
        ],
        [
            'Green Beans (can)', 'No. 2', 10, 1.0, 54, 2, 65, 53.9, 1.6, 4.3,
            32, 431
        ],
        [
            'Pork and Beans (can)', '16 oz.', 7.1, 7.5, 364, 4, 134, 3.5, 8.3,
            7.7, 56, 0
        ],
        ['Corn (can)', 'No. 2', 10.4, 5.2, 136, 0.2, 16, 12, 1.6, 2.7, 42, 218],
        [
            'Peas (can)', 'No. 2', 13.8, 2.3, 136, 0.6, 45, 34.9, 4.9, 2.5, 37,
            370
        ],
        [
            'Tomatoes (can)', 'No. 2', 8.6, 1.3, 63, 0.7, 38, 53.2, 3.4, 2.5,
            36, 1253
        ],
        [
            'Tomato Soup (can)', '10 1/2 oz.', 7.6, 1.6, 71, 0.6, 43, 57.9, 3.5,
            2.4, 67, 862
        ],
        [
            'Peaches, Dried', '1 lb.', 15.7, 8.5, 87, 1.7, 173, 86.8, 1.2, 4.3,
            55, 57
        ],
        [
            'Prunes, Dried', '1 lb.', 9, 12.8, 99, 2.5, 154, 85.7, 3.9, 4.3, 65,
            257
        ],
        [
            'Raisins, Dried', '15 oz.', 9.4, 13.5, 104, 2.5, 136, 4.5, 6.3, 1.4,
            24, 136
        ],
        [
            'Peas, Dried', '1 lb.', 7.9, 20.0, 1367, 4.2, 345, 2.9, 28.7, 18.4,
            162, 0
        ],
        [
            'Lima Beans, Dried', '1 lb.', 8.9, 17.4, 1055, 3.7, 459, 5.1, 26.9,
            38.2, 93, 0
        ],
        [
            'Navy Beans, Dried', '1 lb.', 5.9, 26.9, 1691, 11.4, 792, 0, 38.4,
            24.6, 217, 0
        ],
        ['Coffee', '1 lb.', 22.4, 0, 0, 0, 0, 0, 4, 5.1, 50, 0],
        ['Tea', '1/4 lb.', 17.4, 0, 0, 0, 0, 0, 0, 2.3, 42, 0],
        ['Cocoa', '8 oz.', 8.6, 8.7, 237, 3, 72, 0, 2, 11.9, 40, 0],
        ['Chocolate', '8 oz.', 16.2, 8.0, 77, 1.3, 39, 0, 0.9, 3.4, 14, 0],
        ['Sugar', '10 lb.', 51.7, 34.9, 0, 0, 0, 0, 0, 0, 0, 0],
        ['Corn Syrup', '24 oz.', 13.7, 14.7, 0, 0.5, 74, 0, 0, 0, 5, 0],
        ['Molasses', '18 oz.', 13.6, 9.0, 0, 10.3, 244, 0, 1.9, 7.5, 146, 0],
        [
            'Strawberry Preserves', '1 lb.', 20.5, 6.4, 11, 0.4, 7, 0.2, 0.2,
            0.4, 3, 0
        ],
    ]

    # Instantiate a Glop solver and naming it.
    solver = pywraplp.Solver('StiglerDietExample',
                             pywraplp.Solver.GLOP_LINEAR_PROGRAMMING)

    # Declare an array to hold our variables.
    foods = [solver.NumVar(0.0, solver.infinity(), item[0]) for item in data]

    print('Number of variables =', solver.NumVariables())

    # Create the constraints, one per nutrient.
    constraints = []
    for i, nutrient in enumerate(nutrients):
        constraints.append(solver.Constraint(nutrient[1], solver.infinity()))
        for j, item in enumerate(data):
            constraints[i].SetCoefficient(foods[j], item[i + 3])

    print('Number of constraints =', solver.NumConstraints())

    # Objective function: Minimize the sum of (price-normalized) foods.
    objective = solver.Objective()
    for food in foods:
        objective.SetCoefficient(food, 1)
    objective.SetMinimization()

    status = solver.Solve()

    # Check that the problem has an optimal solution.
    if status != solver.OPTIMAL:
        print('The problem does not have an optimal solution!')
        if status == solver.FEASIBLE:
            print('A potentially suboptimal solution was found.')
        else:
            print('The solver could not solve the problem.')
            exit(1)

    # Display the amounts (in dollars) to purchase of each food.
    nutrients_result = [0] * len(nutrients)
    print('\nAnnual Foods:')
    for i, food in enumerate(foods):
        if food.solution_value() > 0.0:
            print('{}: ${}'.format(data[i][0], 365. * food.solution_value()))
            for j, _ in enumerate(nutrients):
                nutrients_result[j] += data[i][j + 3] * food.solution_value()
    print('\nOptimal annual price: ${:.4f}'.format(365. * objective.Value()))

    print('\nNutrients per day:')
    for i, nutrient in enumerate(nutrients):
        print('{}: {:.2f} (min {})'.format(nutrient[0], nutrients_result[i],
                                           nutrient[1]))

    print('\nAdvanced usage:')
    print('Problem solved in ', solver.wall_time(), ' milliseconds')
    print('Problem solved in ', solver.iterations(), ' iterations')


if __name__ == '__main__':
    main()
```
