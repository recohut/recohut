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

<!-- #region id="zame09hpsG9V" -->
## An Array of Sequences

*As you may have noticed, several of the operations mentioned work equally for texts, lists and tables. Texts, lists and tables together are called trains. […] The FOR command also works generically on trains.* --Geurts, Meertens, and Pemberton, ABC Programmer’s Handbook
<!-- #endregion -->

<!-- #region id="EZsA-9Iw1D8Q" -->
### Cartesian product in a generator expression
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="-6xq0mwG1OTq" executionInfo={"status": "ok", "timestamp": 1630490926257, "user_tz": -330, "elapsed": 470, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="b7614009-0e96-48a2-82e3-f467ad2467de"
colors = ['black', 'white']
sizes = ['S', 'M', 'L']

for tshirt in ('%s %s' % (c, s) for c in colors for s in sizes):
    print(tshirt)
```

<!-- #region id="5tweotjD1Rfw" -->
### Line items from a flat-file invoice
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="Kc3TLEUP1lYW" executionInfo={"status": "ok", "timestamp": 1630491045529, "user_tz": -330, "elapsed": 519, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="db3f9a60-d0fa-4898-f00e-7a27552eddb7"
invoice = """
0.....6.................................40........52...55........
1909 Pimoroni PiBrella                      $17.50    3    $52.50
1489 6mm Tactile Switch x20                  $4.95    2    $9.90
1510 Panavise Jr. - PV-201                  $28.00    1    $28.00
1601 PiTFT Mini Kit 320x240                 $34.95    1    $34.95
"""

SKU = slice(0, 6)
DESCRIPTION = slice(6, 40)
UNIT_PRICE = slice(40, 52)
QUANTITY = slice(52, 55)
ITEM_TOTAL = slice(55, None)

line_items = invoice.split('\n')[2:]

for item in line_items:
    print(item[UNIT_PRICE], item[DESCRIPTION])
```

<!-- #region id="1Bfedg-h1ugd" -->
### List sorting
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="-xckY-pc2R7x" executionInfo={"status": "ok", "timestamp": 1630491197290, "user_tz": -330, "elapsed": 445, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="51415f38-eb70-4502-fed3-c1ddaccffca4"
fruits = ['grape', 'raspberry', 'apple', 'banana']
sorted(fruits)
```

```python colab={"base_uri": "https://localhost:8080/"} id="aWsp4qEb2Tqa" executionInfo={"status": "ok", "timestamp": 1630491203966, "user_tz": -330, "elapsed": 420, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="0f08f4cf-1fad-4363-f48c-683a490a3dce"
sorted(fruits, key=len, reverse=True)
```

<!-- #region id="MY5OQuTj2VU0" -->
### Working with a deque
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="a7_lcg9e3ZuQ" executionInfo={"status": "ok", "timestamp": 1630491514401, "user_tz": -330, "elapsed": 481, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="21471b02-00de-4da4-bc3e-c5826dafe7c1"
import collections

dq = collections.deque(range(10), maxlen=10)
dq
```

```python colab={"base_uri": "https://localhost:8080/"} id="idwgmpsj3g75" executionInfo={"status": "ok", "timestamp": 1630491563627, "user_tz": -330, "elapsed": 452, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="9079bfa2-4354-466d-af61-f9cc74a6c54c"
dq.rotate(3)
dq
```

```python colab={"base_uri": "https://localhost:8080/"} id="rzmwt_Ud3jf7" executionInfo={"status": "ok", "timestamp": 1630491566956, "user_tz": -330, "elapsed": 434, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="26746df9-898b-4427-fb5f-f7fb4d3ef698"
dq.rotate(-4)
dq
```

```python colab={"base_uri": "https://localhost:8080/"} id="cQwzVjxr3jVY" executionInfo={"status": "ok", "timestamp": 1630491570264, "user_tz": -330, "elapsed": 689, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="4a59d3a7-ec2d-4de3-eb05-f6a1b191f231"
dq.appendleft(-1)
dq
```

```python colab={"base_uri": "https://localhost:8080/"} id="eU3Ozg5h3jTl" executionInfo={"status": "ok", "timestamp": 1630491571666, "user_tz": -330, "elapsed": 16, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="3d76ea2b-c9a8-43ef-f49b-e57cc9e005ae"
dq.extend([11, 22, 33])
dq
```

```python colab={"base_uri": "https://localhost:8080/"} id="e5ky-ED63jRJ" executionInfo={"status": "ok", "timestamp": 1630491572977, "user_tz": -330, "elapsed": 8, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="37a20ba0-39eb-4cef-d6a8-9461d3f69759"
dq.extendleft([10, 20, 30, 40])
dq
```

<!-- #region id="nmntjcMM3jOp" -->
### Nested tuple unpacking
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="cFqg_jIS34Sh" executionInfo={"status": "ok", "timestamp": 1630491655863, "user_tz": -330, "elapsed": 737, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="e84c4792-e15e-496a-934a-2378215054f5"
metro_areas = [
    ('Tokyo', 'JP', 36.933, (35.689722, 139.691667)),
    ('Delhi NCR', 'IN', 21.935, (28.613889, 77.208889)),
    ('Mexico City', 'MX', 20.142, (19.433333, -99.133333)),
    ('New York-Newark', 'US', 20.104, (40.808611, -74.020386)),
    ('São Paulo', 'BR', 19.649, (-23.547778, -46.635833)),
]

print(f'{"":15} | {"latitude":>9} | {"longitude":>9}')
for name, _, _, (lat, lon) in metro_areas:
    if lon <= 0:
        print(f'{name:15} | {lat:9.4f} | {lon:9.4f}')
```

<!-- #region id="hXRyT2S54Bqw" -->
## Dictionaries and Sets
<!-- #endregion -->

<!-- #region id="3wMioudd5SPh" -->
*Python is basically dicts wrapped in loads of syntactic sugar.* --Lalo Martins, early digital nomad and Pythonista.
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="XgrJHp_25Pa4" executionInfo={"status": "ok", "timestamp": 1630492378347, "user_tz": -330, "elapsed": 655, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="dd04e00f-1f44-411f-9eee-79695bf33962"
a = dict(one=1, two=2, three=3)
b = {'three': 3, 'two': 2, 'one': 1}
c = dict([('two', 2), ('one', 1), ('three', 3)])
d = dict(zip(['one', 'two', 'three'], [1, 2, 3]))
e = dict({'three': 3, 'one': 1, 'two': 2})
a == b == c == d == e
```

```python colab={"base_uri": "https://localhost:8080/"} id="3fdaezBl6z8X" executionInfo={"status": "ok", "timestamp": 1630492435279, "user_tz": -330, "elapsed": 702, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="60132fde-fb63-46ce-ad1a-5e3fadc65302"
dial_codes = [
    (880, 'Bangladesh'),
    (55,  'Brazil'),
    (86,  'China'),
    (91,  'India'),
    (62,  'Indonesia'),
    (81,  'Japan'),
    (234, 'Nigeria'),
    (92,  'Pakistan'),
    (7,   'Russia'),
    (1,   'United States'),
]

country_dial = {country: code for code, country in dial_codes}
country_dial
```

```python colab={"base_uri": "https://localhost:8080/"} id="C69sZmDB7B2m" executionInfo={"status": "ok", "timestamp": 1630492437258, "user_tz": -330, "elapsed": 6, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="2563a635-de7d-4408-f565-663c10c9b422"
{code: country.upper() 
    for country, code in sorted(country_dial.items())
    if code < 70}
```

<!-- #region id="bO96AWDI7CcH" -->
## Text Versus Bytes
<!-- #endregion -->

<!-- #region id="CEhrfKoKJm8H" -->
*Humans use text. Computers speak bytes.* --Esther Nam and Travis Fischer, Character Encoding and Unicode in Python
<!-- #endregion -->

<!-- #region id="rc2lqzJgJqyR" -->
### Encoding and decoding
<!-- #endregion -->

```python id="kq-INBXIJu0I"
from unicodedata import name

zwg_sample = """
1F468 200D 1F9B0            |man: red hair                      |E11.0
1F9D1 200D 1F91D 200D 1F9D1 |people holding hands               |E12.0
1F3CA 1F3FF 200D 2640 FE0F  |woman swimming: dark skin tone     |E4.0
1F469 1F3FE 200D 2708 FE0F  |woman pilot: medium-dark skin tone |E4.0
1F468 200D 1F469 200D 1F467 |family: man, woman, girl           |E2.0
1F3F3 FE0F 200D 26A7 FE0F   |transgender flag                   |E13.0
1F469 200D 2764 FE0F 200D 1F48B 200D 1F469 |kiss: woman, woman  |E2.0
"""

markers = {'\u200D': 'ZWG', # ZERO WIDTH JOINER
           '\uFE0F': 'V16', # VARIATION SELECTOR-16
          }

for line in zwg_sample.strip().split('\n'):
    code, descr, version = (s.strip() for s in line.split('|'))
    chars = [chr(int(c, 16)) for c in code.split()]
    print(''.join(chars), version, descr, sep='\t', end='')
    while chars:
        char = chars.pop(0)
        if char in markers:
            print(' + ' + markers[char], end='')
        else:
            ucode = f'U+{ord(char):04X}'
            print(f'\n\t{char}\t{ucode}\t{name(char)}', end='')
    print()
```
