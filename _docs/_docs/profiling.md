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

<!-- #region id="-MdGG6JTRKgr" -->
# Python Profiling for High performance
<!-- #endregion -->

<!-- #region id="jN662UHcRogm" -->
### Julia Set
<!-- #endregion -->

<!-- #region id="YeO0jEvORMB5" -->
The Julia set is an interesting CPU-bound problem for us to begin with. It is a fractal sequence that generates a complex output image, named after Gaston Julia.

The code that follows is a little longer than a version you might write yourself. It has a CPU-bound component and a very explicit set of inputs. This configuration allows us to profile both the CPU usage and the RAM usage so we can understand which parts of our code are consuming two of our scarce computing resources. This implementation is deliberately suboptimal, so we can identify memory-consuming operations and slow statements.
<!-- #endregion -->

<!-- #region id="sWJkF-GjTMah" -->
We will analyze a block of code that produces both a false grayscale plot and a pure grayscale variant of the Julia set, at the complex point c=-0.62772-0.42193j. A Julia set is produced by calculating each pixel in isolation; this is an “embarrassingly parallel problem,” as no data is shared between points.
<!-- #endregion -->

```python executionInfo={"elapsed": 588, "status": "ok", "timestamp": 1631019456908, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}, "user_tz": -330} id="-hKTeJn-Vcgy"
import time
from PIL import Image
import array
import matplotlib.pyplot as plt
%matplotlib inline
```

```python executionInfo={"elapsed": 746, "status": "ok", "timestamp": 1631019471297, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}, "user_tz": -330} id="yvqdkI-tVeZj"
# area of complex space to investigate
x1, x2, y1, y2 = -1.8, 1.8, -1.8, 1.8
c_real, c_imag = -0.62772, -.42193
```

```python id="c4YHuSaGVize"
def show_greyscale(output_raw, width, height, max_iterations):
    """Convert list to array, show using PIL"""
    # convert our output to PIL-compatible input
    # scale to [0...255]
    max_iterations = float(max(output_raw))
    print(max_iterations)
    scale_factor = float(max_iterations)
    scaled = [int(o / scale_factor * 255) for o in output_raw]
    output = array.array('B', scaled)  # array of unsigned ints
    # display with PIL
    im = Image.new("L", (width, width))
    # EXPLAIN RAW L 0 -1
    im.frombytes(output.tobytes(), "raw", "L", 0, -1)
    # im.show()
    display(im)
```

```python id="osCm0w_EVkmC"
def show_false_greyscale(output_raw, width, height, max_iterations):
    """Convert list to array, show using PIL"""
    # convert our output to PIL-compatible input
    assert width * height == len(output_raw)  # sanity check our 1D array and desired 2D form
    # rescale output_raw to be in the inclusive range [0..255]
    max_value = float(max(output_raw))
    output_raw_limited = [int(float(o) / max_value * 255) for o in output_raw]
    # create a slightly fancy colour map that shows colour changes with
    # increased contrast (thanks to John Montgomery)
    output_rgb = ((o + (256 * o) + (256 ** 2) * o) * 16 for o in output_raw_limited)  # fancier
    output_rgb = array.array('I', output_rgb)  # array of unsigned ints (size is platform specific)
    # display with PIL/pillow
    im = Image.new("RGB", (width, height))
    # EXPLAIN RGBX L 0 -1
    im.frombytes(output_rgb.tobytes(), "raw", "RGBX", 0, -1)
    # im.show()
    display(im)
```

```python id="yt2RVLlNVmaI"
def calculate_z_serial_purepython(maxiter, zs, cs):
    """Calculate output list using Julia update rule"""
    output = [0] * len(zs)
    for i in range(len(zs)):
        n = 0
        z = zs[i]
        c = cs[i]
        while abs(z) < 2 and n < maxiter:
            z = z * z + c
            n += 1
        output[i] = n
    return output
```

```python id="4nEEK_59VpA_"
def calc_pure_python(draw_output, desired_width, max_iterations):
    """Create a list of complex co-ordinates (zs) and complex parameters (cs), build Julia set and display"""
    x_step = (x2 - x1) / desired_width
    y_step = (y1 - y2) / desired_width
    x = []
    y = []
    ycoord = y2
    while ycoord > y1:
        y.append(ycoord)
        ycoord += y_step
    xcoord = x1
    while xcoord < x2:
        x.append(xcoord)
        xcoord += x_step
    # set width and height to the generated pixel counts, rather than the
    # pre-rounding desired width and height
    width = len(x)
    height = len(y)
    # build a list of co-ordinates and the initial condition for each cell.
    # Note that our initial condition is a constant and could easily be removed,
    # we use it to simulate a real-world scenario with several inputs to our function
    zs = []
    cs = []
    for ycoord in y:
        for xcoord in x:
            zs.append(complex(xcoord, ycoord))
            cs.append(complex(c_real, c_imag))

    print("Length of x:", len(x))
    print("Total elements:", len(zs))
    start_time = time.time()
    output = calculate_z_serial_purepython(max_iterations, zs, cs)
    end_time = time.time()
    secs = end_time - start_time
    print(calculate_z_serial_purepython.__name__ + " took", secs, "seconds")

    assert sum(output) == 33219980  # this sum is expected for 1000^2 grid with 300 iterations

    if draw_output:
        #show_false_greyscale(output, width, height, max_iterations)
        show_greyscale(output, width, height, max_iterations)
```

```python colab={"base_uri": "https://localhost:8080/", "height": 1000} executionInfo={"elapsed": 11066, "status": "ok", "timestamp": 1631019110896, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}, "user_tz": -330} id="EFEuiESeRmrQ" outputId="ee262446-4301-4235-cb49-5b65573445df"
if __name__ == "__main__":
    # Calculate the Julia set using a pure Python solution with
    # reasonable defaults for a laptop
    # set draw_output to True to use PIL to draw an image
    calc_pure_python(draw_output=True, desired_width=1000, max_iterations=300)
```

<!-- #region id="oUI69O-_SAdJ" -->
If we chose a different c, we’d get a different image. The location we have chosen has regions that are quick to calculate and others that are slow to calculate; this is useful for our analysis.

The problem is interesting because we calculate each pixel by applying a loop that could be applied an indeterminate number of times. On each iteration we test to see if this coordinate’s value escapes toward infinity, or if it seems to be held by an attractor. Coordinates that cause few iterations are colored darkly, and those that cause a high number of iterations are colored white. White regions are more complex to calculate and so take longer to generate.


<!-- #endregion -->

<!-- #region id="O-UJ59E2Vrn4" -->
### Defining a decorator to automate timing measurements
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 10598, "status": "ok", "timestamp": 1631019762063, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}, "user_tz": -330} id="rNmzzs6aWVqN" outputId="851a1061-4079-4b61-daa5-39c941d8dd74"
import time
from functools import wraps

# area of complex space to investigate
x1, x2, y1, y2 = -1.8, 1.8, -1.8, 1.8
c_real, c_imag = -0.62772, -.42193


def timefn(fn):
    @wraps(fn)
    def measure_time(*args, **kwargs):
        t1 = time.time()
        result = fn(*args, **kwargs)
        t2 = time.time()
        print(f"@timefn: {fn.__name__} took {t2 - t1} seconds")
        return result
    return measure_time


@timefn
def calculate_z_serial_purepython(maxiter, zs, cs):
    """Calculate output list using Julia update rule"""
    output = [0] * len(zs)
    for i in range(len(zs)):
        n = 0
        z = zs[i]
        c = cs[i]
        while abs(z) < 2 and n < maxiter:
            z = z * z + c
            n += 1
        output[i] = n
    return output


def calc_pure_python(draw_output, desired_width, max_iterations):
    """Create a list of complex co-ordinates (zs) and complex parameters (cs), build Julia set and display"""
    x_step = (x2 - x1) / desired_width
    y_step = (y1 - y2) / desired_width
    x = []
    y = []
    ycoord = y2
    while ycoord > y1:
        y.append(ycoord)
        ycoord += y_step
    xcoord = x1
    while xcoord < x2:
        x.append(xcoord)
        xcoord += x_step
    # build a list of co-ordinates and the initial condition for each cell.
    # Note that our initial condition is a constant and could easily be removed,
    # we use it to simulate a real-world scenario with several inputs to our function
    zs = []
    cs = []
    for ycoord in y:
        for xcoord in x:
            zs.append(complex(xcoord, ycoord))
            cs.append(complex(c_real, c_imag))

    print("Length of x:", len(x))
    print("Total elements:", len(zs))
    start_time = time.time()
    output = calculate_z_serial_purepython(max_iterations, zs, cs)
    end_time = time.time()
    secs = end_time - start_time
    print(calculate_z_serial_purepython.__name__ + " took", secs, "seconds")

    assert sum(output) == 33219980  # this sum is expected for 1000^2 grid with 300 iterations


# Calculate the Julia set using a pure Python solution with
# reasonable defaults for a laptop
# set draw_output to True to use PIL to draw an image
calc_pure_python(draw_output=False, desired_width=1000, max_iterations=300)
```

<!-- #region id="3w6VNBa6WlSX" -->
### Using the cProfile Module
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 680, "status": "ok", "timestamp": 1631020049615, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}, "user_tz": -330} id="I-8I8yRSXMaW" outputId="e61b0944-13b3-4d7a-a6cb-847d28f9966f"
%%writefile julia.py
import time

# area of complex space to investigate
x1, x2, y1, y2 = -1.8, 1.8, -1.8, 1.8
c_real, c_imag = -0.62772, -.42193


def calculate_z_serial_purepython(maxiter, zs, cs):
    """Calculate output list using Julia update rule"""
    output = [0] * len(zs)
    for i in range(len(zs)):
        n = 0
        z = zs[i]
        c = cs[i]
        while abs(z) < 2 and n < maxiter:
            z = z * z + c
            n += 1
        output[i] = n
    return output


def calc_pure_python(draw_output, desired_width, max_iterations):
    """Create a list of complex co-ordinates (zs) and complex parameters (cs), build Julia set and display"""
    x_step = (x2 - x1) / desired_width
    y_step = (y1 - y2) / desired_width
    x = []
    y = []
    ycoord = y2
    while ycoord > y1:
        y.append(ycoord)
        ycoord += y_step
    xcoord = x1
    while xcoord < x2:
        x.append(xcoord)
        xcoord += x_step
    # build a list of co-ordinates and the initial condition for each cell.
    # Note that our initial condition is a constant and could easily be removed,
    # we use it to simulate a real-world scenario with several inputs to our function
    zs = []
    cs = []
    for ycoord in y:
        for xcoord in x:
            zs.append(complex(xcoord, ycoord))
            cs.append(complex(c_real, c_imag))

    print("Length of x:", len(x))
    print("Total elements:", len(zs))
    start_time = time.time()
    output = calculate_z_serial_purepython(max_iterations, zs, cs)
    end_time = time.time()
    secs = end_time - start_time
    print(calculate_z_serial_purepython.__name__ + " took", secs, "seconds")

    assert sum(output) == 33219980  # this sum is expected for 1000^2 grid with 300 iterations


# Calculate the Julia set using a pure Python solution with
# reasonable defaults for a laptop
# set draw_output to True to use PIL to draw an image
calc_pure_python(draw_output=False, desired_width=1000, max_iterations=300)
```

```python colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 15137, "status": "ok", "timestamp": 1631020069833, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}, "user_tz": -330} id="LtrDX36WXg-w" outputId="260b8850-f705-46b4-b149-8cff570e3c81"
!python -m cProfile -s cumulative julia.py
```

```python colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 13396, "status": "ok", "timestamp": 1631020123236, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}, "user_tz": -330} id="vZVGCPU3XuPi" outputId="0a93fb83-310d-4cd3-dc60-456b440ff8e7"
!python -m cProfile -o profile.stats julia.py
```

```python colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 6, "status": "ok", "timestamp": 1631020190380, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}, "user_tz": -330} id="Yf2wbfzrYQGK" outputId="7cdae104-d6a2-48c8-b14f-2e47f54e42d0"
import pstats
p = pstats.Stats("profile.stats")
p.sort_stats("cumulative")
p.print_stats()
```

```python colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 799, "status": "ok", "timestamp": 1631020211197, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}, "user_tz": -330} id="X4RMSJUcYQaq" outputId="b92ed852-a0ce-4c80-bf78-f64ac90aa17a"
p.print_callers()
```

```python colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 769, "status": "ok", "timestamp": 1631020222420, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}, "user_tz": -330} id="FOlb763YYVWg" outputId="b10c3202-dc6f-41b0-e4c0-914915c5b8f3"
p.print_callees()
```

<!-- #region id="NCyrZVhbYYF9" -->
### Visualizing cProfile Output with SnakeViz
<!-- #endregion -->

```python id="ANrw9FWuYa8B"
!pip install snakeviz
```

```python executionInfo={"elapsed": 833, "status": "ok", "timestamp": 1631020349384, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}, "user_tz": -330} id="Q8KxyiZLY2uR"
%load_ext snakeviz
```

```python colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 963, "status": "ok", "timestamp": 1631021171564, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}, "user_tz": -330} id="qNezujolYfAe" outputId="61cea8b3-2308-4ea5-c56b-22e90bebe6bd"
!nohup snakeviz profile.stats -p 5061 -s &
```

<!-- #region id="V9fRh0oXdE6x" -->
### Using line_profiler for Line-by-Line Measurements
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 4137, "status": "ok", "timestamp": 1631022523391, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}, "user_tz": -330} id="2xoY4VOUhFnQ" outputId="fabaedbc-76ad-49d8-8fa7-26cce267d3a8"
!pip install line_profiler
```

```python colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 907, "status": "ok", "timestamp": 1631022661366, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}, "user_tz": -330} id="eEmdypUChnUU" outputId="5d168d7f-af1b-4f2c-8beb-7123e2a106f6"
%%writefile julia_lprof.py
import time

# area of complex space to investigate
x1, x2, y1, y2 = -1.8, 1.8, -1.8, 1.8
c_real, c_imag = -0.62772, -.42193


@profile
def calculate_z_serial_purepython(maxiter, zs, cs):
    """Calculate output list using Julia update rule"""
    output = [0] * len(zs)
    for i in range(len(zs)):
        n = 0
        z = zs[i]
        c = cs[i]
        while True:
            not_yet_escaped = abs(z) < 2
            iterations_left = n < maxiter
            if not_yet_escaped and iterations_left:
                z = z * z + c
                n += 1
            else:
                break
        output[i] = n
    return output


@profile
def calc_pure_python(draw_output, desired_width, max_iterations):
    """Create a list of complex co-ordinates (zs) and complex parameters (cs), build Julia set and display"""
    x_step = (x2 - x1) / desired_width
    y_step = (y1 - y2) / desired_width
    x = []
    y = []
    ycoord = y2
    while ycoord > y1:
        y.append(ycoord)
        ycoord += y_step
    xcoord = x1
    while xcoord < x2:
        x.append(xcoord)
        xcoord += x_step
    # set width and height to the generated pixel counts, rather than the
    # pre-rounding desired width and height
    # build a list of co-ordinates and the initial condition for each cell.
    # Note that our initial condition is a constant and could easily be removed,
    # we use it to simulate a real-world scenario with several inputs to our function
    zs = []
    cs = []
    for ycoord in y:
        for xcoord in x:
            zs.append(complex(xcoord, ycoord))
            cs.append(complex(c_real, c_imag))

    print("Length of x:", len(x))
    print("Total elements:", len(zs))
    start_time = time.time()
    output = calculate_z_serial_purepython(max_iterations, zs, cs)
    end_time = time.time()
    secs = end_time - start_time
    print(calculate_z_serial_purepython.__name__ + " took", secs, "seconds")

    assert sum(output) == 33219980  # this sum is expected for 1000^2 grid with 300 iterations



# Calculate the Julia set using a pure Python solution with
# reasonable defaults for a laptop
# set draw_output to True to use PIL to draw an image
calc_pure_python(draw_output=False, desired_width=1000, max_iterations=300)
```

```python colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 177728, "status": "ok", "timestamp": 1631022844270, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}, "user_tz": -330} id="EunGYiAhhI8a" outputId="8ae94db3-5ca4-4318-fdff-0b718a7c40fc"
!kernprof -l -v julia_lprof.py
```

<!-- #region id="hWLjyrl8h83J" -->
### Using memory_profiler to Diagnose Memory Usage
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 6335, "status": "ok", "timestamp": 1631022922414, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}, "user_tz": -330} id="4QvPHobmipAI" outputId="282119c8-8be8-4e09-a1cb-cac454b45341"
!pip install memory_profiler
```

```python colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 4, "status": "ok", "timestamp": 1631022846550, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}, "user_tz": -330} id="eTWtBsGBheYk" outputId="98fd5712-a978-469c-9f93-475e6a0d1cbe"
%%writefile julia_memprof.py
import time

# area of complex space to investigate
x1, x2, y1, y2 = -1.8, 1.8, -1.8, 1.8
c_real, c_imag = -0.62772, -.42193


@profile
def calculate_z_serial_purepython(maxiter, zs, cs):
    """Calculate output list using Julia update rule"""
    output = [0] * len(zs)
    for i in range(len(zs)):
        n = 0
        z = zs[i]
        c = cs[i]
        while n < maxiter and abs(z) < 2:
            z = z * z + c
            n += 1
        output[i] = n
    return output


@profile
def calc_pure_python(draw_output, desired_width, max_iterations):
    """Create a list of complex co-ordinates (zs) and complex parameters (cs), build Julia set and display"""
    x_step = (x2 - x1) / desired_width
    y_step = (y1 - y2) / desired_width
    x = []
    y = []
    ycoord = y2
    while ycoord > y1:
        y.append(ycoord)
        ycoord += y_step
    xcoord = x1
    while xcoord < x2:
        x.append(xcoord)
        xcoord += x_step
    # set width and height to the generated pixel counts, rather than the
    # pre-rounding desired width and height
    # build a list of co-ordinates and the initial condition for each cell.
    # Note that our initial condition is a constant and could easily be removed,
    # we use it to simulate a real-world scenario with several inputs to our function
    zs = []
    cs = []
    for ycoord in y:
        for xcoord in x:
            zs.append(complex(xcoord, ycoord))
            cs.append(complex(c_real, c_imag))

    print("Length of x:", len(x))
    print("Total elements:", len(zs))
    start_time = time.time()
    output = calculate_z_serial_purepython(max_iterations, zs, cs)
    end_time = time.time()
    secs = end_time - start_time
    print(calculate_z_serial_purepython.__name__ + " took", secs, "seconds")

    assert sum(output) == 33219980  # this sum is expected for 1000^2 grid with 300 iterations


# Calculate the Julia set using a pure Python solution with
# reasonable defaults for a laptop
# set draw_output to True to use PIL to draw an image
#calc_pure_python(draw_output=False, desired_width=1000, max_iterations=300)
calc_pure_python(draw_output=False, desired_width=1000, max_iterations=300)
```

```python colab={"background_save": true, "base_uri": "https://localhost:8080/"} id="ulTz5n_niC3-"
!python -m memory_profiler julia_memprof.py
```
