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

<!-- #region id="V1JS0B802NXL" -->
Before we dive deep into the world of PyTorch development, it’s important to familiarize yourself with the fundamental data structure in PyTorch: the torch.tensor. By understanding the tensor, you will understand how PyTorch handles and stores data, and since deep learning is fundamentally the collection and manipulation of floating-point numbers, understanding tensors will help you understand how PyTorch implements more advanced functions for deep learning. In addition, you may find yourself using tensor operations frequently when preprocessing input data or manipulating output data during model development
<!-- #endregion -->

<!-- #region id="dmq42z9n2qPS" -->
In PyTorch, a tensor is a data structure used to store and manipulate data. Like a NumPy array, a tensor is a multidimensional array containing elements of a single data type. Tensors can be used to represent scalars, vectors, matrices, and n-dimensional arrays and are derived from the torch.Tensor class. However, tensors are more than just arrays of numbers. Creating or instantiating a tensor object from the torch.Tensor class gives us access to a set of built-in class attributes and operations or class methods that provide a robust set of built-in capabilities. This guide describes these attributes and operations in detail.
<!-- #endregion -->

<!-- #region id="ClxVmz532rjS" -->
Tensors also include added benefits that make them more suitable than NumPy arrays for deep learning calculations. First, tensor operations can be performed significantly faster using GPU acceleration. Second, tensors can be stored and manipulated at scale using distributed processing on multiple CPUs and GPUs and across multiple servers. And third, tensors keep track of their graph computations, which is very important in implementing a deep learning library.
<!-- #endregion -->

<!-- #region id="mW6c7P-N22oi" -->
**Simple example**
<!-- #endregion -->

<!-- #region id="JddfsjK83LfC" -->
First, we import the PyTorch library, then we create two tensors, x and y, from two-dimensional lists. Next, we add the two tensors and store the result in z. We can just use the + operator here because the torch.Tensor class supports operator overloading. Finally, we print the new tensor, z, which we can see is the matrix sum of x and y, and we print the size of z. Notice that z is a tensor object itself and the size() method is used to return its matrix dimensions, namely 2 × 3:
<!-- #endregion -->

```python id="FsxVCauz3MKH" executionInfo={"status": "ok", "timestamp": 1631129005705, "user_tz": -330, "elapsed": 4290, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
import torch

x = torch.tensor([[1,2,3],[4,5,6]])
y = torch.tensor([[7,8,9],[10,11,12]])
z = x + y
```

```python colab={"base_uri": "https://localhost:8080/"} id="mEIMUuTr3Udo" executionInfo={"status": "ok", "timestamp": 1631129024511, "user_tz": -330, "elapsed": 467, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="e8894882-b7a2-4255-a9dd-400e80ec6d03"
print(z)
```

```python colab={"base_uri": "https://localhost:8080/"} id="GXEbbu4M3TNI" executionInfo={"status": "ok", "timestamp": 1631129038084, "user_tz": -330, "elapsed": 425, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="c4fec653-9e02-468f-83de-f1317127d6f2"
print(z.size())
```

```python colab={"base_uri": "https://localhost:8080/", "height": 885} id="WyeCg7C73eiY" executionInfo={"status": "ok", "timestamp": 1631129159852, "user_tz": -330, "elapsed": 541, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="fa75bf8e-78fc-4201-f7ce-50913cf15b61"
', '.join(dir(z))
```

<!-- #region id="_dYo0ElY3gmG" -->
**Running it on gpu (if available)**
<!-- #endregion -->

```python id="t7tC4u3r4Fl_" executionInfo={"status": "ok", "timestamp": 1631129342724, "user_tz": -330, "elapsed": 518, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
device = "cuda" if torch.cuda.is_available() else "cpu"

x = torch.tensor([[1,2,3],[4,5,6]],
                 device=device)
y = torch.tensor([[7,8,9],[10,11,12]],
                 device=device)
z = x + y
```

```python colab={"base_uri": "https://localhost:8080/"} id="__m8emIe4o2F" executionInfo={"status": "ok", "timestamp": 1631129347120, "user_tz": -330, "elapsed": 10, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="4efd6e86-e5ad-48ed-8c85-b6562e61a108"
print(z)
```

```python colab={"base_uri": "https://localhost:8080/"} id="zjYvTp7c4p8V" executionInfo={"status": "ok", "timestamp": 1631129356196, "user_tz": -330, "elapsed": 456, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="331b0120-3803-40ff-944d-557bc90b65fd"
print(z.size())
```

```python colab={"base_uri": "https://localhost:8080/"} id="G8z472dV4sLM" executionInfo={"status": "ok", "timestamp": 1631129360441, "user_tz": -330, "elapsed": 505, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="3a4bee36-05c4-4acc-9590-1bfa2fa2d8d9"
print(z.device)
```

<!-- #region id="gkJvL2pC4tMn" -->
The previous section showed a simple way to create tensors; however, there are many other ways to do it. You can create tensors from preexisting numeric data or create random samplings. Tensors can be created from preexisting data stored in array-like structures such as lists, tuples, scalars, or serialized data files, as well as in NumPy arrays.

The following code illustrates some common ways to create tensors. First, it shows how to create a tensor from a list using torch.tensor(). This method can also be used to create tensors from other data structures like tuples, sets, or NumPy arrays:
<!-- #endregion -->

```python id="alVKYnj45bT9" executionInfo={"status": "ok", "timestamp": 1631129714555, "user_tz": -330, "elapsed": 624, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
import numpy 

# Created from pre-existing arrays
w = torch.tensor([1,2,3]) # <1>
w = torch.tensor((1,2,3)) # <2>
w = torch.tensor(numpy.array([1,2,3])) # <3>

# Initialized by size
w = torch.empty(100,200) # <4>
w = torch.zeros(100,200) # <5>
w = torch.ones(100,200)  # <6>

# Initialized by size with random values
w = torch.rand(100,200)     # <7>
w = torch.randn(100,200)    # <8>
w = torch.randint(5,10,(100,200))  # <9> 

# Initialized with specified data type or device
w = torch.empty((100,200), dtype=torch.float64, 
                device="cpu")

# Initialized to have same size, data type, 
#   and device as another tensor
x = torch.empty_like(w)
```

<!-- #region id="2LVb-lUd5pLa" -->
1. from a list
2. from a tuple
3. from a numpy array
4. uninitialized, elements values are not predictable
5. all elements initialized with 0.0
6. all elements initialized with 1.0
7. creates a 100 x 200 tensor with elements from a uniform distribution on the interval [0, 1)
8. elements are random numbers from a normal distribution with mean 0 and variance 1
9. elements are random integers between 5 and 10
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="Xdx5vlig_kIn" executionInfo={"status": "ok", "timestamp": 1631131339172, "user_tz": -330, "elapsed": 619, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="ebc0a95a-8a12-4b1d-acaa-11d079b00ba2"
x = torch.tensor([[1,2,3],[4,5,6]])

print(torch.empty_like(x))
print(torch.empty_like(x))
print(torch.zeros_like(x))
print(torch.ones_like(x))

print(torch.full_like(x, fill_value=5))
```

<!-- #region id="SN3rMtBV6HE4" -->
Following table lists PyTorch functions used to create tensors. You should use each one with the torch namespace, e.g., torch.empty().
<!-- #endregion -->

<!-- #region id="HuBlN-LC6njl" -->
| Function | Description |
| -------- | ----------- |
| torch.tensor(data, dtype=None, device=None, <br /> requires_grad=False, pin_memory=False) | Creates a tensor from an existing data structure |
| torch.empty(*size, out=None, dtype=None, <br />layout=torch.strided, device=None, requires_grad=False) | Creates a tensor from uninitialized elements based on the random state of values in memory |
| torch.zeros(*size, out=None, dtype=None, <br />layout=torch.strided, device=None, requires_grad=False) | Creates a tensor with all elements initialized to 0.0 |
| torch.ones(*size, out=None, dtype=None, <br />layout=torch.strided, device=None, requires_grad=False) | Creates a tensor with all elements initialized to 1.0 |
| torch.arange(start=0, end, step=1, out=None, <br />dtype=None, layout=torch.strided, device=None, requires_grad=False) | Creates a 1D tensor of values over a range with a common step value |
| torch.linspace(start, end, steps=100, <br />out=None, dtype=None, layout=torch.strided, <br />device=None, requires_grad=False) | Creates a 1D tensor of linearly spaced points between the start and end |
| torch.logspace(start, end, steps=100, <br />base=10.0, out=None, dtype=None, <br />layout=torch.strided, device=None, requires_grad=False) | Creates a 1D tensor of logarithmically spaced points between the start and end |
| torch.eye(n, m=None, out=None, dtype=None, <br />layout=torch.strided, device=None, requires_grad=False) | Creates a 2D tensor with ones on the diagonal and zeros everywhere else |
| torch.full(size, fill_value, out=None, <br />dtype=None, layout=torch.strided, device=None, requires_grad=False) | Creates a tensor filled with fill_value |
| torch.load(f) | Loads a tensor from a serialized pickle file |
| torch.save(f) | Saves a tensor to a serialized pickle file |
<!-- #endregion -->

<!-- #region id="d7p4SR2x7TYN" -->
During deep learning development, it’s important to be aware of the data type used by your data and its calculations. So when you create tensors, you should control what data types are being used. As mentioned previously, all tensor elements have the same data type. You can specify the data type when creating the tensor by using the dtype parameter, or you can cast a tensor to a new dtype using the appropriate casting method or the to() method, as shown in the following code:
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="mLRaLgB_93Og" executionInfo={"status": "ok", "timestamp": 1631130783444, "user_tz": -330, "elapsed": 525, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="24ef8ed3-944d-469c-ff3a-aa9b1ec6de1c"
# Specify data type at creation using dtype
w = torch.tensor([1,2,3], dtype=torch.float32)

# Use casting method to cast to a new data type
w.int()       # w remains a float32 after cast
w = w.int()   # w changes to int32 after cast

# Use to() method to cast to a new type
w = w.to(torch.float64) # <1>
w = w.to(dtype=torch.float64) # <2>

# Python automatically converts data types during operations
x = torch.tensor([1,2,3], dtype=torch.int32)
y = torch.tensor([1,2,3], dtype=torch.float32)
z = x + y # <3>
print(z.dtype)
```

<!-- #region id="trT-uz9G_HV8" -->
Table below lists all the available data types in PyTorch. Each data type results in a different tensor class depending on the tensor’s device. The corresponding tensor classes are shown in the two rightmost columns for CPUs and GPUs, respectively.
<!-- #endregion -->

<!-- #region id="D5zCAiIQ-Inm" -->
| Data type | dtype | CPU tensor | GPU tensor |
| --------- | ----- | ---------- | ---------- |
| 32-bit floating point (default) | torch.float32 or torch.float | torch.​​Float⁠Ten⁠sor | torch.cuda.​Float⁠Tensor |
| 64-bit floating point | torch.float64 or torch.dou⁠ble | torch.​​Dou⁠ble⁠Tensor | torch.cuda.​​Dou⁠bleTensor |
| 16-bit floating point | torch.float16 or torch.half | torch.​Half⁠Tensor | torch.cuda.​Half⁠Tensor |
| 8-bit integer (unsigned) | torch.uint8 | torch.​Byte⁠Tensor | torch.cuda.​Byte⁠Tensor |
| 8-bit integer (signed) | torch.int8 | torch.​Char⁠Tensor | torch.cuda.​Char⁠Tensor |
| 16-bit integer (signed) | torch.int16 or torch.short | torch.​Short⁠Tensor | torch.cuda.​Short⁠Tensor |
| 32-bit integer (signed) | torch.int32 or torch.int | torch.​IntTen⁠sor | torch.cuda.​IntTen⁠sor |
| 64-bit integer (signed) | torch.int64 or torch.long | torch.​Long⁠Tensor | torch.cuda.​Long⁠Tensor |
| Boolean | torch.bool | torch.​Bool⁠Tensor | torch.cuda.​Bool⁠Tensor |
<!-- #endregion -->

<!-- #region id="8AjkaSqO-7Zu" -->
**Indexing, Slicing, Combining, and Splitting Tensors**
<!-- #endregion -->

<!-- #region id="m_VAewgyAjwx" -->
Once you have created tensors, you may want to access portions of the data and combine or split tensors to form new tensors. The following code demonstrates how to perform these types of operations. You can slice and index tensors in the same way you would slice and index NumPy arrays.
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="uQGhWD-6AlPD" executionInfo={"status": "ok", "timestamp": 1631131457846, "user_tz": -330, "elapsed": 4, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="d4e71867-d856-4022-e32c-812db216e17a"
x = torch.tensor([[1,2],[3,4],[5,6],[7,8]])
x
```

```python colab={"base_uri": "https://localhost:8080/"} id="7O6X3OYuAtOa" executionInfo={"status": "ok", "timestamp": 1631131458570, "user_tz": -330, "elapsed": 7, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="89f2c247-2efe-48c5-fdff-1009791d1ca7"
# Indexing, returns a tensor
print(x[1,1])
```

```python colab={"base_uri": "https://localhost:8080/"} id="-LbU5OomAthN" executionInfo={"status": "ok", "timestamp": 1631131478134, "user_tz": -330, "elapsed": 807, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="91f2a05b-c509-4295-e596-76a2e511907c"
# Indexing, returns a value as a Python number
print(x[1,1].item())
```

```python colab={"base_uri": "https://localhost:8080/"} id="ptvE3KWXAyKE" executionInfo={"status": "ok", "timestamp": 1631131493805, "user_tz": -330, "elapsed": 451, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="45ac3bac-dce8-4cde-b604-459b1e086b29"
# Slicing
print(x[:2,1])
```

```python colab={"base_uri": "https://localhost:8080/"} id="42-DNNI4A2Df" executionInfo={"status": "ok", "timestamp": 1631131513968, "user_tz": -330, "elapsed": 612, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="02a75a93-d252-4ea5-90a4-bf0bcf4b38aa"
# Boolean indexing
# Only keep elements less than 5
print(x[x<5])
```

```python colab={"base_uri": "https://localhost:8080/"} id="nuG5jm8BA682" executionInfo={"status": "ok", "timestamp": 1631131530965, "user_tz": -330, "elapsed": 510, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="02be09e8-4f1a-45d8-b662-28b7cfb3130b"
# Transpose array; x.t() or x.T can be used
print(x.t())
```

```python colab={"base_uri": "https://localhost:8080/"} id="zHp9ZqpHA_HI" executionInfo={"status": "ok", "timestamp": 1631131548192, "user_tz": -330, "elapsed": 10, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="4e0fe869-b7f9-42c2-c83e-51a49c08f2b0"
# Change shape; usually view() is preferred over
# reshape()
print(x.view((2,4)))
```

<!-- #region id="fCZINmpeBDO1" -->
You can also combine or split tensors by using functions like torch.stack() and torch.unbind(), respectively, as shown in the following code:
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="Wu2qkNljBQlP" executionInfo={"status": "ok", "timestamp": 1631131625385, "user_tz": -330, "elapsed": 446, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="0d3d90ef-0c3c-4d6a-f42a-f295b76be533"
# Combining tensors
y = torch.stack((x, x))
print(y)
```

```python colab={"base_uri": "https://localhost:8080/"} id="W9IKh1U-Bsmn" executionInfo={"status": "ok", "timestamp": 1631131718764, "user_tz": -330, "elapsed": 10, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="a07d0050-9ad3-4442-9854-7570cd066ece"
x
```

```python colab={"base_uri": "https://localhost:8080/"} id="Zxyw4MgwBWMR" executionInfo={"status": "ok", "timestamp": 1631131697996, "user_tz": -330, "elapsed": 890, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="1736171c-1bfa-416d-8736-a15b5dc155c8"
# Splitting tensors
a,b = x.unbind(dim=1)
print(a,b)
```

```python colab={"base_uri": "https://localhost:8080/"} id="85HKy4e_Bn0V" executionInfo={"status": "ok", "timestamp": 1631131777258, "user_tz": -330, "elapsed": 621, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="9226f00b-ed00-459b-db86-e35a8bfa4ebb"
# Splitting tensors
a,b,c,d = x.unbind(dim=0)
print(a,b,c,d)
```

<!-- #region id="w5GOH2K8BvZ5" -->
PyTorch provides a robust set of built-in functions that can be used to access, split, and combine tensors in different ways. Table below lists some commonly used functions to manipulate tensor elements.
<!-- #endregion -->

<!-- #region id="rJUrzJA8CWDm" -->
| Function | Description |
| -------- | ----------- |
| torch.**cat**() | Concatenates the given sequence of tensors in the given dimension. |
| torch.**chunk**() | Splits a tensor into a specific number of chunks. Each chunk is a view of the input tensor. |
| torch.**gather**() | Gathers values along an axis specified by the dimension. |
| torch.**index\_select**() | Returns a new tensor that indexes the input tensor along a dimension using the entries in the index, which is a LongTensor. |
| torch.**masked\_select**() | Returns a new 1D tensor that indexes the input tensor according to the Boolean mask, which is a BoolTensor. |
| torch.**narrow**() | Returns a tensor that is a narrow version of the input tensor. |
| torch.**nonzero**() | Returns the indices of nonzero elements. |
| torch.**reshape**() | Returns a tensor with the same data and number of elements as the input tensor, but a different shape. </br>Use view() instead to ensure the tensor is not copied. |
| torch.**split**() | Splits the tensor into chunks. Each chunk is a view or subdivision of the original tensor. |
| torch.**squeeze**() | Returns a tensor with all the dimensions of the input tensor of size 1 removed. |
| torch.**stack**() | Concatenates a sequence of tensors along a new dimension. |
| torch.**t**() | Expects the input to be a 2D tensor and transposes dimensions 0 and 1. |
| torch.**take**() | Returns a tensor at specified indices when slicing is not continuous. |
| torch.**transpose**() | Transposes only the specified dimensions. |
| torch.**unbind**() | Removes a tensor dimension by returning a tuple of the removed dimension. |
| torch.**unsqueeze**() | Returns a new tensor with a dimension of size 1 inserted at the specified position. |
| torch.**where**() | Returns a tensor of selected elements from either one of two tensors, depending on the specified condition. |
<!-- #endregion -->

<!-- #region id="lJLaGnC7C3xR" -->
Deep learning development is strongly based on mathematical computations, so PyTorch supports a very robust set of built-in math functions. Whether you are creating new data transforms, customizing loss functions, or building your own optimization algorithms, you can speed up your research and development with the math functions provided by PyTorch.
<!-- #endregion -->

<!-- #region id="OsVAAMcODZ2L" -->
PyTorch supports many different types of math functions, including pointwise operations, reduction functions, comparison calculations, and linear algebra operations, as well as spectral and other math computations. The first category of useful math operations we’ll look at are pointwise operations. Pointwise operations perform an operation on each point in the tensor individually and return a new tensor.

They are useful for rounding and truncation as well as trigonometrical and logical operations. By default, the functions will create a new tensor or use one passed in by the out parameter. If you want to perform an in-place operation, remember to append an underscore to the function name.

Table below lists some commonly used pointwise operations.
<!-- #endregion -->

<!-- #region id="hNIlm_R5DlCy" -->
| Operation type | Sample functions |
| -------------- | ---------------- |
| Basic math | add(), div(), mul(), neg(), reciprocal(), true\_divide() |
| Truncation | ceil(), clamp(), floor(), floor\_divide(), fmod(), frac(), lerp(), remainder(), round(), sigmoid(), trunc() |
| Complex numbers | abs(), angle(), conj(), imag(), real() |
| Trigonometry | acos(), asin(), atan(), cos(), cosh(), deg2rad(), rad2deg(), sin(), sinh(), tan(), tanh() |
| Exponents and logarithms | exp(), expm1(), log(), log10(), log1p(), log2(), logaddexp(), pow(), rsqrt(), sqrt(), square() |
| Logical | logical\_and(), logical\_not(), logical\_or(), logical\_xor() |
| Cumulative math | addcdiv(), addcmul() |
| Bitwise operators | bitwise\_not(), bitwise\_and(), bitwise\_or(), bitwise\_xor() |
| Error functions | erf(), erfc(), erfinv() |
| Gamma functions | digamma(), lgamma(), mvlgamma(), polygamma() |
<!-- #endregion -->

<!-- #region id="6SfQFUgAD4sS" -->
The second category of math functions we’ll look at are reduction operations. Reduction operations reduce a bunch of numbers down to a single number or a smaller set of numbers. That is, they reduce the dimensionality or rank of the tensor. Reduction operations include functions for finding maximum or minimum values as well as many statistical calculations, like finding the mean or standard deviation.

These operations are frequently used in deep learning. For example, deep learning classification often uses the argmax() function to reduce softmax outputs to a dominant class.
<!-- #endregion -->

<!-- #region id="5F3pQs2vFAqX" -->
| Function | Description |
| -------- | ----------- |
| torch.**argmax**(_input, dim, keepdim=False, out=None_) | Returns the index(es) of the maximum value across all elements, or just a dimension if it’s specified |
| torch.**argmin**(_input, dim, keepdim=False, out=None_) | Returns the index(es) of the minimum value across all elements, or just a dimension if it’s specified |
| torch.**dist**(_input, dim, keepdim=False, out=None_) | Computes the _p_\-norm of two tensors |
| torch.**logsumexp**(_input, dim, keepdim=False, out=None_) | Computes the log of summed exponentials of each row of the input tensor in the given dimension |
| torch.**mean**(_input, dim, keepdim=False, out=None_) | Computes the mean or average across all elements, or just a dimension if it’s specified |
| torch.**median**(_input, dim, keepdim=False, out=None_) | Computes the median or middle value across all elements, or just a dimension if it’s specified |
| torch.**mode**(_input, dim, keepdim=False, out=None_) | Computes the mode or most frequent value across all elements, or just a dimension if it’s specified |
| torch.**norm**(_input, p='fro', dim=None,__keepdim=False,__out=None, dtype=None_) | Computes the matrix or vector norm across all elements, or just a dimension if it’s specified |
| torch.**prod**(_input, dim, keepdim=False, dtype=None_) | Computes the product of all elements, or of each row of the input tensor if it’s specified |
| torch.**std**(_input, dim, keepdim=False, out=None_) | Computes the standard deviation across all elements, or just a dimension if it’s specified |
| torch.**std\_mean**(_input, unbiased=True_) | Computes the standard deviation and mean across all elements, or just a dimension if it’s specified |
| torch.**sum**(_input, dim, keepdim=False, out=None_) | Computes the sum of all elements, or just a dimension if it’s specified |
| torch.**unique**(_input, dim, keepdim=False, out=None_) | Removes duplicates across the entire tensor, or just a dimension if it’s specified |
| torch.unique\_​consecutive(_input, dim, keepdim=False, out=None_) | Similar to torch.unique() but only removes consecutive duplicates |
| torch.**var**(_input, dim, keepdim=False, out=None_) | Computes the variance across all elements, or just a dimension if it’s specified |
| torch.**var\_mean**(_input, dim, keepdim=False, out=None_) | Computes the mean and variance across all elements, or just a dimension if it’s specified |
<!-- #endregion -->

<!-- #region id="L52k6ec3FQde" -->
Note that many of these functions accept the dim parameter, which specifies the dimension of reduction for multidimensional tensors. This is similar to the axis parameter in NumPy. By default, when dim is not specified, the reduction occurs across all dimensions. Specifying dim = 1 will compute the operation across each row. For example, torch.mean(x,1) will compute the mean for each row in tensor x.
<!-- #endregion -->

<!-- #region id="1chfJ5HJF1Bp" -->
> Tip: It’s common to chain methods together. For example, torch.rand(2,2).max().item() creates a 2 × 2 tensor of random floats, finds the maximum value, and returns the value itself from the resulting tensor.
<!-- #endregion -->

<!-- #region id="qUE7FYCCF3JW" -->
Next, we’ll look at PyTorch’s comparison functions. Comparison functions usually compare all the values within a tensor, or compare one tensor’s values to another’s. They can return a tensor full of Booleans based on each element’s value such as torch.eq() or torch.is_boolean(). There are also functions to find the maximum or minimum value, sort tensor values, return the top subset of tensor elements, and more.

Table below lists some commonly used comparison functions for your reference.
<!-- #endregion -->

<!-- #region id="YXtj9JNOF-mn" -->
| Operation type | Sample functions |
| -------------- | ---------------- |
| Compare a tensor to other tensors | eq(), ge(), gt(), le(), lt(), ne() or \==, \>, \>=, <, <=, !=, respectively |
| Test tensor status or conditions | isclose(), isfinite(), isinf(), isnan() |
| Return a single Boolean for the entire tensor | allclose(), equal() |
| Find value(s) over the entire tensor or along a given dimension | argsort(), kthvalue(), max(), min(), sort(), topk() |
<!-- #endregion -->

<!-- #region id="2rqA_1urGJRf" -->
The next type of mathematical functions we’ll look at are linear algebra functions. Linear algebra functions facilitate matrix operations and are important for deep learning computations.

Many computations, including gradient descent and optimization algorithms, use linear algebra to implement their calculations. PyTorch supports a robust set of built-in linear algebra operations, many of which are based on the Basic Linear Algebra Subprograms (BLAS) and Linear Algebra Package (LAPACK) standardized libraries.
<!-- #endregion -->

<!-- #region id="vYenKx13GSTS" -->
| Function | Description |
| -------- | ----------- |
| torch.**matmul**() | Computes a matrix product of two tensors; supports broadcasting |
| torch.**chain\_matmul**() | Computes a matrix product of _N_ tensors |
| torch.**mm**() | Computes a matrix product of two tensors (if broadcasting is required, use matmul()) |
| torch.**addmm**() | Computes a matrix product of two tensors and adds it to the input |
| torch.**bmm**() | Computes a batch of matrix products |
| torch.**addbmm**() | Computes a batch of matrix products and adds it to the input |
| torch.**baddbmm**() | Computes a batch of matrix products and adds it to the input batch |
| torch.**mv**() | Computes the product of the matrix and vector |
| torch.**addmv**() | Computes the product of the matrix and vector and adds it to the input |
| torch.**matrix\_power** | Returns a tensor raised to the power of _n_ (for square tensors) |
| torch.**eig**() | Finds the eigenvalues and eigenvectors of a real square tensor |
| torch.**inverse**() | Computes the inverse of a square tensor |
| torch.**det**() | Computes the determinant of a matrix or batch of matrices |
| torch.**logdet**() | Computes the log determinant of a matrix or batch of matrices |
| torch.**dot**() | Computes the inner product of two tensors |
| torch.**addr**() | Computes the outer product of two tensors and adds it to the input |
| torch.**solve**() | Returns the solution to a system of linear equations |
| torch.**svd**() | Performs a single-value decomposition |
| torch.**pca\_lowrank**() | Performs a linear principle component analysis |
| torch.**cholesky**() | Computes a Cholesky decomposition |
| torch.**cholesky\_inverse**() | Computes the inverse of a symmetric positive definite matrix and returns the Cholesky factor |
| torch.**cholesky\_solve**() | Solves a system of linear equations using the Cholesky factor |
<!-- #endregion -->

<!-- #region id="rmCdezGDGc-1" -->
The final type of mathematical operations we’ll consider are spectral and other math operations. Depending on the domain of interest, these functions may be useful for data transforms or analysis. For example, spectral operations like the fast Fourier transform (FFT) can play an important role in computer vision or digital signal processing applications.
<!-- #endregion -->

<!-- #region id="e4pOe7roGrGN" -->
| Operation type | Sample functions |
| -------------- | ---------------- |
| Fast, inverse, and short-time Fourier transforms | fft(), ifft(), stft() |
| Real-to-complex FFT and complex-to-real inverse FFT (IFFT) | rfft(), irfft() |
| Windowing algorithms | bartlett\_window(), blackman\_window(),hamming\_window(), hann\_window() |
| Histogram and bin counts | histc(), bincount() |
| Cumulative operations | cummax(), cummin(), cumprod(), cumsum(),trace() (sum of the diagonal), </br> einsum() (sum of products using Einstein summation) |
| Normalization functions | cdist(), renorm() |
| Cross product, dot product, and Cartesian product | cross(), tensordot(), cartesian\_prod() |
| Functions that create a diagonal tensor with elements of the input tensor | diag(), diag\_embed(), diag\_flat(), diagonal() |
| Einstein summation | einsum() |
| Matrix reduction and restructuring functions | flatten(), flip(), rot90(), repeat\_interleave(), meshgrid(), roll(), combinations() |
| Functions that return the lower or upper triangles and their indices | tril(), tril\_indices, triu(), triu\_indices() |
<!-- #endregion -->

<!-- #region id="xo2INk3MG-yP" -->
One function, backward(), is worth calling out in its own subsection because it’s what makes PyTorch so powerful for deep learning development. The backward() function uses PyTorch’s automatic differentiation package, torch.autograd, to differentiate and compute gradients of tensors based on the chain rule.
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="0y6hSl9PHPwh" executionInfo={"status": "ok", "timestamp": 1631133192775, "user_tz": -330, "elapsed": 522, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="71d6cb87-37f6-414b-8427-2dfeb7cb75c5"
x = torch.tensor([[1,2,3],[4,5,6]], 
         dtype=torch.float, requires_grad=True)
print(x)
```

```python colab={"base_uri": "https://localhost:8080/"} id="ZzqsahapHWKV" executionInfo={"status": "ok", "timestamp": 1631133204109, "user_tz": -330, "elapsed": 425, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="3a474ad1-549f-4497-c43f-b6acf9b708a3"
f = x.pow(2).sum()
print(f)
```

```python colab={"base_uri": "https://localhost:8080/"} id="aajVVfvhHU1k" executionInfo={"status": "ok", "timestamp": 1631133219903, "user_tz": -330, "elapsed": 634, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="33b1db4f-a128-47f6-8475-edc6d9c05283"
f.backward()
print(x.grad) # df/dx = 2x
```
