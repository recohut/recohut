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

<!-- #region id="GvJoPkNIlpiY" -->
# PyTorch Fundamentals
<!-- #endregion -->

<!-- #region id="L_1mg9UYj7Q2" -->
## Tensors
<!-- #endregion -->

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

```python id="FsxVCauz3MKH"
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

```python id="t7tC4u3r4Fl_"
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

```python id="alVKYnj45bT9"
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

<!-- #region id="o955FXZZkCEk" -->
## Gradient Descent
<!-- #endregion -->

<!-- #region id="JJlouCu108PF" -->
we'll implement the basic functions of the Gradient Descent algorithm to find the boundary in a small dataset. First, we'll start with some functions that will help us plot and visualize the data.
<!-- #endregion -->

```python id="NSmwqCcu1L6W"
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

#Some helper functions for plotting and drawing lines

def plot_points(X, y):
    admitted = X[np.argwhere(y==1)]
    rejected = X[np.argwhere(y==0)]
    plt.scatter([s[0][0] for s in rejected], [s[0][1] for s in rejected], s = 25, color = 'blue', edgecolor = 'k')
    plt.scatter([s[0][0] for s in admitted], [s[0][1] for s in admitted], s = 25, color = 'red', edgecolor = 'k')

def display(m, b, color='g--'):
    plt.xlim(-0.05,1.05)
    plt.ylim(-0.05,1.05)
    x = np.arange(-10, 10, 0.1)
    plt.plot(x, m*x+b, color)
```

```python colab={"base_uri": "https://localhost:8080/", "height": 265} id="ZRHt-Mj01Slt" executionInfo={"status": "ok", "timestamp": 1631195611817, "user_tz": -330, "elapsed": 525, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="00c2c3d3-31ee-4e01-80f9-0f0aa9b4c2ca"
data = pd.read_csv('https://raw.githubusercontent.com/udacity/deep-learning-v2-pytorch/master/intro-neural-networks/gradient-descent/data.csv', header=None)
X = np.array(data[[0,1]])
y = np.array(data[2])
plot_points(X,y)
plt.show()
```

<!-- #region id="BvrTusgE12Xe" -->
- Sigmoid activation function

$$\sigma(x) = \frac{1}{1+e^{-x}}$$

- Output (prediction) formula

$$\hat{y} = \sigma(w_1 x_1 + w_2 x_2 + b)$$

- Error function

$$Error(y, \hat{y}) = - y \log(\hat{y}) - (1-y) \log(1-\hat{y})$$

- The function that updates the weights

$$ w_i \longrightarrow w_i + \alpha (y - \hat{y}) x_i$$

$$ b \longrightarrow b + \alpha (y - \hat{y})$$
<!-- #endregion -->

```python id="v7_iirSK1b2M"
# Activation (sigmoid) function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Output (prediction) formula
def output_formula(features, weights, bias):
    return sigmoid(np.dot(features, weights) + bias)

# Error (log-loss) formula
def error_formula(y, output):
    return - y*np.log(output) - (1 - y) * np.log(1-output)

# Gradient descent step
def update_weights(x, y, weights, bias, learnrate):
    output = output_formula(x, weights, bias)
    d_error = y - output
    weights += learnrate * d_error * x
    bias += learnrate * d_error
    return weights, bias
```

<!-- #region id="g55v6K862D3T" -->
The following training function will help us iterate the gradient descent algorithm through all the data, for a number of epochs. It will also plot the data, and some of the boundary lines obtained as we run the algorithm.
<!-- #endregion -->

```python id="_jJ3uDFO2Gq9"
np.random.seed(44)

epochs = 100
learnrate = 0.01

def train(features, targets, epochs, learnrate, graph_lines=False):
    
    errors = []
    n_records, n_features = features.shape
    last_loss = None
    weights = np.random.normal(scale=1 / n_features**.5, size=n_features)
    bias = 0
    for e in range(epochs):
        del_w = np.zeros(weights.shape)
        for x, y in zip(features, targets):
            weights, bias = update_weights(x, y, weights, bias, learnrate)
        
        # Printing out the log-loss error on the training set
        out = output_formula(features, weights, bias)
        loss = np.mean(error_formula(targets, out))
        errors.append(loss)
        if e % (epochs / 10) == 0:
            print("\n========== Epoch", e,"==========")
            if last_loss and last_loss < loss:
                print("Train loss: ", loss, "  WARNING - Loss Increasing")
            else:
                print("Train loss: ", loss)
            last_loss = loss
            
            # Converting the output (float) to boolean as it is a binary classification
            # e.g. 0.95 --> True (= 1), 0.31 --> False (= 0)
            predictions = out > 0.5
            
            accuracy = np.mean(predictions == targets)
            print("Accuracy: ", accuracy)
        if graph_lines and e % (epochs / 100) == 0:
            display(-weights[0]/weights[1], -bias/weights[1])
            

    # Plotting the solution boundary
    plt.title("Solution boundary")
    display(-weights[0]/weights[1], -bias/weights[1], 'black')

    # Plotting the data
    plot_points(features, targets)
    plt.show()

    # Plotting the error
    plt.title("Error Plot")
    plt.xlabel('Number of epochs')
    plt.ylabel('Error')
    plt.plot(errors)
    plt.show()
```

<!-- #region id="tDM8b5Kg2-9T" -->
When we run the function, we'll obtain the following:
- 10 updates with the current training loss and accuracy
- A plot of the data and some of the boundary lines obtained. The final one is in black. Notice how the lines get closer and closer to the best fit, as we go through more epochs.
- A plot of the error function. Notice how it decreases as we go through more epochs.
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 1000} id="A_zVS6ES2_KU" executionInfo={"status": "ok", "timestamp": 1631196445953, "user_tz": -330, "elapsed": 1818, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="f86b820e-f754-4b38-cb81-aecc6e226c97"
train(X, y, epochs, learnrate, True)
```

<!-- #region id="wMJvhqEB5iC-" -->
## Predicting Student Admissions with Neural Networks
In this section, we predict student admissions to graduate school at UCLA based on three pieces of data:
- GRE Scores (Test)
- GPA Scores (Grades)
- Class rank (1-4)

The dataset originally came from here: http://www.ats.ucla.edu/
<!-- #endregion -->

<!-- #region id="qQHxydHZ5qaV" -->
### Loading the data
<!-- #endregion -->

```python id="X2hOA87D5iDB" colab={"base_uri": "https://localhost:8080/", "height": 204} executionInfo={"status": "ok", "timestamp": 1631196738357, "user_tz": -330, "elapsed": 521, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="e6239a19-beb5-488d-a8e3-09927caaec92"
# Importing pandas and numpy
import pandas as pd
import numpy as np

# Reading the csv file into a pandas DataFrame
data = pd.read_csv('https://raw.githubusercontent.com/udacity/deep-learning-v2-pytorch/master/intro-neural-networks/student-admissions/student_data.csv')

# Printing out the first 10 rows of our data
data.head()
```

<!-- #region id="nHyMLmcM5iDE" -->
### Plotting the data

First let's make a plot of our data to see how it looks. In order to have a 2D plot, let's ingore the rank.
<!-- #endregion -->

```python id="lOVx1Waa5iDF" colab={"base_uri": "https://localhost:8080/", "height": 279} executionInfo={"status": "ok", "timestamp": 1631196744289, "user_tz": -330, "elapsed": 674, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="e83bb84c-0446-4354-aa89-9a3c7aa8288e"
# %matplotlib inline
import matplotlib.pyplot as plt

# Function to help us plot
def plot_points(data):
    X = np.array(data[["gre","gpa"]])
    y = np.array(data["admit"])
    admitted = X[np.argwhere(y==1)]
    rejected = X[np.argwhere(y==0)]
    plt.scatter([s[0][0] for s in rejected], [s[0][1] for s in rejected], s = 25, color = 'red', edgecolor = 'k')
    plt.scatter([s[0][0] for s in admitted], [s[0][1] for s in admitted], s = 25, color = 'cyan', edgecolor = 'k')
    plt.xlabel('Test (GRE)')
    plt.ylabel('Grades (GPA)')
    
# Plotting the points
plot_points(data)
plt.show()
```

<!-- #region id="WsL43x5v5iDG" -->
Roughly, it looks like the students with high scores in the grades and test passed, while the ones with low scores didn't, but the data is not as nicely separable as we hoped it would. Maybe it would help to take the rank into account? Let's make 4 plots, each one for each rank.
<!-- #endregion -->

```python id="XNtmwfBS5iDG" colab={"base_uri": "https://localhost:8080/", "height": 1000} executionInfo={"status": "ok", "timestamp": 1631196748158, "user_tz": -330, "elapsed": 1230, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="b52a77a1-58c1-4764-de1d-f644c9367d43"
# Separating the ranks
data_rank1 = data[data["rank"]==1]
data_rank2 = data[data["rank"]==2]
data_rank3 = data[data["rank"]==3]
data_rank4 = data[data["rank"]==4]

# Plotting the graphs
plot_points(data_rank1)
plt.title("Rank 1")
plt.show()
plot_points(data_rank2)
plt.title("Rank 2")
plt.show()
plot_points(data_rank3)
plt.title("Rank 3")
plt.show()
plot_points(data_rank4)
plt.title("Rank 4")
plt.show()
```

<!-- #region id="OtZrzIZJ5iDH" -->
This looks more promising, as it seems that the lower the rank, the higher the acceptance rate. Let's use the rank as one of our inputs. In order to do this, we should one-hot encode it.

### One-hot encoding the rank
Use the `get_dummies` function in pandas in order to one-hot encode the data.

Hint: To drop a column, it's suggested that you use `one_hot_data`[.drop( )](https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.drop.html).
<!-- #endregion -->

```python id="JH_R-AIm5iDI" colab={"base_uri": "https://localhost:8080/", "height": 359} executionInfo={"status": "ok", "timestamp": 1631196791193, "user_tz": -330, "elapsed": 498, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="03a91117-96b5-4ae0-f241-af27b22ad1d5"
# Make dummy variables for rank
one_hot_data = pd.concat([data, pd.get_dummies(data['rank'], prefix='rank')], axis=1)

# Drop the previous rank column
one_hot_data = one_hot_data.drop('rank', axis=1)

# Print the first 10 rows of our data
one_hot_data[:10]
```

<!-- #region id="j-AfPHnK5iDI" -->
### Scaling the data
The next step is to scale the data. We notice that the range for grades is 1.0-4.0, whereas the range for test scores is roughly 200-800, which is much larger. This means our data is skewed, and that makes it hard for a neural network to handle. Let's fit our two features into a range of 0-1, by dividing the grades by 4.0, and the test score by 800.
<!-- #endregion -->

```python id="X68T9T2f5iDJ" colab={"base_uri": "https://localhost:8080/", "height": 359} executionInfo={"status": "ok", "timestamp": 1631196829494, "user_tz": -330, "elapsed": 525, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="bac0abcf-e771-4ed3-c2e7-ae2bf9b6038c"
# Copying our data
processed_data = one_hot_data[:]

# Scaling the columns
processed_data['gre'] = processed_data['gre']/800
processed_data['gpa'] = processed_data['gpa']/4.0
processed_data[:10]
```

<!-- #region id="mQB4RAJh5iDK" -->
### Splitting the data into Training and Testing
<!-- #endregion -->

<!-- #region id="2nm16-HL5iDK" -->
In order to test our algorithm, we'll split the data into a Training and a Testing set. The size of the testing set will be 10% of the total data.
<!-- #endregion -->

```python id="m2MzzBT65iDK" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1631196836661, "user_tz": -330, "elapsed": 430, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="98650c0d-8e50-4003-b1b2-de9d1653e363"
sample = np.random.choice(processed_data.index, size=int(len(processed_data)*0.9), replace=False)
train_data, test_data = processed_data.iloc[sample], processed_data.drop(sample)

print("Number of training samples is", len(train_data))
print("Number of testing samples is", len(test_data))
print(train_data[:10])
print(test_data[:10])
```

<!-- #region id="CxTuMi0B5iDL" -->
### Splitting the data into features and targets (labels)
Now, as a final step before the training, we'll split the data into features (X) and targets (y).
<!-- #endregion -->

```python id="fHTwjUce5iDL" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1631196840524, "user_tz": -330, "elapsed": 417, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="4dd5991a-a39b-4a27-d890-c004409ba32b"
features = train_data.drop('admit', axis=1)
targets = train_data['admit']
features_test = test_data.drop('admit', axis=1)
targets_test = test_data['admit']

print(features[:10])
print(targets[:10])
```

<!-- #region id="YeIFAlZj5iDM" -->
### Training the 1-layer Neural Network
The following function trains the 1-layer neural network.  
First, we'll write some helper functions.
<!-- #endregion -->

```python id="obL2qqRx5iDM"
# Activation (sigmoid) function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_prime(x):
    return sigmoid(x) * (1-sigmoid(x))
    
def error_formula(y, output):
    return - y*np.log(output) - (1 - y) * np.log(1-output)
```

<!-- #region id="X1rQSGrW5iDM" -->
### Backpropagate the error
Now it's your turn to shine. Write the error term. Remember that this is given by the equation $$ (y-\hat{y})x $$ for binary cross entropy loss function and 
$$ (y-\hat{y})\sigma'(x)x $$ for mean square error. 
<!-- #endregion -->

```python id="6l8htF5M5iDN"
def error_term_formula(x, y, output):
#    for binary cross entropy loss
    return (y - output)*x
#    for mean square error
#    return (y - output)*sigmoid_prime(x)*x
```

```python id="xkPopj_E5iDN" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1631196887826, "user_tz": -330, "elapsed": 3917, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="9eb95e1e-bd44-4592-815d-8c00e5863c42"
# Neural Network hyperparameters
epochs = 1000
learnrate = 0.0001

# Training function
def train_nn(features, targets, epochs, learnrate):
    
    # Use to same seed to make debugging easier
    np.random.seed(42)

    n_records, n_features = features.shape
    last_loss = None

    # Initialize weights
    weights = np.random.normal(scale=1 / n_features**.5, size=n_features)

    for e in range(epochs):
        del_w = np.zeros(weights.shape)
        for x, y in zip(features.values, targets):
            # Loop through all records, x is the input, y is the target

            # Activation of the output unit
            #   Notice we multiply the inputs and the weights here 
            #   rather than storing h as a separate variable 
            output = sigmoid(np.dot(x, weights))

            # The error term
            error_term = error_term_formula(x, y, output)

            # The gradient descent step, the error times the gradient times the inputs
            del_w += error_term

        # Update the weights here. The learning rate times the 
        # change in weights
        # don't have to divide by n_records since it is compensated by the learning rate
        weights += learnrate * del_w #/ n_records  

        # Printing out the mean square error on the training set
        if e % (epochs / 10) == 0:
            out = sigmoid(np.dot(features, weights))
            loss = np.mean(error_formula(targets, out))
            print("Epoch:", e)
            if last_loss and last_loss < loss:
                print("Train loss: ", loss, "  WARNING - Loss Increasing")
            else:
                print("Train loss: ", loss)
            last_loss = loss
            print("=========")
    print("Finished training!")
    return weights
    
weights = train_nn(features, targets, epochs, learnrate)
```

<!-- #region id="UpSbWJFB5iDO" -->
### Calculating the Accuracy on the Test Data
<!-- #endregion -->

```python id="AReeZ67y5iDO" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1631196888241, "user_tz": -330, "elapsed": 6, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="2339ead3-1b6e-4fee-98a7-3f7920ceb00e"
# Calculate accuracy on test data
test_out = sigmoid(np.dot(features_test, weights))
predictions = test_out > 0.5
accuracy = np.mean(predictions == targets_test)
print("Prediction accuracy: {:.3f}".format(accuracy))
```

<!-- #region id="WD2vgrnLlJLn" -->
## Vision model on CIFAR image dataset
<!-- #endregion -->

<!-- #region id="xCdsm69B-zyJ" -->
You’ll build a deep learning model and train the model using a common training loop structure. Then, you’ll test your model’s performance and tweak hyperparameters to improve your results and training speed. Finally, we’ll explore ways to deploy your model to prototype systems or production.
<!-- #endregion -->

<!-- #region id="X_NpMaWa-0It" -->
First, we load this data and convert it to numeric values in the form of tensors. The tensors will act as inputs during the model training stage; however, before they are passed in, the tensors are usually preprocessed via transforms and grouped into batches for better training performance. Thus, the data preparation stage takes generic data and converts it to batches of tensors that can be passed into your NN model.
<!-- #endregion -->

<!-- #region id="w3DtG7yY_FSm" -->
Next, in the model experimentation and development stage, we will design an NN model, train the model with our training data, test its performance, and optimize our hyperparameters to improve performance to a desired level. To do so, we will separate our dataset into three parts: one for training, one for validation, and one for testing. We’ll design an NN model and train its parameters with our training data. PyTorch provides elegantly designed modules and classes in the torch.nn module to help you create and train your NNs. We will define a loss function and optimizer from a selection of the many built-in PyTorch functions. Then we’ll perform backpropagation and update the model parameters in our training loop.
<!-- #endregion -->

<!-- #region id="MDCzltgR_Wf7" -->
Within each epoch, we’ll also validate our model by passing in validation data, measuring performance, and potentially tuning hyperparameters. Finally, we’ll test our model by passing in test data and measuring the model’s performance against unseen data. In practice, validation and test loops may be optional, but we show them here for completeness.
<!-- #endregion -->

<!-- #region id="JJltU6wr_h7o" -->
The last stage of deep learning model development is the model deployment stage. In this stage, we have a fully trained model—so what do we do with it? If you are a deep learning research scientist conducting experiments, you may want to simply save the model to a file and load it for further research and experimentation, or you may want to provide access to it via a repository like PyTorch Hub. You may also want to deploy it to an edge device or local server to demonstrate a prototype or a proof of concept.

On the other hand, if you are a software developer or systems engineer, you may want to deploy your model to a product or service. In this case, you can deploy your model to a production environment on a cloud server or deploy it to an edge device or mobile phone. When deploying trained models, the model often requires additional postprocessing. For example, you may classify a batch of images, but you only want to report the most confident result. The model deployment stage also handles any postprocessing that is needed to go from your model’s output values to the final solution.
<!-- #endregion -->

<!-- #region id="Ml-b4gkD_xYS" -->
PyTorch provides powerful built-in classes and utilities, such as the Dataset, DataLoader, and Sampler classes, for loading various types of data. The Dataset class defines how to access and preprocess data from a file or data sources. The Sampler class defines how to sample data from a dataset in order to create batches, while the DataLoader class combines a dataset with a sampler and allows you to iterate over a set of batches.
<!-- #endregion -->

```python id="pIC4ZPIOATAb"
import torch
import torchvision

from torchvision.datasets import CIFAR10
```

```python colab={"base_uri": "https://localhost:8080/", "height": 83, "referenced_widgets": ["3b1bbde53e234e2ea985f0f07877c792", "0a799c17626b43b8b38b79dacde55151", "dfee4821887947a08e77015f08dbb098", "d21e616effd549e88a3b6ae5a9111dd3", "b6b4b2c8fb56426391335802bb4a308b", "e1ea823e09fc45dc8ae837c9f125f3f3", "ba713e8d5e2a4fbd985879ace34a8ab2", "f40e9824762146c8b7da2a0fa7c40899", "85df6f0c4928469dbed5c8eb554e612e", "5e10a87722e747c3861e535c1bbcb6ee", "0a9fb274ebdb43d5901e59bd3c9caacb"]} id="d-kAioq6BZMk" executionInfo={"status": "ok", "timestamp": 1631165207095, "user_tz": -330, "elapsed": 7346, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="70f85ff8-da99-413f-fc9a-570e6f7f73d3"
train_data = CIFAR10(root="./train/",
                    train=True, 
                    download=True)
```

```python colab={"base_uri": "https://localhost:8080/"} id="qcw-ruQ6BbLr" executionInfo={"status": "ok", "timestamp": 1631165341115, "user_tz": -330, "elapsed": 656, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="d4e36831-db04-43cd-fc06-045b1a60256b"
print(train_data)
print(len(train_data))
print(train_data.data.shape)
print(len(train_data.targets))
print(train_data.classes)
print(train_data.class_to_idx)
print(type(train_data[0]))
print(len(train_data[0]))

data, label = train_data[0]
print(type(data))
print(data)
print(type(label))
print(label)
print(train_data.classes[label])
```

```python colab={"base_uri": "https://localhost:8080/", "height": 83, "referenced_widgets": ["81e9d38f7d484dd790e7b6927de2b447", "be1fa3882b9646aa84f518b34b4382a8", "4986bf68729f42a283f2ff29fb07e293", "ce6f449b2b94421c9d71c9c01d9c0026", "6d82a6732a28409aadba1a17802e0a58", "c239f2f37cee48e68ec47b676dd5be3b", "aaf297df8b214dabb0b7cfc4153b25e3", "566005b140a843ec95618ad33f3ee20b", "295277ab3eb84e4180f04e3cd3c49f1e", "433630fb825a4f518a328b59d93a70ad", "1063afb7a51f4e7bb8c48189cac7a4b5"]} id="X8PkwxC_Bbar" executionInfo={"status": "ok", "timestamp": 1631165383490, "user_tz": -330, "elapsed": 6130, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="3ddb8d77-9593-4b61-e987-fa18fb6dd55c"
test_data = CIFAR10(root="./test/", 
                    train=False, 
                    download=True)
```

```python colab={"base_uri": "https://localhost:8080/"} id="l2PA41w2B2zb" executionInfo={"status": "ok", "timestamp": 1631165383493, "user_tz": -330, "elapsed": 9, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="6b390aaa-e9e8-42f2-c080-8dfd465ae69b"
print(test_data)
print(len(test_data))
print(test_data.data.shape)
```

```python colab={"base_uri": "https://localhost:8080/"} id="JtXwIBBTCG08" executionInfo={"status": "ok", "timestamp": 1631165511329, "user_tz": -330, "elapsed": 1533, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="3cfbcc61-a412-46a9-94d6-e65a195a815e"
from torchvision import transforms

train_transforms = transforms.Compose([
  transforms.RandomCrop(32, padding=4),
  transforms.RandomHorizontalFlip(),
  transforms.ToTensor(),
  transforms.Normalize(
      (0.4914, 0.4822, 0.4465),
      (0.2023, 0.1994, 0.2010))])

train_data = CIFAR10(root="./train/",
                    train=True, 
                    download=True,
                    transform=train_transforms)

print(train_data)
print(train_data.transforms)

data, label = train_data[0]
print(type(data))
print(data.size())
print(data)
```

```python colab={"base_uri": "https://localhost:8080/"} id="vbfud9n2Cdk3" executionInfo={"status": "ok", "timestamp": 1631165571307, "user_tz": -330, "elapsed": 1019, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="324530ee-f55b-42ae-95c1-773f4b8712d2"
test_transforms = transforms.Compose([
  transforms.ToTensor(),
  transforms.Normalize(
      (0.4914, 0.4822, 0.4465),
      (0.2023, 0.1994, 0.2010))])

test_data = torchvision.datasets.CIFAR10(
      root="./test/", 
      train=False, 
      transform=test_transforms)

print(test_data)
```

<!-- #region id="-5xEDXNeC1o-" -->
Now that we have defined the transforms and created the datasets, we can access data samples one at a time. However, when you train your model, you will want to pass in small batches of data at each iteration. Sending data in batches not only allows more efficient training but also takes advantage of the parallel nature of GPUs to accelerate training.

Batch processing can easily be implemented using the torch.utils.data.DataLoader class. Let’s start with an example of how Torchvision uses this class, and then we’ll cover it in more detail.
<!-- #endregion -->

```python id="SS_RrZScDFvw"
trainloader = torch.utils.data.DataLoader(
                    train_data,
                    batch_size=16,
                    shuffle=True)
```

```python id="x-IEueVMDs-w"
testloader = torch.utils.data.DataLoader(
                    test_data,
                    batch_size=16,
                    shuffle=False)
```

<!-- #region id="JW_yheMmDR9m" -->
The dataloader object combines a dataset and a sampler, and provides an iterable over the given dataset. In other words, your training loop can use this object to sample your dataset and apply transforms one batch at a time instead of applying them for the complete dataset at once. This considerably improves efficiency and speed when training and testing models.
<!-- #endregion -->

<!-- #region id="7QQ6ss7YDSPi" -->
The following code shows how to retrieve a batch of samples from the trainloader:
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="yFxW7zDFDaCU" executionInfo={"status": "ok", "timestamp": 1631165738759, "user_tz": -330, "elapsed": 376, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="328cc60b-8537-4e7c-f2e5-918e7e81c0c5"
data_batch, labels_batch = next(iter(trainloader))

print(data_batch.size())
print(labels_batch.size())
```

<!-- #region id="DJcGmSRLDerp" -->
We need to use iter() to cast the trainloader to an iterator and then use next() to iterate over the data one more time. This is only necessary when accessing one batch. As we’ll see later, our training loops will access the dataloader directly without the need for iter() and next(). After checking the sizes of the data and labels, we see they return batches of size 16.
<!-- #endregion -->

<!-- #region id="Te-zeXO7DpsL" -->
So far, I’ve shown you how to load, transform, and batch image data using Torchvision. However, you can use PyTorch to prepare other types of data as well. PyTorch libraries such as Torchtext and Torchaudio provide dataset and dataloader classes for text and audio data, and new external libraries are being developed all the time.

PyTorch also provides a submodule called torch.utils.data that you can use to create your own dataset and dataloader classes like the ones you saw in Torchvision. It consists of Dataset, Sampler, and DataLoader classes.
<!-- #endregion -->

<!-- #region id="semtaM19D4jV" -->
PyTorch supports map- and iterable-style dataset classes. A map-style dataset is derived from the abstract class torch.utils.data.Dataset. It implements the getitem() and len() functions, and represents a map from (possibly nonintegral) indices/keys to data samples. For example, such a dataset, when accessed with dataset[idx], could read the idx-th image and its corresponding label from a folder on the disk. Map-style datasets are more commonly used than iterable-style datasets, and all datasets that represent a map made from keys or data samples should use this subclass.
<!-- #endregion -->

<!-- #region id="YtARpb62O95_" -->
All subclasses should overwrite getitem(), which fetches a data sample for a given key. Subclasses can also optionally overwrite len(), which returns the size of the dataset by many Sampler implementations and the default options of DataLoader.
<!-- #endregion -->

<!-- #region id="DDbVqGrpPLFv" -->
An iterable-style dataset, on the other hand, is derived from the torch.utils.data.IterableDataset abstract class. It implements the iter() protocol and represents an iterable over data samples. This type of dataset is typically used when reading data from a database or a remote server, as well as data generated in real time. Iterable datasets are useful when random reads are expensive or uncertain, and when the batch size depends on fetched data.
<!-- #endregion -->

<!-- #region id="jKbI_kYyPeEV" -->
In addition to dataset classes PyTorch also provides sampler classes, which offer a way to iterate over indices of dataset samples. Sampler are derived from the torch.utils.data.Sampler base class.

Every Sampler subclass needs to implement an iter() method to provide a way to iterate over indices of dataset elements and a len() method that returns the length of the returned iterators.
<!-- #endregion -->

<!-- #region id="P4-YSshXQCyK" -->
The dataset and sampler objects are not iterables, meaning you cannot run a for loop on them. The dataloader object solves this problem. The Dataset class returns a dataset object that includes data and information about the data. The Sampler class returns the actual data itself in a specified or random fashion. The DataLoader class combines a dataset with a sampler and returns an iterable.
<!-- #endregion -->

<!-- #region id="XlIm6jP-QKRZ" -->
One of the most powerful features of PyTorch is its Python module torch.nn, which makes it easy to design and experiment with new models. The following code illustrates how you can create a simple model with torch.nn. In this example, we will create a fully connected model called SimpleNet. It consists of an input layer, a hidden layer, and an output layer that takes in 2,048 input values and returns 2 output values for classification:
<!-- #endregion -->

```python id="l4Td7yLBQ8Lt"
import torch.nn as nn
import torch.nn.functional as F

class SimpleNet(nn.Module):

    def __init__(self):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(2048, 256)
        self.fc2 = nn.Linear(256, 64)
        self.fc3 = nn.Linear(64,2)

    def forward(self, x):
        x = x.view(-1, 2048)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.softmax(self.fc3(x),dim=1)
        return x
```

```python colab={"base_uri": "https://localhost:8080/"} id="Ub45qjGyRTLT" executionInfo={"status": "ok", "timestamp": 1631169362176, "user_tz": -330, "elapsed": 5, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="4265488e-5330-400f-c02b-ab43a6a66723"
simplenet = SimpleNet()
print(simplenet)
```

<!-- #region id="zaOuo05hRTXq" -->
This simple model demonstrates the following decisions you need to make during model design:
1. **Module definition**: How will you define the layers of your NN? How will you combine these layers into building blocks? In the example, we chose three linear or fully connected layers.
2. **Activation functions**: Which activation functions will you use at the end of each layer or module? In the example, we chose to use relu activation for the input and hidden layers and softmax for the output layer.
3. **Module connections**: How will your modules be connected to each other? In the example, we chose to simply connect each linear layer in sequence.
4. **Output selection**: What output values and formats will be returned? In this example, we return two values from the softmax() function.
<!-- #endregion -->

<!-- #region id="ZiE7TMF-R_z1" -->
The next step in model development is to train your model with your training data. Training a model involves nothing more than estimating the model’s parameters, passing in data, and adjusting the parameters to achieve a more accurate representation of how the data is generally modeled.

In other words, you set the parameters to some values, pass through data, and then compare the model’s outputs with true outputs to measure the error. The goal is to change the parameters and repeat the process until the error is minimized and the model’s outputs are the same as the true outputs.
<!-- #endregion -->

<!-- #region id="EGhjUdCvTJ5W" -->
In this example, we will train the LeNet5 model with the CIFAR-10 dataset that we used earlier in this chapter. The LeNet5 model is a simple convolutional NN developed by Yann LeCun and his team at Bell Labs in the 1990s to classify hand-written digits. (Unbeknownst to me at the time, I actually worked for Bell Labs in the same building in Holmdel, NJ, while this work was being performed.)
<!-- #endregion -->

```python id="qyM0Bi4iTgUu"
from torch import nn
import torch.nn.functional as F

class LeNet5(nn.Module):
    def __init__(self):
        super(LeNet5, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5) # <1>
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, int(x.nelement() / x.shape[0]))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

device = "cuda" if torch.cuda.is_available() else "cpu"
model = LeNet5().to(device=device)
```

<!-- #region id="JV3k8lOLTszU" -->
Next, we need to define the loss function (which is also called the criterion) and the optimizer algorithm. The loss function determines how we measure the performance of our model and computes the loss or error between predictions and truth. We’ll attempt to minimize the loss by adjusting the model parameters during training. The optimizer defines how we update our model’s parameters during training.

To define the loss function and the optimizer, we use the torch.optim and torch.nn packages as shown in the following code:
<!-- #endregion -->

```python id="xR3rFFNiTtDZ"
from torch import optim
from torch import nn

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(),
                      lr=0.001, 
                      momentum=0.9)
```

```python colab={"base_uri": "https://localhost:8080/"} id="xArCva7zT5fM" executionInfo={"status": "ok", "timestamp": 1631170522614, "user_tz": -330, "elapsed": 341743, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="afecda60-2966-473d-c9c7-72fe74dd7936"
N_EPOCHS = 10 
for epoch in range(N_EPOCHS): # <1>

    epoch_loss = 0.0
    for inputs, labels in trainloader:
        inputs = inputs.to(device) # <2>
        labels = labels.to(device)

        optimizer.zero_grad() # <3>

        outputs = model(inputs) # <4>
        loss = criterion(outputs, labels) # <5>
        loss.backward() # <6>
        optimizer.step() # <7>

        epoch_loss += loss.item() # <8>
    print("Epoch: {} Loss: {}".format(epoch, 
                  epoch_loss/len(trainloader)))
```

<!-- #region id="nzNfnkUgUbQF" -->
1. Outer training loop; loop over 10 epochs.
2. Move inputs and labels to GPU if available.
3. Zero out gradients before each backpropagation pass, or they’ll accumulate.
4. Perform forward pass.
5. Compute loss.
6. Perform backpropagation; compute gradients.
7. Adjust parameters based on gradients.
8. Accumulate batch loss so we can average over the epoch.
<!-- #endregion -->

<!-- #region id="7izhHuA_Uo70" -->
The training loop consists of two loops. In the outer loop, we will process the entire set of training data during every iteration or epoch. However, instead of waiting to process the entire dataset before updating the model’s parameters, we process smaller batches of data, one batch at a time. The inner loop loops over each batch.
<!-- #endregion -->

<!-- #region id="1i5HdMfOU7Uv" -->
> Warning: By default, PyTorch accumulates the gradients during each call to loss.backward() (i.e., the backward pass). This is convenient while training some types of NNs, such as RNNs; however, it is not desired for convolutional neural networks (CNNs). In most cases, you will need to call optimizer.zero_grad() to zero the gradients before doing backpropagation so the optimizer updates the model parameters correctly.
<!-- #endregion -->

<!-- #region id="tNCw6Ci5Wv9u" -->
Now that we have trained our model and attempted to minimize the loss, how can we evaluate its performance? How do we know that our model will generalize and work with data it has never seen before?

Model development often includes validation and testing loops to ensure that overfitting does not occur and that the model will perform well against unseen data. Let’s address validation first. Here, I’ll provide you with a quick reference for how you can add validation to your training loops with PyTorch.

Typically, we will reserve a portion of the training data for validation. The validation data will not be used to train the NN; instead, we’ll use it to test the performance of the model at the end of each epoch.

Validation is good practice when training your models. It’s commonly performed when adjusting hyperparameters. For example, maybe we want to slow down the learning rate after five epochs.
<!-- #endregion -->

<!-- #region id="g00vfkZEW80J" -->
Before we perform validation, we need to split our training dataset into a training dataset and a validation dataset. We use the random_split() function from torch.utils.data to reserve 10,000 of our 50,000 training images for validation. Once we create our train_set and val_set, we create our dataloaders for each one.
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="viaCRVKdXJer" executionInfo={"status": "ok", "timestamp": 1631170911674, "user_tz": -330, "elapsed": 497, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="6ae8b391-d7bd-4caa-918d-bfb817416e10"
from torch.utils.data import random_split

train_set, val_set = random_split(
                      train_data,
                      [40000, 10000])

trainloader = torch.utils.data.DataLoader(
                    train_set,
                    batch_size=16,
                    shuffle=True)

valloader = torch.utils.data.DataLoader(
                    val_set,
                    batch_size=16,
                    shuffle=True)

print(len(trainloader))
print(len(valloader))
```

<!-- #region id="-wY5C8ycX4hI" -->
If the loss decreases for validation data, then the model is doing well. However, if the training loss decreases but the validation loss does not, then there’s a good chance the model is overfitting.
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="q3JcA2nLXjFO" executionInfo={"status": "ok", "timestamp": 1631171348512, "user_tz": -330, "elapsed": 333293, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="c69de3b0-c8c2-436e-fa85-13b50510abcb"
from torch import optim
from torch import nn

model = LeNet5().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), 
                      lr=0.001, 
                      momentum=0.9)

N_EPOCHS = 10
for epoch in range(N_EPOCHS):

    # Training 
    train_loss = 0.0
    model.train() # <1>
    for inputs, labels in trainloader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    # Validation
    val_loss = 0.0
    model.eval() # <2>
    for inputs, labels in valloader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        outputs = model(inputs)
        loss = criterion(outputs, labels)

        val_loss += loss.item()

    print("Epoch: {} Train Loss: {} Val Loss: {}".format(
                  epoch, 
                  train_loss/len(trainloader), 
                  val_loss/len(valloader)))
```

<!-- #region id="eFNMCQiyXm62" -->
> Note: Running the .train() or .eval() method on your model object puts the model in training or testing mode, respectively. Calling these methods is only necessary if your model operates differently for training and evaluation. For example, dropout and batch normalization are used in training but not in validation or testing. It’s good practice to call .train() and .eval() in your loops.
<!-- #endregion -->

<!-- #region id="M2AsMS_YX03g" -->
As you can see, our model is training well and does not seem to be overfitting, since both the training loss and the validation loss are decreasing. If we train the model for more epochs, we may get even better results.

We’re not quite finished, though. Our model may still be overfitting. We might have just gotten lucky with our choice of hyperparameters, leading to good validation results. As a further test against overfitting, we will run some test data through our model.

The model has never seen the test data during training, nor has the test data had any influence on the hyperparameters. Let’s see how we perform against the test dataset.
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="GXWxEZP8YBFl" executionInfo={"status": "ok", "timestamp": 1631171352290, "user_tz": -330, "elapsed": 3808, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="139bdffb-e3b6-48ff-90e9-a3dd72dd011a"
num_correct = 0.0

for x_test_batch, y_test_batch in testloader:
  model.eval()
  y_test_batch = y_test_batch.to(device)
  x_test_batch = x_test_batch.to(device)
  y_pred_batch = model(x_test_batch)
  _, predicted = torch.max(y_pred_batch, 1)
  num_correct += (predicted == y_test_batch).float().sum()
  
accuracy = num_correct/(len(testloader)*testloader.batch_size) 

print(len(testloader), testloader.batch_size)

print("Test Accuracy: {}".format(accuracy))
```

<!-- #region id="cosznu2cYl3A" -->
> Tip: You now know how to create training, validation, and test loops using PyTorch. Feel free to use this code as a reference when creating your own loops.
<!-- #endregion -->

<!-- #region id="xu8H1j2mYdUl" -->
Now that you have a fully trained model, let’s explore what you can do with it in the model deployment stage. One of the simplest things you can do is save your trained model for future use. When you want to run your model against new inputs, you can simply load it and call the model with the new values.

The following code illustrates the recommended way to save and load a trained model. It uses the state_dict() method, which creates a dictionary object that maps each layer to its parameter tensor. In other words, we only need to save the model’s learned parameters. We already have the model’s design defined in our model class, so we don’t need to save the architecture. When we load the model, we use the constructor to create a “blank model,” and then we use load_state_dict() to set the parameters for each layer:
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="iiJyV2pvYpiv" executionInfo={"status": "ok", "timestamp": 1631171428335, "user_tz": -330, "elapsed": 408, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="86a16ec6-0b0d-48e0-d2b1-0a8029fe91d5"
torch.save(model.state_dict(), "./lenet5_model.pt")

model = LeNet5().to(device)
model.load_state_dict(torch.load("./lenet5_model.pt"))
```

<!-- #region id="PF7IUdY_ZT0B" -->
> Note: A common PyTorch convention is to save models using either a .pt or .pth file extension.
<!-- #endregion -->
