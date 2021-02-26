# Introducing Libraries: NumPy

![libgif](https://media0.giphy.com/media/7E8lI6TkLrvvAcPXso/giphy.gif?cid=790b76115d360a95792e4333770609b8&rid=giphy.gif)

<h1>Table of Contents<span class="tocSkip"></span></h1>
<div class="toc"><ul class="toc-item"><li><span><a href="#Introducing-Libraries:-NumPy" data-toc-modified-id="Introducing-Libraries:-NumPy-1">Introducing Libraries: NumPy</a></span><ul class="toc-item"><li><span><a href="#Introduction/Overview" data-toc-modified-id="Introduction/Overview-1.1">Introduction/Overview</a></span></li></ul></li><li><span><a href="#Importing-Python-packages" data-toc-modified-id="Importing-Python-packages-2">Importing Python packages</a></span><ul class="toc-item"><li><span><a href="#Terminology" data-toc-modified-id="Terminology-2.1">Terminology</a></span></li><li><span><a href="#pip-&amp;-the-Python-Package-Index" data-toc-modified-id="pip-&amp;-the-Python-Package-Index-2.2">pip &amp; the Python Package Index</a></span></li></ul></li><li><span><a href="#First-library-we-will-import-is-Numpy" data-toc-modified-id="First-library-we-will-import-is-Numpy-3">First library we will import is <code>Numpy</code></a></span><ul class="toc-item"><li><span><a href="#NumPy-versus-base-Python" data-toc-modified-id="NumPy-versus-base-Python-3.1">NumPy versus base Python</a></span></li></ul></li><li><span><a href="#Making-Your-Own-Module:-You're-not-limited-to-PyPI" data-toc-modified-id="Making-Your-Own-Module:-You're-not-limited-to-PyPI-4">Making Your Own Module: You're not limited to PyPI</a></span></li><li><span><a href="#⏰-ACTIVITY:-Making-our-own-module" data-toc-modified-id="⏰-ACTIVITY:-Making-our-own-module-5">⏰ ACTIVITY: Making our own module</a></span><ul class="toc-item"><li><span><a href="#Next:-Adding-New-Functions" data-toc-modified-id="Next:-Adding-New-Functions-5.1">Next: Adding New Functions</a></span></li><li><span><a href="#Congrats!!" data-toc-modified-id="Congrats!!-5.2">Congrats!!</a></span></li></ul></li><li><span><a href="#Appendix" data-toc-modified-id="Appendix-6">Appendix</a></span></li></ul></div>

## Introduction/Overview

### Learning Objectives

#### Our goals today are to be able to:

- Identify and import Python modules and packages (libraries)
- Identify differences between NumPy and base Python in usage and operation
- Create a new module of our own

#### Big questions for this lesson

- What is a package, what do packages do, and why might we want to use them?
- When do we want to use NumPy?
- How can we reuse our own functions beyond one notebook?


# Importing Python packages 

In an earlier lesson, we wrote a function to calculate the mean of an list. That was **tedious**. To make our code efficient we could store that function in a *python module* and call it later when we need it. 

And thankfully, other people have _also_ wrote and optimized functions and wrapped them into **modules** and **packages** (also known as _libraries_ )

![numpy](https://raw.githubusercontent.com/donnemartin/data-science-ipython-notebooks/master/images/numpy.png)

[NumPy](https://www.numpy.org/) is the fundamental package for scientific computing with Python. 


To import a package type `import` followed by the name of the library as shown below.

## Terminology

![mod2](https://raw.githubusercontent.com/jirvingphd/dsc-lp-libraries-numpy/master/img/modules2.png)

![packages3](https://raw.githubusercontent.com/jirvingphd/dsc-lp-libraries-numpy/master/img/packages3.png)

## pip & the Python Package Index

![python-fact](https://raw.githubusercontent.com/jirvingphd/dsc-lp-libraries-numpy/master/img/python_def.png)

![pypi](https://raw.githubusercontent.com/jirvingphd/dsc-lp-libraries-numpy/master/img/pypi_packages.png)


```python
## One of My PyPi packages for flatiron school data science  (fsds)
# !pip install -U fsds
# from fsds.imports import *
```

# First library we will import is `Numpy`


![numpy](https://raw.githubusercontent.com/donnemartin/data-science-ipython-notebooks/master/images/numpy.png)

[NumPy](https://www.numpy.org/) is the fundamental package for scientific computing with Python. 


To import a package type `import` followed by the name of the package as shown below.


```python
import numpy # Look, ma! we're importing!
numpy
```




    <module 'numpy' from '/opt/anaconda3/envs/learn-env/lib/python3.6/site-packages/numpy/__init__.py'>




```python
l = [1,2,3]
x=numpy.array([1,2,3])
x
```




    array([1, 2, 3])



#### New type of object


```python
type(x)
```




    numpy.ndarray



#### Alias libraries

Many packages have a canonical way to import them with an abbreviated alias.


```python
import numpy as np # np = alias
y=np.array([4,5,6])
y
```




    array([4, 5, 6])



#### Other standard aliases 


```python
import scipy
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels as sm
```

### Import specific modules from a larger package


```python
# sometimes we will want to import a specific module from a package
import matplotlib.pyplot as plt
from matplotlib import pyplot as plt 
```

What happens if we mess with naming conventions? For example, import one of our previous libraries as `print`.


>**PLEASE NOTE THAT WE WILL HAVE TO RESET THE KERNEL AFTER RUNNING THIS.**<br> Comment out your code after running it.



```python
# import seaborn as print
```


```python
#Did we get an error? What about when we run the following command?

print(x)

#Restart your kernel and clear cells
```

    [1 2 3]


#### Helpful links: package documentation

Packages have associated documentation to explain how to use the different tools included in a package.

_Sample of libraries_
- [NumPy](https://docs.scipy.org/doc/numpy/)
- [SciPy](https://docs.scipy.org/doc/scipy/reference/)
- [Pandas](http://pandas.pydata.org/pandas-docs/stable/)
- [Matplotlib](https://matplotlib.org/contents.html)

## NumPy versus base Python

Now that we know packages exist, why do we want to use them? Let us examine a comparison between base Python and Numpy.

Python has lists and normal python can do basic math. NumPy, however, has the helpful objects called **arrays**.

Numpy has a few advantages over base Python which we will look at.

### Numpy makes math easy

Because of numpy we can now get the **mean** and other quick math of lists and arrays.


```python
example = [4,3,25,40,62,20]
print(np.mean(example))
```

    25.666666666666668


#### Different types of arrays


```python
names_list=['Bob','John','Sally']
names_array=numpy.char.array(['Bob','John','Sally']) #use numpy.array for numbers and numpy.char.array for strings
print(names_list)
print(names_array)
```

    ['Bob', 'John', 'Sally']
    ['Bob' 'John' 'Sally']



```python
display(names_list)
display(names_array)
```


    ['Bob', 'John', 'Sally']



    chararray(['Bob', 'John', 'Sally'], dtype='<U5')


#### Array math in action


```python
# Make a list and an array of three numbers

#your code here
numbers_list = [5,22,33,90]
numbers_array = np.array([5,22,33,90])
```


```python
# divide your array by 2

numbers_array/2
```




    array([ 2.5, 11. , 16.5, 45. ])




```python
# divide your list by 2

numbers_list/2
```


    ---------------------------------------------------------------------------

    TypeError                                 Traceback (most recent call last)

    <ipython-input-15-a9b88a30b652> in <module>
          1 # divide your list by 2
          2 
    ----> 3 numbers_list/2
    

    TypeError: unsupported operand type(s) for /: 'list' and 'int'


Numpy arrays support the `div()` operator while python lists do not. There are other things that make it useful to utilize numpy over base python for evaluating data.


```python
# shape tells us the size of the array

numbers_array.shape
```




    (4,)



### Slicing in NumPy


```python
# We remember slicing from lists
numbers_list = list(range(10))
numbers_list
```




    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]




```python
numbers_list[3:7]
```




    [3, 4, 5, 6]




```python
# Slicing in NumPy Arrays is very similar!
a = np.array([[1,2,3,4], [5,6,7,8], [9,10,11,12]])
print(a.shape)
a
```

    (3, 4)





    array([[ 1,  2,  3,  4],
           [ 5,  6,  7,  8],
           [ 9, 10, 11, 12]])




```python
a.shape
```




    (3, 4)




```python
a[:,2]
```




    array([ 3,  7, 11])




```python
# first 2 rows, columns 1 & 2 (remember 0-index!)
b = a[:2, 1:3]
b
```




    array([[2, 3],
           [6, 7]])



### Datatypes in NumPy



```python
a.dtype
```




    dtype('int64')




```python
names_list.dtype
```


    ---------------------------------------------------------------------------

    AttributeError                            Traceback (most recent call last)

    <ipython-input-24-7b8bd1c66aa1> in <module>
    ----> 1 names_list.dtype
    

    AttributeError: 'list' object has no attribute 'dtype'



```python
a.astype(np.float64)
```




    array([[ 1.,  2.,  3.,  4.],
           [ 5.,  6.,  7.,  8.],
           [ 9., 10., 11., 12.]])



### More Array Math 
#### Adding matrices 


```python
x = np.array([[1,2],[3,4]], dtype=np.float64)
y = np.array([[5,6],[7,8]], dtype=np.float64)

# Elementwise sum; both produce the array
# [[ 6.0  8.0]
#  [10.0 12.0]]
print(x + y)
```

    [[ 6.  8.]
     [10. 12.]]



```python
print(np.add(x, y))
```

    [[ 6.  8.]
     [10. 12.]]


#### Subtracting matrices 


```python
# Elementwise difference; both produce the array
# [[-4.0| -4.0]
#  [-4.0 -4.0]]
print(x - y)
```

    [[-4. -4.]
     [-4. -4.]]



```python
print(np.subtract(x, y))
```

    [[-4. -4.]
     [-4. -4.]]


#### Multiplying matrices 


```python
# Elementwise product; both produce the array
# [[ 5.0 12.0]
#  [21.0 32.0]]
print(x * y)
```

    [[ 5. 12.]
     [21. 32.]]



```python
print(np.multiply(x, y))
```

    [[ 5. 12.]
     [21. 32.]]


#### Dividing matrices 


```python
# Elementwise division; both produce the array
# [[ 0.2         0.33333333]
#  [ 0.42857143  0.5       ]]
print(x / y)
```

    [[0.2        0.33333333]
     [0.42857143 0.5       ]]



```python
print(np.divide(x, y))
```

    [[0.2        0.33333333]
     [0.42857143 0.5       ]]


#### Raising matrices to powers 


```python
# Elementwise square root; both produce the same array
# [[ 1.          1.41421356]
#  [ 1.73205081  2.        ]]
print(x ** (1/2))
```

    [[1.         1.41421356]
     [1.73205081 2.        ]]



```python
print(np.sqrt(x))
```

    [[1.         1.41421356]
     [1.73205081 2.        ]]


### Numpy is faster

Below, you will find a piece of code we will use to compare the speed of operations on a list and operations on an array. In this speed test, we will use the package [time](https://docs.python.org/3/library/time.html).


```python
import time
import numpy as np

size_of_vec = 1000

def pure_python_version():
    t1 = time.time()
    X = range(size_of_vec)
    Y = range(size_of_vec)
    Z = [X[i] + Y[i] for i in range(len(X))]
    return time.time() - t1

def numpy_version():
    t1 = time.time()
    X = np.arange(size_of_vec)
    Y = np.arange(size_of_vec)
    Z = X + Y
    return time.time() - t1


t1 = pure_python_version()
t2 = numpy_version()
print("python: " + str(t1), "numpy: "+ str(t2))
print("Numpy is in this example " + str(t1/t2) + " times faster!")
```

    python: 0.00027108192443847656 numpy: 4.100799560546875e-05
    Numpy is in this example 6.6104651162790695 times faster!


___

# Making Your Own Module: You're not limited to PyPI

Make your own modules
![pipmod](https://raw.githubusercontent.com/jirvingphd/dsc-lp-libraries-numpy/master/img/import_modules.png)

![pippack](https://raw.githubusercontent.com/jirvingphd/dsc-lp-libraries-numpy/master/img/package_redo.png)

# ⏰ ACTIVITY: Making our own module

- The remainder of this notebook is a Pair Programming Activity (time=15-30 min(?))

    - We will break into groups of 2-3 students. 
    - One student will share their screen while the group works together to solve the tasks. 

    - Clone the activity repository to your computer (or [visit GitHub and fork it](https://github.com/jirvingphd/dtsc-pt-041320-sect-04-python-libraries-study-group) first so you can save your work.)
- `git clone https://github.com/jirvingphd/dtsc-pt-041320-sect-04-python-libraries-study-group.git`



![modlife](https://media1.giphy.com/media/dW0KhIROCaAdCO0V3S/giphy.gif?cid=790b76115d36096678416c65519d8082&rid=giphy.gif)


```python
# this option will re-import your module each time you save an update to it
%load_ext autoreload
%autoreload 2
```


```python
import temperizer as tp
help(tp)
tp
```

### Example: Convert F to C

1. This function is already implemented in `temperizer.py`.
2. Notice that we can call the imported function and see the result.


```python
# 32F should equal 0C
tp.convert_f_to_c(32)
```


```python
# -40F should equal -40C
tp.convert_f_to_c(-40)
```


```python
# 212F should equal 100C
tp.convert_f_to_c(212)
```

### Your turn: Convert C to F

> 1. Find the stub function in `temperizer.py`
2. The word `pass` means "this space intentionally left blank."
3. Add your code _in place of_ the `pass` keyword, _below_ the docstring.
4. Run these cells and make sure that your code works.

#### Small note:

Docstrings (those lines with `""" """` on either side) allow us to create self-documented code. 

Now later, if you forget what each function does, you can use the `?` or `help` functions the same way you would with other functions, and your documentation will show up!


```python
# 0C should equal 32F
tp.convert_c_to_f(0)
```


```python
# -40C should equal -40F
tp.convert_c_to_f(-40)
```


```python
# 100C should equal 212F
tp.convert_c_to_f(100)
```

## Next: Adding New Functions

You need to add support for Kelvin to the `temperizer` library.

1. Create new _stub functions_ in `temperizer.py`:

    * `convert_c_to_k`
    * `convert_f_to_k`
    * `convert_k_to_c`
    * `convert_k_to_f`

    Start each function with a docstring and the `pass` keyword, e.g.:

    ```python
    def convert_f_to_k(temperature_f):
        """Convert Fahrenheit to Kelvin."""
        pass
    ```

2. Then, go back to `temperizer.py` to replace `pass` with your code.


3. Use the cells below to test and validate these functions, similar to the ones above. 
    - Use the temperatures from line 2 of each cell to confirm your function works.
    
    
4. Run the notebook cells to make sure that your new functions work.


```python
# convert_c_to_k 
# 32 C -> 305.15 K

```


```python
# convert_f_to_k
# 60 F -> 288.71 K

```


```python
# convert_k_to_c
# 100 K -> -173.15 C

```


```python
# convert_k_to_f
# 100 K -> -279.67

```

### Extra credit:

make a function in your temperizer that will take a temp in F, and print out:

```
The temperature [number] F is:
    - x in C
    - y in k
```


```python
tp.convert_f_to_all(89)
```

## Congrats!!

You've now made your own module of temperature conversion functions!

#### _Our goals today were to be able to_: <br/>

- Identify and import Python modules and packages (libraries)
- Identify differences between NumPy and base Python in usage and operation
- Create a new module of our own

# Appendix


```python

```

### Numpy array and matrix creation functions

Take 5 minutes and explore each of the following functions.  What does each one do?  What is the syntax of each?
- `np.zeros()`
- `np.ones()`
- `np.full()`
- `np.eye()`
- `np.random.random()`


```python
np.zeros(5)
```


```python
np.ones(5)
```


```python
np.full((3,3),3.3)

```


```python
np.eye(6)
```


```python
np.random.random(6)
```
