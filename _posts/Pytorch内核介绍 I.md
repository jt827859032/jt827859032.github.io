---
title: Pytorch内核介绍 I
date: 2018-04-12 19:55:23
categories:
- Pytorch
tags:
- Pytorch
- Deep Learning
---

# Pytorch内核介绍 I

转载请注明出处，原文：http://pytorch.org/2017/05/11/Internals.html      	

Pytorch的基本单元是Tensor。本文将说明我们怎样在Pytorch里面实现Tensor的以便用户可以通过Python和Tensor进行交互。本文中，我们主要讲述一下四个问题：

- PyTorch如何扩展Python解释器来定义可以从Python代码操纵的Tensor类型？
- PyTorch如何包装那些实际定义张量属性和方法的C库？
- PyTorch的C库包装如何为Tensor方法生成代码？
- PyTorch的构建系统如何将所有这些组件编译并生成可用的应用程序？

## 扩展Python解释器
Pytorch定义了一个名为`torch`的package。本文中我们将会讲到`._C`模块。这个模块被称为是一个“扩展模块”（一个用C实现的Python模块）。这些模块允许我们定义新的内建对象类型（例如：`Tensor`）并调用 C/C++ 函数。

`._C`模块在`torch/csrc/Module.cpp`中定义。其中，`init_C()`和`PyInit__C`函数负责创建模块并根据需要添加方法定义。这个模块被传递给许多不同的`__init()`函数，这些函数将更多的对象添加到模块，注册新类型等等。

这些`__init()`函数调用的一个集合如下所示：
```
ASSERT_TRUE(THPDoubleTensor_init(module));
ASSERT_TRUE(THPFloatTensor_init(module));
ASSERT_TRUE(THPHalfTensor_init(module));
ASSERT_TRUE(THPLongTensor_init(module));
ASSERT_TRUE(THPIntTensor_init(module));
ASSERT_TRUE(THPShortTensor_init(module));
ASSERT_TRUE(THPCharTensor_init(module));
ASSERT_TRUE(THPByteTensor_init(module));
```

这些`__init()`函数将每种类型的张量对象添加到`._C`模块，以便它们可以在模块中使用。下面我们来了解这些方法是如何工作的。

## THPTensor类型
就想底层的`TH`和`THC`库一样，Pytorch定义了一个“通用”的`Tensor`,然后专门用于许多不同的类型。在讲解这种专用机制之前，我们首先考虑如何在Python中定义新类型，以及如何创建通用的`THPTensor`类型。
Python运行时将所有Python对象当作`PyObject *`类型的变量，这种变量被用作所有Python对象的基本类型。每个Python类型都包含该对象的引用计数（refcount）以及指向该对象的类型行对象的指针。类型对象决定了类型的属性。例如，它可能包含一个与类型关联的方法列表，以及调用那些C函数来实现这些方法。该对象还包含表示其状态所需的任何字段（fields）。

定义新类型的步骤如下：

- 创建一个定义了新对象将包含什么东西的结构
- 定义类型的类型对象

结构本身可能很简单。在Python中，所有的浮点类型实际上都是堆中的对象。Python中的浮点对象结构被定义为：
```
typedef struct {
    PyObject_HEAD
    double ob_fval;
} PyFloatObject;
```
其中，`PyObject_HEAD`是一个宏，它引入了实现对象引用计数的代码，以及一个指向相应类型对象的指针。因此在这种情况下，要实现一个浮点数，唯一需要的其他“状态”就是浮点数值本身。

现在让我们看看`THPTensor`的结构：
```
struct THPTensor {
    PyObject_HEAD
    THTensor *cdata;
};
```
十分简单，对么？我们只是通过存储指向它的指针来包装底层的`TH Tensor`。

关键部分是为新类型定义“类型对象”。我们的Python浮点类型对象的示例定义采用以下形式：
```
static PyTypeObject py_FloatType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    "py.FloatObject",          /* tp_name */
    sizeof(PyFloatObject),     /* tp_basicsize */
    0,                         /* tp_itemsize */
    0,                         /* tp_dealloc */
    0,                         /* tp_print */
    0,                         /* tp_getattr */
    0,                         /* tp_setattr */
    0,                         /* tp_as_async */
    0,                         /* tp_repr */
    0,                         /* tp_as_number */
    0,                         /* tp_as_sequence */
    0,                         /* tp_as_mapping */
    0,                         /* tp_hash  */
    0,                         /* tp_call */
    0,                         /* tp_str */
    0,                         /* tp_getattro */
    0,                         /* tp_setattro */
    0,                         /* tp_as_buffer */
    Py_TPFLAGS_DEFAULT,        /* tp_flags */
    "A floating point number", /* tp_doc */
};
```
想象一个类型对象最简单的方法就是定义对象属性的一组字段（fields）。例如，`tp_basicsize`字段被设置为`sizeof(PyFloatObjec)`。这是为了让Python在为`PyFloatObject`调用`PyObject_New()`时知道分配多少内存。你可以设置的字段的完整列表在CPython中的[`object.h`](https://github.com/python/cpython/blob/master/Include/object.h)中定义。 

我们`THPTensor`的类型对象是`THPTensorType`，在`csrc/generic/Tensor.cpp`中定义。这个对象定义了`THPTensor`的名称、大小、映射方法等。

举个例子，我们来看看我们在`PyTypeObject`中设置的`tp_new`函数：

```
PyTypeObject THPTensorType = {
  PyVarObject_HEAD_INIT(NULL, 0)
  ...
  THPTensor_(pynew), /* tp_new */
};
```

`tp_new`函数可以创建对象。它负责创建（而不是初始化）该类型的对象，相当于Python里的`__new__()`方法。C中实现的是一个静态方法，它传递被实例化的类型和任何参数，并返回一个新创建的对象。

```
static PyObject * THPTensor_(pynew)(PyTypeObject *type, PyObject *args, PyObject *kwargs)
{
  HANDLE_TH_ERRORS
  Py_ssize_t num_args = args ? PyTuple_Size(args) : 0;

  THPTensorPtr self = (THPTensor *)type->tp_alloc(type, 0);
// more code below
```

我们函数做的第一件事是分配`THPTensor`，然后它根据传递给函数的参数进行一系列初始化。例如，当我们从另一个`THPTensor` $y$ 来创建一个新的`THPTensor` $x$ 时，我们将新创建的`THPTensor`的`cdata`字段设置为以 $y$ 的基础`TH Tensor`作为参数调用`THTensor_(newWithTensor)`的结果。类似的构造器在 `sizes`, `storages`, `Numpy arrays`和`sequences`里也存在。

** 请注意，我们只用了`tp_new`，而不是`tp_nwe`和`tp_init`（对应于`__init__()`函数）的组合。

`Tensor.cpp`中定义的另一件重要事情是索引如何工作的。Pytorch的`Tensors`支持Python的`Mapping Protocol`。这允许我们如下类似的事：

```
x = torch.Tensor(10).fill_(1)
y = x[3] // y == 1
x[4] = 2
// etc.
```

** 请注意，这种索引可以扩展到超过一维的`Tensors`。

我们能够通过[这里](https://docs.python.org/3.7/c-api/typeobj.html#c.PyMappingMethods)描述的定义三种映射方法来使用 `[]`风格的符号。

最重要的方法是`THPTensor_(getValue)`和`THPTensor_(setValue)`，它们描述了如何为`Tensor`索引，返回新的`Tensor/Scalar`，或者更新现有`Tensor`的值。通读这些实现可以更好地理解PyTorch如何支持基本的`Tensor`索引。

## 通用构建（Part 1）

我们可以花大量的时间来探索`THPTensor`的各个方面，以及它与定义一个新的Python对象的关系。但是我们仍需要看看`THPTensor_(init)()`函数如何转换为我们再模块初始化中使用的`THPIntTensor_int()`。我们如何使用定义了”通用“ `Tensor`的`Tensor.cpp`文件并使用它来为所有类型的排列（permutations）生成Python对象？换句话说，`Tensor.cpp`中散布着一些代码，比如：

```
return THPTensor_(New)(THTensor_(new)(LIBRARY_STATE_NOARGS));
```

这说明了我们需要生成特定类型的两种情况：

- 我们的输出代码将会调用`THP<Type>Tensor_New(...)`替代`THPTensor_(New)`。
- 我们的输出代码将会调用`TH<Type>Tensor_new(...)`替代`THPTensor_(New)`。

换句话说，对所有支持的`Tensor`类型，我们需要“生成”完成上述替换的源代码。这是Pytorch的“构建”过程的一部分。Pytorch依赖于[Setuptools](https://setuptools.readthedocs.io/en/latest/)构建package，而且我们再根目录中定义一个`setup.py`文件来定制构建过程。

使用Setuptools构建扩展模块的一个组件是列出编译中涉及的源文件。然而，我们的`csrc/generic/Tensor.cpp`文件并没有列出来。那么这个文件中的代码如何成为最终产品的一部分呢？

回想一下，我们从`generic`上面的目录调用`THPTensor *`函数（比如`init`）。如果我们看一下这个目录，就会定义另一个文件`Tensor.cpp`，这个文件的最后一行很重要：

```
//generic_include TH torch/csrc/generic/Tensor.cpp
```

请注意，这个`Tensor.cpp`文件包含在`setup.py`中，但它被包装在一个名为`split_types`的Python帮助函数中。该函数将输入文件作为输入，并在文件内容中查找、`\\generic_include`字符串。如果找到，它会为每个`Tensor`类型生成一个新的输出文件，并进行一下更改：

- 输出文件被重命名为`Tensor<Type>.cpp`

- 输出文件稍作如下修改：

  ```
  # Before:
  //generic_include TH torch/csrc/generic/Tensor.cpp

  # After:
  #define TH_GENERIC_FILE "torch/src/generic/Tensor.cpp"
  #include "TH/THGenerate<Type>Type.h"
  ```

  在第二行包含头文件的作用是在`Tensor.cpp`中包含源代码并定义了一些额外的上下文。让我们看看其中的一个头文件：

  ```
  #ifndef TH_GENERIC_FILE
  #error "You must define TH_GENERIC_FILE before including THGenerateFloatType.h"
  #endif

  #define real float
  #define accreal double
  #define TH_CONVERT_REAL_TO_ACCREAL(_val) (accreal)(_val)
  #define TH_CONVERT_ACCREAL_TO_REAL(_val) (real)(_val)
  #define Real Float
  #define THInf FLT_MAX
  #define TH_REAL_IS_FLOAT
  #line 1 TH_GENERIC_FILE
  #include TH_GENERIC_FILE
  #undef accreal
  #undef real
  #undef Real
  #undef THInf
  #undef TH_REAL_IS_FLOAT
  #undef TH_CONVERT_REAL_TO_ACCREAL
  #undef TH_CONVERT_ACCREAL_TO_REAL

  #ifndef THGenerateManyTypes
  #undef TH_GENERIC_FILE
  #endif
  ```

  它做的是从通用`Tensor.cpp`文件中引入代码并用下面的宏定义包围它。例如，我们将`real`定义为一个浮点数，因此任何在通用`Tensor`的实现代码都将定义为`real`的东西都会替换为浮点数。在相应的文件`THGenerateIntType.h`中，同一个宏将用`int`替换为`real`。

  这些输出文件从`split_types`返回并添加到源文件列表中，因此我们可以看出如何创建不同类型的`.cpp`代码。

  这里有几点需要注意：首先，`split_types`函数不是绝对必要的。我们可以将`Tensor.cpp`中的代码封装在单个文件中，并为每种类型重复该代码。我们将代码拆分为单独的文件的原因是为了加快编译速度。其次，我们讨论类型替换（例如用`float`替换`real`）是指在C预处理器在编译期间执行这些替换。仅仅使用这些宏包围源代码直到预处理才会产生效果。

  ## 通用构建（Part 2）

  现在我们有所有`Tensor`类型的源文件，我们需要考虑如何创建相应的头部声明，以及如何从`THTensor_(method)`和`THPTensor_(method)`转换为`TH<Type>Tensor_method`和`THP<Type>Tensor_method`的。例如，`csrc/reneric/Tensor.h`有以下声明：

  ```
  THP_API PyObject * THPTensor_(New)(THTensor *ptr);
  ```

  我们使用相同的策略在头文件的源文件中生成代码。在`csrc/Tensor.h`，我们做了以下事情：

  ```
  #include "generic/Tensor.h"
  #include <TH/THGenerateAllTypes.h>

  #include "generic/Tensor.h"
  #include <TH/THGenerateHalfType.h>
  ```

  这也有同样的效果，我们从通用头文件中抽取代码，用相同的宏定义包装每种类型的代码。唯一的区别是生成的代码全部包含在同一个头文件中，而不是被拆分成多个源文件。

  最后，我们需要考虑如何“转换”或“替代”函数类型。如果我们查看相同的头文件，我们会看到一堆`#define`语句，其中包括：

  ```
  #define THPTensor_(NAME)            TH_CONCAT_4(THP,Real,Tensor_,NAME)
  ```

  这个宏表示源码中与`THPTensor_(NAME)`格式匹配的任何字符串都应该替换为`THPRealTensor_Name`，其中Real是从当前`#define`定义出来的。因为我们的头文件代码和源码被所有类型的宏定义所包围，所以在预处理器运行后，得到的代码就是我们所期望的。`TH`库中的代码为`THTensor_(NAME)`定义了相同的宏，并支持这些函数的翻译（translation）。通过这种方式，我们最终得到了带有专用代码的头文件和源文件。

  ### 模块对象和类型方法

  现在我们已经看到我们如何在`THP`中包含`TH`的`Tensor`定义，并生成`THP`方法，如`THPFloatTensor_init(...)`。现在我们可以根据我们创建的模块探索上面的代码实际上做了什么。`THPTensor_(init)`中的关键行是：

  ```
  # THPTensorBaseStr, THPTensorType are also macros that are specific 
  # to each type
  PyModule_AddObject(module, THPTensorBaseStr, (PyObject *)&THPTensorType);
  ```

  该函数将我们的`Tensor`对象注册到扩展模块，所以我们可以在我们的Python代码中使用`THPFloatTensor`，`THPIntTensor`等。

  仅仅能够创建`Tensor`其实并不是很有用，我们需要能够调用`TH`定义的所有方法。一个简单的例子展示了调用`Tensor`上的内部替换的`zero_`方法。

  ```
  x = torch.FloatTensor(10)
  x.zero_()
  ```

  我们先看看如何将方法添加到新定义的类型。“类型对象”中的一个字段是`tp_methods`。该字段包含一组方法定义（`PyMethodDef`s），用于将方法（及其C/C++实现）与类型关联。假设我们想在我们的`PyFloatObject`上定义一个替换值的新方法。我们可以如下实现：

  ```
  static PyObject * replace(PyFloatObject *self, PyObject *args) {
  	double val;
  	if (!PyArg_ParseTuple(args, "d", &val))
  		return NULL;
  	self->ob_fval = val;
  	Py_RETURN_NONE
  }
  ```

  这个等价于Python方法：

  ```
  def replace(self, val):
  	self.ob_fval = val
  ```

  阅读关于如何在CPython中定义方法的更多信息很有帮助。通常，方法将第一个参数作为对象的实例，并且可选地包含位置参数和关键字参数的参数。这个静态函数在我们的`float`上注册为一个方法：

  ```
  static PyMethodDef float_methods[] = {
  	{"replace", (PyCFunction)replace, METH_VARARGS,
  	"replace the value in the float"
  	},
  	{NULL} /* Sentinel */
  }
  ```

  这注册了一个名为`replace`的方法，它由具有相同名称的C函数实现。`METH_VARARGS`标志指示该方法采用表示函数的所有参数的参数元组。该数组设置为类型对象的`tp_methods`字段，然后我们可以在该类型的对象上使用`replace`方法。

  我们希望能够在`THP Tensor`上等价地调用`TH Tensor`的所有方法。但是，编写所有`TH`方法的包装将会非常耗时且容易出错。我们需要一个更好的方式来做到这一点。

  ​

  ## PyTorch cwrap

  PyTorch实现了自己的cwrap工具来包装用于Python后端的TH张量方法。我们定义一个`.cwrap`文件，其中包含一系列C方法声明，并以我们自定义的`YAML`格式。cwrap工具读取这个文件并输出`.cpp`源文件，这个源文件包含与我们的`THPTensor` Python对象和Python C 扩展方法调用格式相兼容的包装方法。这个工具用于生成代码来包装不仅仅是`TH`，而且包含`CuDNN`。它被定义为可扩展的。

  一个内部替换的`addmv_`方法的`YAML`声明样例如下：

  ```
  [[
    name: addmv_
    cname: addmv
    return: self
    arguments:
      - THTensor* self
      - arg: real beta
        default: AS_REAL(1)
      - THTensor* self
      - arg: real alpha
        default: AS_REAL(1)
      - THTensor* mat
      - THTensor* vec
  ]]
  ```

  cwrap工具的架构非常简单。它读入一个文件，然后用一些列插件对其进行处理。有关插件可以更改代码的所有方法，请参阅`tools/cwrap/plugins/__init__.py`。

  源码生成发生在一系列的过程中。首先，解析和处理`YAML`声明。然后源码逐个生成：添加诸如参数检查和提取之类的东西，定义方法头文件以及实际调用底层库（如`TH`）。最后，cwrap工具允许一次处理整个文件。`addmv_`的结果输出可以在[这里](https://gist.github.com/killeent/c00de46c2a896335a52552604cc4d74b)查看。

  为了与CPython后端交互，该工具生成一个`PyMehtodDef`s数组，该数组可以存储或附加到`THPTensor`的`tp_methods`字段。在包装`Tensor`方法的特定情况下，构建过程首先从`TensorMethods.cwrap`生成输出源文件。该源文件`#include`在通用`Tensor`源文件中。这一切都放生在预处理器起作用的时候。结果，生成的所有方法包装都与上面的`THPTensor`代码进行相同的传递。因此，单个通用声明和定义也是针对每种类型专门设计的。

  ## Putting It All Together

  到目前为止，我们已经展示了如何扩展Python解释器来创建新的扩展模块，这样的模块如何定义我们的新`THPTensor`类型，以及如何为与`TH`接口的所有类型的`Tensor`生成源码。简而言之，我们将讲讲编译。

  Setuptools允许我们定义一个用于编译的扩展。整个`torch._C`扩展通过手机所有源文件，头文件、库等来编译，并创建一个`setuptools extension`。然后setuptools处理构建扩展本身，我们将在后续的文章中探讨构建过程。

  总而言之，让我们重温我们的四个问题：

  - PyTorch如何扩展Python解释器来定义可以从Python代码操纵的Tensor类型？

    它使用CPython的框架来扩展Python解释器并定义新的类型，同时特别注重为所有类型生成代码。

  - PyTorch如何包装那些实际定义张量属性和方法的C库？

    它通过定义一个新的类型`THPTensor`来实现，`THPTensor`支持TH张量。函数调用通过CPython后端的约定转发给此向量。

  - PyTorch的C库包装如何为Tensor方法生成代码？

    它采用我们自定义的`YAML`格式的代码，并通过使用多个插件通过一系列步骤处理它来为每种方法生成源代码。

  - PyTorch的构建系统如何将所有这些组件编译并生成可用的应用程序？

    它需要一些源/头文件，库和编译指令来使用Setuptools构建扩展。

  ​

  ---

  这只是Pytorch构建系统部分的快照。有更多的细微差别和细节，但我希望这可以作为我们`Tensor library`和许多组件的一种简单介绍。

  ​

  ### Resources

  - <https://docs.python.org/3.7/extending/index.html> is invaluable for understanding how to write C/C++ Extension to Python