# 30min速成Python指南

[toc]

## 一、为什么要学习Python

原因很简单：简单，好上手。对于一个初学者来说，大家需要一门非常容易上手实现代码的语言来入门深度学习，Python是一种解释性语言，即使大家没有丰富的计算机知识，也可以快速学习。整体的代码非常清晰，易于阅读，并且功能强大，可以应用到多种场景。目前深度学习很多代码，尤其是发的Paper里的代码大多都有python代码实现，非常方便大家复现和实验。

## 二、如何安装Python

### 1.下载并安装Python
打开[Python官网](https://www.python.org)，进入[下载页面](https://www.python.org/downloads/)：

![image-20210422010607373](https://tva1.sinaimg.cn/large/008i3skNgy1gprxgc8miuj31tq0u0drp.jpg)

下载时，可以选择具体的版本号，这里我们以Python 3.7.5为例![image-20210422010836481](https://tva1.sinaimg.cn/large/008i3skNgy1gprxgkh3vkj31t60p247s.jpg)

点进去根据你的系统（Windows/Mac OS X/Ubuntu等），选择相应的版本包下载安装即可

![image-20210422011032662](https://tva1.sinaimg.cn/large/008i3skNgy1gprxgp5kurj31t20siwpd.jpg)

下载完成后，在终端输入python，输入'Hello,TinyMS'：

![image-20210422011827582](https://tva1.sinaimg.cn/large/008i3skNgy1gprxgugew4j32260h041l.jpg)

### 2.选择合适的Pip源？

<details class="note" open="open" style="box-sizing: inherit; box-shadow: rgba(0, 0, 0, 0.14) 0px 2px 2px 0px, rgba(0, 0, 0, 0.12) 0px 1px 5px 0px, rgba(0, 0, 0, 0.2) 0px 3px 1px -2px; position: relative; margin: 1.5625em 0px; padding: 0px 0.6rem; border-left-width: 0.2rem; border-left-style: solid; border-left-color: rgb(68, 138, 255); border-top-left-radius: 0.1rem; border-top-right-radius: 0.1rem; border-bottom-right-radius: 0.1rem; border-bottom-left-radius: 0.1rem; font-size: 0.64rem; overflow: auto; display: block; caret-color: rgba(0, 0, 0, 0.87); color: rgba(0, 0, 0, 0.87); font-family: &quot;Fira Sans&quot;, &quot;Helvetica Neue&quot;, Helvetica, Arial, sans-serif; font-style: normal; font-variant-caps: normal; font-weight: normal; letter-spacing: normal; orphans: auto; text-align: start; text-indent: 0px; text-transform: none; white-space: normal; widows: auto; word-spacing: 0px; -webkit-text-size-adjust: auto; -webkit-text-stroke-width: 0px; text-decoration: none;"><span style="display: contents;"><summary style="box-sizing: inherit; display: block; outline: none; cursor: pointer; margin: 0px -0.6rem; padding: 0.4rem 2rem; border-bottom-width: 0.05rem; border-bottom-style: solid; border-bottom-color: rgba(68, 138, 255, 0.1); background-color: rgba(68, 138, 255, 0.1); font-weight: 700;"><span style="display: contents;"><a href="https://pypi.org/project/pip/" style="box-sizing: inherit; text-decoration-skip: objects; color: rgb(63, 81, 181); text-decoration: none; word-break: break-word; transition: color 0.125s;"><strong style="box-sizing: inherit;">pip</strong></a><span class="Apple-converted-space">&nbsp;</span>是什么？</span></summary><p style="box-sizing: inherit; margin: 1em 0px 0.6rem;">Python 的默认包管理器，用来安装第三方 Python 库。它的功能很强大，能够处理版本依赖关系，还能通过 wheel 文件支持二进制安装。pip 的库现在托管在<span class="Apple-converted-space">&nbsp;</span><a href="https://pypi.org/" style="box-sizing: inherit; text-decoration-skip: objects; color: rgb(63, 81, 181); text-decoration: none; word-break: break-word; transition: color 0.125s;">PyPI</a>（即“Python 包索引”）平台上，用户也可以指定第三方的包托管平台。</p></span></details>

关于 PyPI 的镜像，可以使用如下大镜像站的资源：

- [华为开源镜像站](https://mirrors.huaweicloud.com/)
- [清华大学 TUNA 镜像站](https://mirrors.tuna.tsinghua.edu.cn/help/pypi/)
- [中国科学技术大学镜像站](http://mirrors.ustc.edu.cn/help/pypi.html)
- [豆瓣的 PyPI 源](https://pypi.douban.com/simple)

## 三、Python基础语法

### 1. 原始数据类型与运算符

Python有五个标准的数据类型：

- Numbers（数字）
- String（字符串）
- List（列表）
- Tuple（元祖）
- Dictionary（字典）

也支持多种运算符操作，按照优先级排列如下：

| 运算符                   | 描述                                                   |
| :----------------------- | :----------------------------------------------------- |
| **                       | 指数 (最高优先级)                                      |
| ~ + -                    | 按位翻转, 一元加号和减号 (最后两个的方法名为 +@ 和 -@) |
| * / % //                 | 乘，除，取模和取整除                                   |
| + -                      | 加法减法                                               |
| >> <<                    | 右移，左移运算符                                       |
| &                        | 位 'AND'                                               |
| ^ \|                     | 位运算符                                               |
| <= < > >=                | 比较运算符                                             |
| <> == !=                 | 等于运算符                                             |
| = %= /= //= -= += *= **= | 赋值运算符                                             |
| is is not                | 身份运算符                                             |
| in not in                | 成员运算符                                             |
| not and or               | 逻辑运算符                                             |

```python
# 整数
3  # => 3

# 基础算数
1 + 1  # => 2
8 - 1  # => 7
10 * 2  # => 20

# 但是除法例外，会自动转换成浮点数
35 / 5  # => 7.0
5 / 3  # => 1.6666666666666667

# 整数除法的结果都是向下取整
5 // 3     # => 1
5.0 // 3.0 # => 1.0 # 浮点数也可以
-5 // 3  # => -2
-5.0 // 3.0 # => -2.0

# 浮点数的运算结果也是浮点数
3 * 2.0 # => 6.0

# 模除
7 % 3 # => 1

# x的y次方
2**4 # => 16

# 用括号决定优先级
(1 + 3) * 2  # => 8

# 布尔值
True
False

# 用not取非
not True  # => False
not False  # => True

# 逻辑运算符，注意and和or都是小写
True and False # => False
False or True # => True

# 整数也可以当作布尔值
0 and 2 # => 0
-5 or 0 # => -5
0 == False # => True
2 == True # => False
1 == True # => True

# 用==判断相等
1 == 1  # => True
2 == 1  # => False

# 用!=判断不等
1 != 1  # => False
2 != 1  # => True

# 比较大小
1 < 10  # => True
1 > 10  # => False
2 <= 2  # => True
2 >= 2  # => True

# 大小比较可以连起来！
1 < 2 < 3  # => True
2 < 3 < 2  # => False

# 字符串用单引双引都可以
"这是个字符串"
'这也是个字符串'

# 用加号连接字符串
"Hello " + "world!"  # => "Hello world!"

# 字符串可以被当作字符列表
"This is a string"[0]  # => 'T'

# 用.format来格式化字符串
"{} can be {}".format("strings", "interpolated")

# 可以重复参数以节省时间
"{0} be nimble, {0} be quick, {0} jump over the {1}".format("Jack", "candle stick")
# => "Jack be nimble, Jack be quick, Jack jump over the candle stick"

# 如果不想数参数，可以用关键字
"{name} wants to eat {food}".format(name="Bob", food="lasagna") 
# => "Bob wants to eat lasagna"

# 如果你的Python3程序也要在Python2.5以下环境运行，也可以用老式的格式化语法
"%s can be %s the %s way" % ("strings", "interpolated", "old")

# None是一个对象
None  # => None

# 当与None进行比较时不要用 ==，要用is。is是用来比较两个变量是否指向同一个对象。
"etc" is None  # => False
None is None  # => True

# None，0，空字符串，空列表，空字典都算是False
# 所有其他值都是True
bool(0)  # => False
bool("")  # => False
bool([]) # => False
bool({}) # => False
```



注：常见的Python规范：

1. 注释：单行注释以#开头，多行注释可以用多个 **#** 号，还有 **'''** 和 **" " "**，也可以使用 **''' '''** 的格式在三引号之间书写较长的注释，或者在函数的首部对函数进行一个说明：

   ```python
   def test(test_string):
       '''形参为任意类型的对象，
          这个示例函数会将其原样返回。
       '''
       return test_string
   ```

2. 缩进：Python使用缩进来表示代码块，同一个代码块的语句必须包含相同的缩进空格数，否则会报错：

   ```python
   if True:
       print ("Answer")
       print ("True")
   else:
       print ("Answer")
     print ("False")    # 缩进不一致，会导致运行错误
   ```

3. import模块
   ```python
      1. 将整个模块导入，例如：import time，在引用时格式为：time.sleep(1)。
      2. 将整个模块中全部函数导入，例如：from time import *，在引用时格式为：sleep(1)。
      3. 将模块中特定函数导入，例如：from time import sleep，在引用时格式为：sleep(1)。
      4. 将模块换个别名，例如：import time as abc，在引用时格式为：abc.sleep(1)。
   ```

### 2.变量和集合

Python 中的变量赋值不需要类型声明，但在使用前都必须赋值，赋值以后该变量才会被创建。

等号 **=** 用来给变量赋值，左边是一个变量名，右边是存储在变量中的值。

```python
# print是内置的打印函数
print("Hello，Welcome to Learn Python in 30 Minutes")

# 在给变量赋值前不用提前声明
# 传统的变量命名是小写，用下划线分隔单词
some_var = 5
some_var  # => 5

# 访问未赋值的变量会抛出异常
some_unknown_var  # 抛出NameError


####################################################
#####################  list  #######################
####################################################

# 用列表(list)储存序列
a = []
# 创建列表时也可以同时赋给元素
b = [4, 5, 6]

# 用append在列表最后追加元素
a.append(1)    # li现在是[1]
a.append(2)    # li现在是[1, 2]
a.append(4)    # li现在是[1, 2, 4]
a.append(3)    # li现在是[1, 2, 4, 3]
# 用pop从列表尾部删除
a.pop()        # => 3 且li现在是[1, 2, 4]
# 把3再放回去
a.append(3)    # li变回[1, 2, 4, 3]

# 列表存取跟数组一样
a[0]  # => 1
# 取出最后一个元素
a[-1]  # => 3

# 越界存取会造成IndexError
a[4]  # 抛出IndexError

# 列表有切割语法
a[1:3]  # => [2, 4]
# 取尾
a[2:]  # => [4, 3]
# 取头
a[:3]  # => [1, 2, 4]
# 隔一个取一个
a[::2]   # =>[1, 4]
# 倒排列表
a[::-1]   # => [3, 4, 2, 1]
# 可以用三个参数的任何组合来构建切割
# li[始:终:步伐]

# 用del删除任何一个元素
del a[2]   # li is now [1, 2, 3]

# 列表可以相加
# 注意：a 和 b 的值都不变
a + b   # => [1, 2, 3, 4, 5, 6]

# 用extend拼接列表
a.extend(b)   # li现在是[1, 2, 3, 4, 5, 6]

# 用in测试列表是否包含值
1 in a   # => True

# 用len取列表长度
len(a)   # => 6


####################################################
#####################  tuple #######################
####################################################

# 元组是不可改变的序列
tup = (1, 2, 3)
tup[0]   # => 1
tup[0] = 3  # 抛出TypeError

# 列表允许的操作元组大都可以
len(tup)   # => 3
tup + (4, 5, 6)   # => (1, 2, 3, 4, 5, 6)
tup[:2]   # => (1, 2)
2 in tup   # => True

# 可以把元组合列表解包，赋值给变量
a, b, c = (1, 2, 3)     # 现在a是1，b是2，c是3
# 元组周围的括号是可以省略的
d, e, f = 4, 5, 6
# 交换两个变量的值就这么简单
e, d = d, e     # 现在d是5，e是4


####################################################
#####################  dict  #######################
####################################################

# 用字典表达映射关系
empty_dict = {}
# 初始化的字典
filled_dict = {"one": 1, "two": 2, "three": 3}

# 用[]取值
filled_dict["one"]   # => 1

# 用 keys 获得所有的键。
# 因为 keys 返回一个可迭代对象，所以在这里把结果包在 list 里。我们下面会详细介绍可迭代。
# 注意：字典键的顺序是不定的，你得到的结果可能和以下不同。
list(filled_dict.keys())   # => ["three", "two", "one"]

# 用values获得所有的值。跟keys一样，要用list包起来，顺序也可能不同。
list(filled_dict.values())   # => [3, 2, 1]


# 用in测试一个字典是否包含一个键
"one" in filled_dict   # => True
1 in filled_dict   # => False

# 访问不存在的键会导致KeyError
filled_dict["four"]   # KeyError

# 用get来避免KeyError
filled_dict.get("one")   # => 1
filled_dict.get("four")   # => None
# 当键不存在的时候get方法可以返回默认值
filled_dict.get("one", 4)   # => 1
filled_dict.get("four", 4)   # => 4

# setdefault方法只有当键不存在的时候插入新值
filled_dict.setdefault("five", 5)  # filled_dict["five"]设为5
filled_dict.setdefault("five", 6)  # filled_dict["five"]还是5

# 字典赋值
filled_dict.update({"four":4}) # => {"one": 1, "two": 2, "three": 3, "four": 4}
filled_dict["four"] = 4  # 另一种赋值方法

# 用del删除
del filled_dict["one"]  # 从filled_dict中把one删除


####################################################
#####################  set  ########################
####################################################

# 用set表达集合
empty_set = set()
# 初始化一个集合，语法跟字典相似。
some_set = {1, 1, 2, 2, 3, 4}   # some_set现在是{1, 2, 3, 4}

# 可以把集合赋值于变量
filled_set = some_set

# 为集合添加元素
filled_set.add(5)   # filled_set现在是{1, 2, 3, 4, 5}

# & 取交集
other_set = {3, 4, 5, 6}
filled_set & other_set   # => {3, 4, 5}

# | 取并集
filled_set | other_set   # => {1, 2, 3, 4, 5, 6}

# - 取补集
{1, 2, 3, 4} - {2, 3, 5}   # => {1, 4}

# in 测试集合是否包含元素
2 in filled_set   # => True
10 in filled_set   # => False
```

### 3. 控制流和迭代器

Python条件语句是通过一条或多条语句的执行结果（True或者False）来决定执行的代码块，没有switch/case语句

```python
# 先随便定义一个变量
some_var = 5

####################################################
#####################  if 语句 ######################
####################################################

# 这是个if语句。注意缩进在Python里是有意义的
# 印出"some_var比10小"
if some_var > 10:
    print("some_var比10大")
elif some_var < 10:    # elif句是可选的
    print("some_var比10小")
else:                  # else也是可选的
    print("some_var就是10")

# 如果 if 语句中的条件过长，可以用接续符 \ 来换行。例如：
if 2>1 and 3>2 and 4>3 and \
    5>4 and 6>5 and 7>6 and \
    8>7:
    print("OK")

    
####################################################
#####################  for语句  #####################
####################################################   
    
"""
用for循环语句遍历列表
打印:
    dog is a mammal
    cat is a mammal
    mouse is a mammal
"""
for animal in ["dog", "cat", "mouse"]:
    print("{} is a mammal".format(animal))

"""
"range(number)"返回数字列表从0到给的数字
打印:
    0
    1
    2
    3
"""
for i in range(4):
    print(i)

"""
while循环直到条件不满足
打印:
    0
    1
    2
    3
"""
x = 0
while x < 4:
    print(x)
    x += 1  # x = x + 1 的简写

# 用try/except块处理异常状况
try:
    # 用raise抛出异常
    raise IndexError("This is an index error")
except IndexError as e:
    pass    # pass是无操作，但是应该在这里处理错误
except (TypeError, NameError):
    pass    # 可以同时处理不同类的错误
else:   # else语句是可选的，必须在所有的except之后
    print("All good!")   # 只有当try运行完没有错误的时候这句才会运行


# Python提供一个叫做可迭代(iterable)的基本抽象。一个可迭代对象是可以被当作序列
# 的对象。比如说上面range返回的对象就是可迭代的。

filled_dict = {"one": 1, "two": 2, "three": 3}
our_iterable = filled_dict.keys()
print(our_iterable) # => dict_keys(['one', 'two', 'three'])，是一个实现可迭代接口的对象

# 可迭代对象可以遍历
for i in our_iterable:
    print(i)    # 打印 one, two, three

# 但是不可以随机访问
our_iterable[1]  # 抛出TypeError

# 可迭代对象知道怎么生成迭代器
our_iterator = iter(our_iterable)

# 迭代器是一个可以记住遍历的位置的对象
# 用__next__可以取得下一个元素
our_iterator.__next__()  # => "one"

# 再一次调取__next__时会记得位置
our_iterator.__next__()  # => "two"
our_iterator.__next__()  # => "three"

# 当迭代器所有元素都取出后，会抛出StopIteration
our_iterator.__next__() # 抛出StopIteration

# 可以用list一次取出迭代器所有的元素
list(filled_dict.keys())  # => Returns ["one", "two", "three"]
```
 
### 4. 迭代器与生成器

这个功能非常常见，也是Python最强大的功能之一，尤其在数据读取时，经常会用到迭代器和生成器，可以节省内存空间，显著提升代码运行速度。

#### 迭代器

##### 1.迭代器是什么？

- 迭代器是一个可以记住遍历的位置的对象，迭代器的对象从集合的第一个元素开始访问，直到所有的元素被访问完结束；迭代器只能往前不会后退；字符串，列表或元组对象都可用于创建迭代器。
- 基本方法：**iter()** 和 **next()**
- 创建迭代器对象：

```python
# 字符串
for s in "Hello,TinyMS":
    print (s)
    
# 列表  
>>> list_a=[1,2,3,4]
>>> it = iter(list_a)    # 创建迭代器对象
>>> print (next(it))   # 输出迭代器的下一个元素
1
>>> print (next(it))
2

# 元祖
>>> tuple_a=(1,2,3,4)
>>> it = iter(tutple_a)
>>> print (next(it))   # 输出迭代器的下一个元素
1
>>> print (next(it))
2

# 字典(注意字典是无序的)
d = {
  'apple' : 'tasty',
  'bananas' : 'the best',
  'brussel sprouts' : 'evil',
  'cauliflower' : 'pretty good'
}

for sKey in d:
  print "{0} are {1}".format(sKey,d[sKey])

```

- 遍历迭代器对象：

```python
list=[1,2,3,4]
it = iter(list)    # 创建迭代器对象
for x in it:
    print (x, end=" ")
```

输出结果：

```python
1 2 3 4
```

- 构造一个迭代器

把一个类作为一个迭代器使用需要在类中实现两个方法 __iter__() 与 __next__()

​    1. _iter__() 方法返回一个特殊的迭代器对象， 这个迭代器对象实现了 __next__() 方法并通过 StopIteration 异常标识迭代的完成。

​    2. __next__() 方法（Python 2 里是 next()）会返回下一个迭代器对象。

创建一个返回数字的迭代器，初始值为 1，逐步递增 1：

```python
class MyNumbers:
  def __iter__(self):
    self.a = 1
    return self
 
  def __next__(self):
    x = self.a
    self.a += 1
    return x
 
myclass = MyNumbers()
myiter = iter(myclass)
 
print(next(myiter))
print(next(myiter))
print(next(myiter))
print(next(myiter))
print(next(myiter))
```

输出结果：

```python
1
2
3
4
5
```

- 结束迭代标识：StopIteration

  StopIteration 异常用于标识迭代的完成，防止出现无限循环的情况，在 __next__() 方法中我们可以设置在完成指定循环次数后触发 StopIteration 异常来结束迭代。

  在 10 次迭代后停止执行：

  ```python
  class MyNumbers:
    def __iter__(self):
      self.a = 1
      return self
   
    def __next__(self):
      if self.a <= 20:
        x = self.a
        self.a += 1
        return x
      else:
        raise StopIteration
   
  myclass = MyNumbers()
  myiter = iter(myclass)
   
  for x in myiter:
    print(x)
  ```

  输出结果：

  ```python
  1
  2
  3
  4
  5
  6
  7
  8
  9
  10
  ```

##### 2.TinyMS中的迭代器

TinyMS中的data模块，继承了MindSpore的dataset类中的engine模块的功能，可以看到在engine的代码中我们定义了数据读取的基类迭代器Iterator](https://gitee.com/mindspore/mindspore/blob/master/mindspore/dataset/engine/iterators.py)，再根据不同的数据类型，分别创建了DictIterator、TupleIterator等，举例看一下:

```python
class DictIterator(Iterator):
    """
    The derived class of Iterator with dict type.
    """

    def _get_next(self):
        """
        Returns the next record in the dataset as dictionary

        Returns:
            Dict, the next record in the dataset.
        """
        try:
            return {k: self._transform_tensor(t) for k, t in self._iterator.GetNextAsMap().items()}
        except RuntimeError as err:
            ## maybe "Out of memory" / "MemoryError" error
            err_info = str(err)
            if err_info.find("Out of memory") >= 0 or err_info.find("MemoryError") >= 0:
                logger.error("Memory error occurred, process will exit.")
                os.kill(os.getpid(), signal.SIGKILL)
            raise err
```



#### 生成器

##### 1. 生成器是什么？

在 Python 中，使用了 yield 的函数被称为生成器（generator），生成器是一个返回迭代器的函数，只能用于迭代操作，更简单点理解生成器就是一个迭代器。 

在调用生成器运行的过程中，每次遇到 yield 时函数会暂停并保存当前所有的运行信息，返回 yield 的值, 并在下一次执行 next() 方法时从当前位置继续运行。

调用一个生成器函数，返回的是一个迭代器对象。

实例：使用 yield 实现斐波那契数列

```python
import sys
 
def fibonacci(n): # 生成器函数 - 斐波那契
    a, b, counter = 0, 1, 0
    while True:
        if (counter > n): 
            return
        yield a
        # print a
        a, b = b, a + b
        counter += 1
f = fibonacci(10) # f 是一个迭代器，由生成器返回生成
 
while True:
    try:
        print (next(f), end=" ")
    except StopIteration:
        sys.exit()
```

输出结果：

```python
0 1 1 2 3 5 8 13 21 34 55
```



##### 2. 为什么要使用yield函数？

一个带有 yield 的函数就是一个 generator，它和普通函数不同，生成一个 generator 看起来像函数调用，但不会执行任何函数代码，直到对其调用 next()（在 for 循环中会自动调用 next()）才开始执行。虽然执行流程仍按函数的流程执行，但每执行到一个 yield 语句就会中断，并返回一个迭代值，下次执行时从 yield 的下一个语句继续执行。看起来就好像一个函数在正常执行的过程中被 yield 中断了数次，每次中断都会通过 yield 返回当前的迭代值。

yield 的好处是显而易见的，把一个函数改写为一个 generator 就获得了迭代能力，比起用类的实例保存状态来计算下一个 next() 的值，不仅代码简洁，而且执行流程异常清晰。

```python
>>> f = fibonacci(5)
>>> next(f)
0
>>> next(f)
1
>>> next(f)
1
>>> next(f)
2
>>> next(f)
3
>>> next(f)
5
>>> next(f)
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
StopIteration
```



- 另一个 yield 的例子来源于文件读取。如果直接对文件对象调用 read() 方法，会导致不可预测的内存占用。好的方法是利用固定长度的缓冲区来不断读取文件内容。通过 yield，我们不再需要编写读文件的迭代类，就可以轻松实现文件读取：

  ```python
  def read_file(fpath): 
      BLOCK_SIZE = 1024 
      with open(fpath, 'rb') as f: 
          while True: 
              block = f.read(BLOCK_SIZE) 
              if block: 
                  yield block 
              else: 
                  return
  ```

##### 3. 在TinyMS中如何使用yield函数？

我们先使用TinyMS的data模块中的[GeneratorDataset](https://tinyms.readthedocs.io/zh_CN/latest/tinyms/tinyms.data.html)函数构建了一个随机数据集，然后对其进行混洗操作，最后展示了混洗后的数据结果

```python
import numpy as np
import tinyms as ts
from tinyms.data import GeneratorDataset as gd

gd.config.set_seed(0)

def generator_func():
    for i in range(5):
        yield (np.array([i, i+1, i+2]),)

dataset1 = gd.GeneratorDataset(generator_func, ["data"])

dataset1 = dataset1.shuffle(buffer_size=2)
for data in dataset1.create_dict_iterator():
    print(data)
```

输出结果：

```python
{'data': Tensor(shape=[3], dtype=Int64, value= [3, 4, 5])}
{'data': Tensor(shape=[3], dtype=Int64, value= [2, 3, 4])}
{'data': Tensor(shape=[3], dtype=Int64, value= [4, 5, 6])}
{'data': Tensor(shape=[3], dtype=Int64, value= [1, 2, 3])}
{'data': Tensor(shape=[3], dtype=Int64, value= [0, 1, 2])}
```
### 5. 函数

语法：

```python
def 函数名（参数列表）:
    函数体
```

```python
# 用def定义新函数
def add(x, y):
    print("x is {} and y is {}".format(x, y))
    return x + y    # 用return语句返回

# 调用函数
add(5, 6)   # => 印出"x is 5 and y is 6"并且返回11

# 也可以用关键字参数来调用函数
add(y=6, x=5)   # 关键字参数可以用任何顺序


# 我们可以定义一个可变参数函数
def varargs(*args):
    return args

varargs(1, 2, 3)   # => (1, 2, 3)


# 我们也可以定义一个关键字可变参数函数
def keyword_args(**kwargs):
    return kwargs

# 我们来看看结果是什么：
keyword_args(big="foot", loch="ness")   # => {"big": "foot", "loch": "ness"}


# def(**kwargs) 把N个关键字参数转化为字典:
def func(country,province,**kwargs):
		print(country,province,kwargs)

# 调用函数
>>>func("China","Sichuan",city = "Chengdu", section = "JingJiang")
China Sichuan {'city': 'Chengdu', 'section': 'JingJiang'}


# 这两种可变参数可以混着用
def all_the_args(*args, **kwargs):
    print(args)
    print(kwargs)
"""
all_the_args(1, 2, a=3, b=4) prints:
    (1, 2)
    {"a": 3, "b": 4}
"""

# 调用可变参数函数时可以做跟上面相反的，用*展开序列，用**展开字典。
args = (1, 2, 3, 4)
kwargs = {"a": 3, "b": 4}
all_the_args(*args)   # 相当于 all_the_args(1, 2, 3, 4)
all_the_args(**kwargs)   # 相当于 all_the_args(a=3, b=4)
all_the_args(*args, **kwargs)   # 相当于 all_the_args(1, 2, 3, 4, a=3, b=4)


# 函数作用域
x = 5

def setX(num):
    # 局部作用域的x和全局域的x是不同的
    x = num # => 43
    print (x) # => 43

def setGlobalX(num):
    global x
    print (x) # => 5
    x = num # 现在全局域的x被赋值
    print (x) # => 6

setX(43)
setGlobalX(6)


# 函数在Python是一等公民
def create_adder(x):
    def adder(y):
        return x + y
    return adder

add_10 = create_adder(10)
add_10(3)   # => 13

# 也有匿名函数
(lambda x: x > 2)(3)   # => True

# 内置的高阶函数
map(add_10, [1, 2, 3])   # => [11, 12, 13]
filter(lambda x: x > 5, [3, 4, 5, 6, 7])   # => [6, 7]

# 用列表推导式可以简化映射和过滤。列表推导式的返回值是另一个列表。
[add_10(i) for i in [1, 2, 3]]  # => [11, 12, 13]
[x for x in [3, 4, 5, 6, 7] if x > 5]   # => [6, 7]
```

### 6.类

```python
# 定义一个继承object的类
class Human(object):

    # 类属性，被所有此类的实例共用。
    species = "H. sapiens"

    # 构造方法，当实例被初始化时被调用。注意名字前后的双下划线，这是表明这个属
    # 性或方法对Python有特殊意义，但是允许用户自行定义。你自己取名时不应该用这
    # 种格式。
    def __init__(self, name):
        # Assign the argument to the instance's name attribute
        self.name = name

    # 实例方法，第一个参数总是self，就是这个实例对象
    def say(self, msg):
        return "{name}: {message}".format(name=self.name, message=msg)

    # 类方法，被所有此类的实例共用。第一个参数是这个类对象。
    @classmethod
    def get_species(cls):
        return cls.species

    # 静态方法。调用时没有实例或类的绑定。
    @staticmethod
    def grunt():
        return "*grunt*"


# 构造一个实例
i = Human(name="Ian")
print(i.say("hi"))     # 印出 "Ian: hi"

j = Human("Joel")
print(j.say("hello"))  # 印出 "Joel: hello"

# 调用一个类方法
i.get_species()   # => "H. sapiens"

# 改一个共用的类属性
Human.species = "H. neanderthalensis"
i.get_species()   # => "H. neanderthalensis"
j.get_species()   # => "H. neanderthalensis"

# 调用静态方法
Human.grunt()   # => "*grunt*"
```

### 7. 模块

```Python
# 用import导入模块
import math
print(math.sqrt(16))  # => 4.0

# 也可以从模块中导入个别值
from math import ceil, floor
print(ceil(3.7))  # => 4.0
print(floor(3.7))   # => 3.0

# 可以导入一个模块中所有值
# 警告：不建议这么做
from math import *

# 如此缩写模块名字
import math as m
math.sqrt(16) == m.sqrt(16)   # => True

# Python模块其实就是普通的Python文件。你可以自己写，然后导入，
# 模块的名字就是文件的名字。

# 你可以这样列出一个模块里所有的值
import math
dir(math)

```

### 8. 高级用法

```python
# 用生成器(generators)方便地写惰性运算
def double_numbers(iterable):
    for i in iterable:
        yield i + i

# 生成器只有在需要时才计算下一个值。它们每一次循环只生成一个值，而不是把所有的
# 值全部算好。
#
# range的返回值也是一个生成器，不然一个1到900000000的列表会花很多时间和内存。
#
# 如果你想用一个Python的关键字当作变量名，可以加一个下划线来区分。
range_ = range(1, 900000000)
# 当找到一个 >=30 的结果就会停
# 这意味着 `double_numbers` 不会生成大于30的数。
for i in double_numbers(range_):
    print(i)
    if i >= 30:
        break


# 装饰器(decorators)
# 这个例子中，beg装饰say
# beg会先调用say。如果返回的say_please为真，beg会改变返回的字符串。
from functools import wraps


def beg(target_function):
    @wraps(target_function)
    def wrapper(*args, **kwargs):
        msg, say_please = target_function(*args, **kwargs)
        if say_please:
            return "{} {}".format(msg, "Please! I am poor :(")
        return msg

    return wrapper


@beg
def say(say_please=False):
    msg = "Can you buy me a beer?"
    return msg, say_please


print(say())  # Can you buy me a beer?
print(say(say_please=True))  # Can you buy me a beer? Please! I am poor :(
```

### 9.常用内置库

通常我们写算法可能会用一些Python的内置库，下面列举了一些常用的内置库，具体用法大家可以阅读 [官方文档](https://docs.python.org/3/library/index.html)。

| 包名                                                         | 用途                             |
| :----------------------------------------------------------- | :------------------------------- |
| [`array`](https://docs.python.org/3/library/array.html)      | 定长数组                         |
| [`argparse`](https://docs.python.org/3/library/argparse.html) | 命令行参数处理                   |
| [`bisect`](https://docs.python.org/3/library/bisect.html)    | 二分查找                         |
| [`collections`](https://docs.python.org/3/library/collections.html) | 提供有序字典、双端队列等数据结构 |
| [`fractions`](https://docs.python.org/3/library/fractions.html) | 有理数                           |
| [`heapq`](https://docs.python.org/3/library/heapq.html)      | 基于堆的优先级队列               |
| [`io`](https://docs.python.org/3/library/io.html)            | 文件流、内存流                   |
| [`itertools`](https://docs.python.org/3/library/itertools.html) | 迭代器相关                       |
| [`math`](https://docs.python.org/3/library/math.html)        | 常用数学函数                     |
| [`os.path`](https://docs.python.org/3/library/os.html)       | 系统路径相关                     |
| [`random`](https://docs.python.org/3/library/random.html)    | 随机数                           |
| [`re`](https://docs.python.org/3/library/re.html)            | 正则表达式                       |
| [`struct`](https://docs.python.org/3/library/struct.html)    | 转换结构体和二进制数据           |
| [`sys`](https://docs.python.org/3/library/sys.html)          | 系统信息                         |




