# 30min速成Python指南

<h1>Table of Contents<span class="tocSkip"></span></h1>
<div class="toc"><ul class="toc-item"><li><span><a href="#一、为什么要学习Python" data-toc-modified-id="一、为什么要学习Python-1">一、为什么要学习Python</a></span></li><li><span><a href="#二、如何安装Python" data-toc-modified-id="二、如何安装Python-2">二、如何安装Python</a></span><ul class="toc-item"><li><span><a href="#1.-下载并安装Python" data-toc-modified-id="1.-下载并安装Python-2.1">1. 下载并安装Python</a></span></li><li><span><a href="#2.选择合适的Pip源？" data-toc-modified-id="2.选择合适的Pip源？-2.2">2.选择合适的Pip源？</a></span></li></ul></li><li><span><a href="#三、Python基础语法" data-toc-modified-id="三、Python基础语法-3">三、Python基础语法</a></span><ul class="toc-item"><li><span><a href="#1.-原始数据类型与运算符" data-toc-modified-id="1.-原始数据类型与运算符-3.1">1. 原始数据类型与运算符</a></span><ul class="toc-item"><li><span><a href="#Numbers（数字）" data-toc-modified-id="Numbers（数字）-3.1.1">Numbers（数字）</a></span></li><li><span><a href="#String（字符串）" data-toc-modified-id="String（字符串）-3.1.2">String（字符串）</a></span></li><li><span><a href="#List（列表）" data-toc-modified-id="List（列表）-3.1.3">List（列表）</a></span></li><li><span><a href="#Tuples（元祖）" data-toc-modified-id="Tuples（元祖）-3.1.4">Tuples（元祖）</a></span></li><li><span><a href="#Dictionary（字典）" data-toc-modified-id="Dictionary（字典）-3.1.5">Dictionary（字典）</a></span></li><li><span><a href="#Set（集合）" data-toc-modified-id="Set（集合）-3.1.6">Set（集合）</a></span></li><li><span><a href="#数据类型转换" data-toc-modified-id="数据类型转换-3.1.7">数据类型转换</a></span></li></ul></li><li><span><a href="#2.条件语句" data-toc-modified-id="2.条件语句-3.2">2.条件语句</a></span></li><li><span><a href="#3.循环语句" data-toc-modified-id="3.循环语句-3.3">3.循环语句</a></span></li><li><span><a href="#4.迭代器" data-toc-modified-id="4.迭代器-3.4">4.迭代器</a></span><ul class="toc-item"><li><span><a href="#1.迭代器是什么？" data-toc-modified-id="1.迭代器是什么？-3.4.1">1.迭代器是什么？</a></span></li><li><span><a href="#2.TinyMS中的迭代器" data-toc-modified-id="2.TinyMS中的迭代器-3.4.2">2.TinyMS中的迭代器</a></span></li></ul></li><li><span><a href="#5.生成器" data-toc-modified-id="5.生成器-3.5">5.生成器</a></span><ul class="toc-item"><li><span><a href="#1.-生成器是什么？" data-toc-modified-id="1.-生成器是什么？-3.5.1">1. 生成器是什么？</a></span></li><li><span><a href="#2.-为什么要使用yield函数？" data-toc-modified-id="2.-为什么要使用yield函数？-3.5.2">2. 为什么要使用yield函数？</a></span></li><li><span><a href="#3.-在TinyMS中如何使用yield函数？" data-toc-modified-id="3.-在TinyMS中如何使用yield函数？-3.5.3">3. 在TinyMS中如何使用yield函数？</a></span></li></ul></li><li><span><a href="#6.函数" data-toc-modified-id="6.函数-3.6">6.函数</a></span><ul class="toc-item"><li><span><a href="#1.-定义函数" data-toc-modified-id="1.-定义函数-3.6.1">1. 定义函数</a></span></li><li><span><a href="#2.-函数作用域" data-toc-modified-id="2.-函数作用域-3.6.2">2. 函数作用域</a></span></li><li><span><a href="#3.嵌套函数" data-toc-modified-id="3.嵌套函数-3.6.3">3.嵌套函数</a></span></li><li><span><a href="#4.匿名函数" data-toc-modified-id="4.匿名函数-3.6.4">4.匿名函数</a></span></li><li><span><a href="#5.内置的高阶函数" data-toc-modified-id="5.内置的高阶函数-3.6.5">5.内置的高阶函数</a></span></li><li><span><a href="#6.列表推导式" data-toc-modified-id="6.列表推导式-3.6.6">6.列表推导式</a></span></li></ul></li><li><span><a href="#7.类" data-toc-modified-id="7.类-3.7">7.类</a></span><ul class="toc-item"><li><span><a href="#1.定义一个继承object的类" data-toc-modified-id="1.定义一个继承object的类-3.7.1">1.定义一个继承object的类</a></span></li><li><span><a href="#2.构造一个实例" data-toc-modified-id="2.构造一个实例-3.7.2">2.构造一个实例</a></span></li><li><span><a href="#3.调用一个类方法" data-toc-modified-id="3.调用一个类方法-3.7.3">3.调用一个类方法</a></span></li><li><span><a href="#4.改一个共用的类属性" data-toc-modified-id="4.改一个共用的类属性-3.7.4">4.改一个共用的类属性</a></span></li><li><span><a href="#5.调用静态方法" data-toc-modified-id="5.调用静态方法-3.7.5">5.调用静态方法</a></span></li></ul></li><li><span><a href="#8.模块" data-toc-modified-id="8.模块-3.8">8.模块</a></span><ul class="toc-item"><li><span><a href="#1.用import导入模块" data-toc-modified-id="1.用import导入模块-3.8.1">1.用import导入模块</a></span></li><li><span><a href="#2.from-xxx-import" data-toc-modified-id="2.from-xxx-import-3.8.2">2.from xxx import</a></span></li><li><span><a href="#3.其他用法" data-toc-modified-id="3.其他用法-3.8.3">3.其他用法</a></span></li></ul></li><li><span><a href="#9.高级用法" data-toc-modified-id="9.高级用法-3.9">9.高级用法</a></span><ul class="toc-item"><li><span><a href="#1.-用生成器(generators)写函数计算" data-toc-modified-id="1.-用生成器(generators)写函数计算-3.9.1">1. 用生成器(generators)写函数计算</a></span></li><li><span><a href="#2.-装饰器" data-toc-modified-id="2.-装饰器-3.9.2">2. 装饰器</a></span></li></ul></li><li><span><a href="#10.-常用内置库" data-toc-modified-id="10.-常用内置库-3.10">10. 常用内置库</a></span></li></ul></li><li><span><a href="#四、深度学习常用Python库" data-toc-modified-id="四、深度学习常用Python库-4">四、深度学习常用Python库</a></span><ul class="toc-item"><li><span><a href="#1.-Numpy" data-toc-modified-id="1.-Numpy-4.1">1. Numpy</a></span></li><li><span><a href="#2.Scipy" data-toc-modified-id="2.Scipy-4.2">2.Scipy</a></span></li><li><span><a href="#3.Matplotlib" data-toc-modified-id="3.Matplotlib-4.3">3.Matplotlib</a></span></li></ul></li><li><span><a href="#五、几个练习小技巧" data-toc-modified-id="五、几个练习小技巧-5">五、几个练习小技巧</a></span></li><li><span><a href="#六、参考文献" data-toc-modified-id="六、参考文献-6">六、参考文献</a></span></li></ul></div>

## 一、为什么要学习Python

原因很简单：简单，好上手。对于一个初学者来说，大家需要一门非常容易上手实现代码的语言来入门深度学习，Python是一种解释性语言，即使大家没有丰富的计算机知识，也可以快速学习。整体的代码非常清晰，易于阅读，并且功能强大，可以应用到多种场景。目前深度学习很多代码，尤其是发的Paper里的代码大多都有python代码实现，非常方便大家复现和实验。

## 二、如何安装Python

### 1. 下载并安装Python

打开[Python官网](https://www.python.org)，进入[下载页面](https://www.python.org/downloads/)：

![image-20210422010607373](https://tva1.sinaimg.cn/large/008i3skNgy1gprxgc8miuj31tq0u0drp.jpg)

下载时，可以选择具体的版本号，这里我们以Python 3.7.5为例![image-20210422010836481](https://tva1.sinaimg.cn/large/008i3skNgy1gprxqzrt1ij31t60p2gq4.jpg)

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



#### Numbers（数字）

Python3 支持 int、float、bool、complex（复数）。
在Python 3里，只有一种整数类型 int，表示为长整型，没有 python2 中的 Long。
像大多数语言一样，数值类型的赋值和计算都是很直观的。
内置的 type() 函数可以用来查询变量所指的对象类型。



```python
a, b, c, d = 20, 5.5, True, 4+3j
print(type(a), type(b), type(c), type(d))

```

    <class 'int'> <class 'float'> <class 'bool'> <class 'complex'>


- 数值运算


```python
# 整数
3  # => 3
```




    3




```python
1 + 1  # => 2
```




    2




```python
8 - 1  # => 7
```




    7




```python
10 * 2  # => 20
```




    20




```python
# 除法，会自动转换成浮点数
35 / 5  # => 7.0
```




    7.0




```python
5 / 3  # => 1.6666666666666667
```




    1.6666666666666667




```python
# 浮点数的运算结果也是浮点数
3 * 2.0 # => 6.0
```




    6.0




```python
# 模除
7 % 3 # => 1
```




    1




```python
# x的y次方
2**4 # => 16
```




    16




```python
# 用括号决定优先级
(1 + 3) * 2  # => 8
```




    8



-----
注意：
- 1、Python可以同时为多个变量赋值，如a, b = 1, 2。
- 2、一个变量可以通过赋值指向不同类型的对象。
- 3、数值的除法包含两个运算符：/ 返回一个浮点数，// 返回一个整数。
- 4、在混合计算时，Python会把整型转换成为浮点数。


#### String（字符串）

- Python中的字符串用单引号 ' 或双引号 " 括起来，同时使用反斜杠 \ 转义特殊字符。截取格式：变量[头下标:尾下标]


```python
str = 'Hello，TinyMS'

print (str)          # 输出字符串
print (str[0:-1])    # 输出第一个到倒数第二个的所有字符
print (str[0])       # 输出字符串第一个字符
print (str[2:5])     # 输出从第三个开始到第五个的字符
print (str[2:])      # 输出从第三个开始的后的所有字符
print (str * 2)      # 输出字符串两次，也可以写成 print (2 * str) 
print (str + "TEST") # 连接字符串
```

    Hello，TinyMS
    Hello，TinyM
    H
    llo
    llo，TinyMS
    Hello，TinyMSHello，TinyMS
    Hello，TinyMSTEST


- 使用反斜杠 \ 来转义特殊字符，如果你不想让反斜杠发生转义，可以在字符串前面添加一个 r，表示原始字符串：


```python
print('Hello\nTinyMS')

print(r'Hello\nTinyMS')
```

    Hello
    TinyMS
    Hello\nTinyMS


- 字符串拼接，可使用 + 进行拼接，也可以使用join方法


```python
"Hello" + " " + "TinyMS!"
```




    'Hello TinyMS!'




```python
" ".join(["Hello", "TinyMS!"])
```




    'Hello TinyMS!'



- 字符串长度


```python
len("Hello TinyMS!")
```




    13



- 字符串转大小写


```python
"Hello TinyMS!".upper()
```




    'HELLO TINYMS!'




```python
"Hello TinyMS!".lower()
```




    'hello tinyms!'



#### List（列表）

List（列表） 在其他语言中通常叫数组`Array`，是 Python 中使用最频繁的数据类型。
列表可以完成大多数集合类的数据结构实现。列表中元素的类型可以不相同，它支持数字，字符串甚至可以包含列表（所谓嵌套）。
列表是写在方括号 [] 之间、用逗号分隔开的元素列表。
和字符串一样，列表同样可以被索引和截取，列表被截取后返回一个包含所需元素的新列表。
列表截取的语法格式如下：

> 变量[头下标:尾下标]

索引值以 0 为开始值，-1 为从末尾的开始位置。

- 定义列表


```python
# 定义一个空列表
empty = []
empty
```




    []




```python
# 定义一个含有数字元素的列表
numbers = [1, 2, 3, 100]
numbers
```




    [1, 2, 3, 100]




```python
# 定义一个含有字符串元素的列表
stringlist = ["this", "is", "a", "python", "tutorial"]
stringlist
```




    ['this', 'is', 'a', 'python', 'tutorial']




```python
# 定义含有多重数据类型的列表
mixed_types = ["Hello TinyMS", [1, 2, 3], False]
mixed_types
```




    ['Hello TinyMS', [1, 2, 3], False]



- 修改列表


```python
numbers[0] = 5
numbers
```




    [5, 2, 3, 100]




```python
numbers[1:3] = [111,222]
numbers
```




    [5, 111, 222, 100]




```python
numbers[2:4] = []
numbers
```




    [5, 111]



- 列表相加


```python
a = ["奶茶","火锅","小龙虾"]
b = ["披萨","炸鸡","啤酒"]
deliciousfood = a + b
deliciousfood
```




    ['奶茶', '火锅', '小龙虾', '披萨', '炸鸡', '啤酒']



除了以上常用的语法以外，list还内置了很多方法，譬如`append()`、`pop()`等等，下面来一一说明:

- list.append():添加元素


```python
deliciousfood.append("橙子")
deliciousfood
```




    ['啤酒', '奶茶', '小龙虾', '披萨', '火锅', '炸鸡', '橙子', '橙子']



- list.sort()：元素排序


```python
deliciousfood.sort()
deliciousfood
```




    ['啤酒', '奶茶', '小龙虾', '披萨', '橙子', '橙子', '火锅', '炸鸡']




```python
a = [1,4,12,52,67,24,77,23,0]
a.sort()
a
```




    [0, 1, 4, 12, 23, 24, 52, 67, 77]



- list.pop()：移除元素


```python
a.pop(2)
a
```




    [0, 1, 12, 23, 24, 52, 67, 77]



#### Tuples（元祖）

元祖和列表类似，但是元祖的元素不能修改，创建以后不能改变。


```python
employee = ("Jane", "Doe", 31, "Software Developer")

employee[0] = "John"
```


    ---------------------------------------------------------------------------
    
    TypeError                                 Traceback (most recent call last)
    
    <ipython-input-37-b12d1c27d9d5> in <module>
          1 employee = ("Jane", "Doe", 31, "Software Developer")
          2 
    ----> 3 employee[0] = "John"


    TypeError: 'tuple' object does not support item assignment


- 元祖切分


```python
employee = ("Jane", "Doe", 31, "Software Developer")
employee[0]
```




    'Jane'




```python
employee[1:3]
```




    ('Doe', 31)



- 元祖相加


```python
tuple1 = (1,2)
tuple2 = (3,4)
tuple3 = tuple1 + tuple2
tuple3
```




    (1, 2, 3, 4)



- 元祖和列表转化


```python
tuple1 = (1,2)
list(tuple1)
```




    [1, 2]



因为元祖是不可变序列，所以很多对列表适用的方法对元祖并不是适用，但元祖也有两个内置的方法：
- tuple.count()：统计元祖中出现元素的数量
- tuple.index()：输出元祖中某个元素的索引index，如果元祖中不存在，会报ValueError


```python
# tuple.count()
letters = ("a", "b", "b", "c", "a")
letters.count("a")
```




    2




```python
# tuple.index()
letters = ("a", "b", "b", "c", "a")
letters.index("a")
# letters.index("d")
```




    0



#### Dictionary（字典）
列表是有序的对象集合，字典是无序的对象集合。两者之间的区别在于：字典当中的元素是通过键来存取的，而不是通过偏移存取。

字典是一种映射类型，字典用 { } 标识，它是一个无序的 键(key) : 值(value) 的集合。

键(key)必须使用不可变类型。

在同一个字典中，键(key)必须是唯一的。

- 创建字典


```python
person1 = {"name": "Charlotte", "age": 27, "sex": "female"}
person1
```




    {'name': 'Charlotte', 'age': 27, 'sex': 'female'}




```python
#person2 = dict(name="Charlotte", age=27, sex="female")
person2 = dict([('name','Charlotte'),('age',27),('sex','female')])
person2
```




    {'name': 'Charlotte', 'age': 27, 'sex': 'female'}




```python
print (person1['name'])        # 输出键为 'name' 的值
print (person1['age'])         # 输出键为 age 的值
print (person2)                # 输出完整的字典
print (person2.keys())         # 输出所有键
print (person2.values())       # 输出所有值
print (person2.items())
```

    Charlotte
    27
    {'name': 'Charlotte', 'age': 27, 'sex': 'female'}
    dict_keys(['name', 'age', 'sex'])
    dict_values(['Charlotte', 27, 'female'])
    dict_items([('name', 'Charlotte'), ('age', 27), ('sex', 'female')])


------
注意：
- 1、字典是一种映射类型，它的元素是键值对。
- 2、字典的关键字必须为不可变类型，且不能重复。
- 3、创建空字典使用 { }。

#### Set（集合）

集合（set）是由一个或数个形态各异的大小整体组成的，构成集合的事物或对象称作元素或是成员。

基本功能是进行成员关系测试和删除重复元素。

可以使用大括号 { } 或者 set() 函数创建集合，注意：创建一个空集合必须用 set() 而不是 { }，因为 { } 是用来创建一个空字典。


```python
employees1 = {"John", "Jane", "Linda"}
employees1
```




    {'Jane', 'John', 'Linda'}




```python
employees2 = set(["David", "Mark", "Marie"])
employees2
```




    {'David', 'Marie', 'Mark'}




```python
empty = set()
empty
```




    set()



集合最常用的功能是用来移除重复元素：


```python
set([1,1,2,2,3,3,4,5,6,7])
```




    {1, 2, 3, 4, 5, 6, 7}



#### 数据类型转换

有时候，我们需要对数据内置的类型进行转换，数据类型的转换，你只需要将数据类型作为函数名即可。

以下几个内置的函数可以执行数据类型之间的转换。这些函数返回一个新的对象，表示转换的值。

| 函数                                                         | 描述                                                |
| :----------------------------------------------------------- | :-------------------------------------------------- |
| [int(x [,base\])](https://www.runoob.com/python3/python-func-int.html) | 将x转换为一个整数                                   |
| [float(x)](https://www.runoob.com/python3/python-func-float.html) | 将x转换到一个浮点数                                 |
| [complex(real [,imag\])](https://www.runoob.com/python3/python-func-complex.html) | 创建一个复数                                        |
| [str(x)](https://www.runoob.com/python3/python-func-str.html) | 将对象 x 转换为字符串                               |
| [repr(x)](https://www.runoob.com/python3/python-func-repr.html) | 将对象 x 转换为表达式字符串                         |
| [eval(str)](https://www.runoob.com/python3/python-func-eval.html) | 用来计算在字符串中的有效Python表达式,并返回一个对象 |
| [tuple(s)](https://www.runoob.com/python3/python3-func-tuple.html) | 将序列 s 转换为一个元组                             |
| [list(s)](https://www.runoob.com/python3/python3-att-list-list.html) | 将序列 s 转换为一个列表                             |
| [set(s)](https://www.runoob.com/python3/python-func-set.html) | 转换为可变集合                                      |
| [dict(d)](https://www.runoob.com/python3/python-func-dict.html) | 创建一个字典。d 必须是一个 (key, value)元组序列。   |
| [frozenset(s)](https://www.runoob.com/python3/python-func-frozenset.html) | 转换为不可变集合                                    |
| [chr(x)](https://www.runoob.com/python3/python-func-chr.html) | 将一个整数转换为一个字符                            |
| [ord(x)](https://www.runoob.com/python3/python-func-ord.html) | 将一个字符转换为它的整数值                          |
| [hex(x)](https://www.runoob.com/python3/python-func-hex.html) | 将一个整数转换为一个十六进制字符串                  |
| [oct(x)](https://www.runoob.com/python3/python-func-oct.html) | 将一个整数转换为一个八进制字符串                    |



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

### 2.条件语句

Python条件语句是通过一条或多条语句的执行结果（True或者False）来决定执行的代码块，没有switch/case语句

常见格式：


```python
if expr0:
    # Run if expr0 is true
    # Your code goes here...
elif expr1:
    # Run if expr1 is true
    # Your code goes here...
elif expr2:
    # Run if expr2 is true
    # Your code goes here...
...
else:
    # Run if all expressions are false
    # Your code goes here...

# Next statement
```


```python
age = 21
if age >= 18:
    print("You're a legal adult")
```

    You're a legal adult



```python
age = 16
if age >= 18:
    print("You're a legal adult")
else:
    print("You're NOT an adult")
```

    You're NOT an adult



```python
age = 18
if age > 18:
    print("You're over 18 years old")
elif age == 18:
    print("You're exactly 18 years old")
```

    You're exactly 18 years old


### 3.循环语句
- for循环：有限迭代，或者执行确定次数的重复
- while循环：无限迭代，或者在设定条件下停止前重复


```python
for i in (1, 2, 3, 4, 5):
    print(i)
else:
    print("The loop wasn't interrupted")
```

    1
    2
    3
    4
    5
    The loop wasn't interrupted



```python
count = 1
while count < 5:
    print(count)
    count = count + 1
else:
    print("The loop wasn't interrupted")
```

    1
    2
    3
    4
    The loop wasn't interrupted


### 4.迭代器

迭代器和生成器在Python中非常常见，也是Python最强大的功能之一，尤其在数据读取时，经常会用到迭代器和生成器，可以节省内存空间，显著提升代码运行速度。

#### 1.迭代器是什么？

- 迭代器是一个可以记住遍历的位置的对象，迭代器的对象从集合的第一个元素开始访问，直到所有的元素被访问完结束；迭代器只能往前不会后退；字符串，列表或元组对象都可用于创建迭代器。
- 基本方法：**iter()** 和 **next()**
- 创建迭代器对象：


```python
# 字符串
for s in "Hello,TinyMS":
    print (s)
```

    H
    e
    l
    l
    o
    ,
    T
    i
    n
    y
    M
    S



```python
# 列表  
list_a=[1,2,3,4]
it = iter(list_a)    # 创建迭代器对象
print (next(it))   # 输出迭代器的下一个元素

# print (next(it))
```

    1



```python
# 元祖
tuple_a=(1,2,3,4)
it = iter(tuple_a)
print (next(it))   # 输出迭代器的下一个元素

#print (next(it))
```

    1



```python
# 字典(注意字典是无序的)
d = {
  'apple' : 'tasty',
  'bananas' : 'the best',
  'brussel sprouts' : 'evil',
  'cauliflower' : 'pretty good'
}
for Key in d:
    print  ("{0} are {1}".format(Key,d[Key]))
```

    apple are tasty
    bananas are the best
    brussel sprouts are evil
    cauliflower are pretty good


- 遍历迭代器对象


```python
list=[1,2,3,4]
it = iter(list)    # 创建迭代器对象
for x in it:
    print (x, end=" ")
```

    1 2 3 4 

- 构造一个迭代器

把一个类作为一个迭代器使用需要在类中实现两个方法 __iter__() 与 __next__()
- 1. __iter__() 方法返回一个特殊的迭代器对象， 这个迭代器对象实现了 __next__() 方法并通过 StopIteration 异常标识迭代的完成
- 2. __next__() 方法（Python 2 里是 next()）会返回下一个迭代器对象。

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

    1
    2
    3
    4
    5


- 结束迭代标识：StopIteration

StopIteration 异常用于标识迭代的完成，防止出现无限循环的情况，在 __next__() 方法中我们可以设置在完成指定循环次数后触发 StopIteration 异常来结束迭代。

在 10 次迭代后停止执行：


```python
class MyNumbers:
    def __iter__(self):
        self.a = 1
        return self
    def __next__(self):
        if self.a <= 10:
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


#### 2.TinyMS中的迭代器

TinyMS中的data模块，继承了MindSpore的dataset类中的engine模块的功能，可以看到在engine的代码中我们定义了数据读取的基类迭代器[Iterator](https://gitee.com/mindspore/mindspore/blob/master/mindspore/dataset/engine/iterators.py)，再根据不同的数据类型，分别创建了DictIterator、TupleIterator等，举例看一下:


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

### 5.生成器

#### 1. 生成器是什么？

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
```


```python
while True:
    try:
        print (next(f),end=" ")
    except StopIteration:
        break
```

    0 1 1 2 3 5 8 13 21 34 55 

#### 2. 为什么要使用yield函数？

一个带有 yield 的函数就是一个 generator，它和普通函数不同，生成一个 generator 看起来像函数调用，但不会执行任何函数代码，直到对其调用 next()（在 for 循环中会自动调用 next()）才开始执行。虽然执行流程仍按函数的流程执行，但每执行到一个 yield 语句就会中断，并返回一个迭代值，下次执行时从 yield 的下一个语句继续执行。看起来就好像一个函数在正常执行的过程中被 yield 中断了数次，每次中断都会通过 yield 返回当前的迭代值。

yield 的好处是显而易见的，把一个函数改写为一个 generator 就获得了迭代能力，比起用类的实例保存状态来计算下一个 next() 的值，不仅代码简洁，而且执行流程异常清晰。


```python
f = fibonacci(5)
```


```python
next(f)
```




    0




```python
next(f)
```




    1




```python
next(f)
```




    1




```python
next(f)
```




    2




```python
next(f)
```




    3




```python
next(f)
```




    5




```python
next(f)
```


    ---------------------------------------------------------------------------
    
    StopIteration                             Traceback (most recent call last)
    
    <ipython-input-131-aff1dd02a623> in <module>
    ----> 1 next(f)


    StopIteration: 


另一个使用 **yield** 的好处在文件读取过程。如果直接对文件对象调用 read() 方法，会导致不可预测的内存占用。好的方法是利用固定长度的缓冲区来不断读取文件内容。通过 yield，我们不再需要编写读文件的迭代类，就可以轻松实现文件读取：


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

#### 3. 在TinyMS中如何使用yield函数？

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


```python
# 输出结果
{'data': Tensor(shape=[3], dtype=Int64, value= [3, 4, 5])}
{'data': Tensor(shape=[3], dtype=Int64, value= [2, 3, 4])}
{'data': Tensor(shape=[3], dtype=Int64, value= [4, 5, 6])}
{'data': Tensor(shape=[3], dtype=Int64, value= [1, 2, 3])}
{'data': Tensor(shape=[3], dtype=Int64, value= [0, 1, 2])}
```

### 6.函数

语法：
```python
def 函数名（参数列表）:
    函数体
```

#### 1. 定义函数

- 用def定义新函数


```python
def add(x, y):
    print("x is {} and y is {}".format(x, y))
    return x + y    # 用return语句返回
```

- 调用函数


```python

add(5, 6)   # => 印出"x is 5 and y is 6"并且返回11
```

    x is 5 and y is 6





    11



- 用关键字参数来调用函数


```python
add(y=6, x=5)   # 关键字参数可以用任何顺序
```

    x is 5 and y is 6





    11



- 定义可变参数函数


```python
def varargs(*args):
    return args

varargs(1, 2, 3)   # => (1, 2, 3)
```




    (1, 2, 3)



- 定义一个关键字可变参数函数


```python
def keyword_args(**kwargs):
    return kwargs
```


```python
keyword_args(big="foot", loch="ness")
```




    {'big': 'foot', 'loch': 'ness'}



- def(**kwargs)：把N个关键字参数转化为字典


```python
def func(country,province,**kwargs):
    print(country,province,kwargs)
```


```python
# 调用函数
func("China","Sichuan",city = "Chengdu", section = "JingJiang")
```

    China Sichuan {'city': 'Chengdu', 'section': 'JingJiang'}



```python
# 这两种可变参数可以混着用
def all_the_args(*args, **kwargs):
    print(args)
    print(kwargs)
```


```python
all_the_args(1, 2, a=3, b=4)
```

    (1, 2)
    {'a': 3, 'b': 4}



```python
# 调用可变参数函数时可以做跟上面相反的，用*展开序列，用**展开字典。
args = (1, 2, 3, 4)
kwargs = {"a": 3, "b": 4}
all_the_args(*args)   # 相当于 all_the_args(1, 2, 3, 4)
all_the_args(**kwargs)   # 相当于 all_the_args(a=3, b=4)
all_the_args(*args, **kwargs)   # 相当于 all_the_args(1, 2, 3, 4, a=3, b=4)
```

    (1, 2, 3, 4)
    {}
    ()
    {'a': 3, 'b': 4}
    (1, 2, 3, 4)
    {'a': 3, 'b': 4}


#### 2. 函数作用域


```python
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
```

    43
    5
    6


#### 3.嵌套函数


```python
def create_adder(x):
    def adder(y):
        return x + y
    return adder

add_10 = create_adder(10)
add_10(3)   # => 13
```




    13



#### 4.匿名函数


```python
(lambda x: x > 2)(3)   # => True
```




    True



#### 5.内置的高阶函数


```python
map(add_10, [1, 2, 3])   # => [11, 12, 13]
filter(lambda x: x > 5, [3, 4, 5, 6, 7])   # => [6, 7]
```




    <filter at 0x112e06588>



#### 6.列表推导式
可以简化映射和过滤：列表推导式的返回值是另一个列表。


```python
[add_10(i) for i in [1, 2, 3]]  # => [11, 12, 13]
[x for x in [3, 4, 5, 6, 7] if x > 5]   # => [6, 7]
```




    [6, 7]



### 7.类

#### 1.定义一个继承object的类


```python
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
```

#### 2.构造一个实例


```python
i = Human(name="Ian")
print(i.say("hi"))     # 印出 "Ian: hi"

j = Human("Joel")
print(j.say("hello"))  # 印出 "Joel: hello"
```

    Ian: hi
    Joel: hello


#### 3.调用一个类方法


```python
i.get_species()   # => "H. sapiens"
```




    'H. sapiens'



#### 4.改一个共用的类属性


```python
Human.species = "H. neanderthalensis"
i.get_species()   # => "H. neanderthalensis"
j.get_species()   # => "H. neanderthalensis"
```




    'H. neanderthalensis'



#### 5.调用静态方法


```python
Human.grunt()   # => "*grunt*"
```




    '*grunt*'



### 8.模块

#### 1.用import导入模块


```python
import math
print(math.sqrt(16))  # => 4.0
```

    4.0


#### 2.from xxx import


```python
from math import ceil, floor
print(ceil(3.7))  # => 4.0
print(floor(3.7))   # => 3.0
```

    4
    3


#### 3.其他用法


```python
# 可以导入一个模块中所有，但不建议
from math import *

# 缩写模块名字
import math as m
math.sqrt(16) == m.sqrt(16)   # => True

# Python模块其实就是普通的Python文件。你可以自己写，然后导入，
# 模块的名字就是文件的名字。

# 你可以这样列出一个模块里所有的值
import math
dir(math)
```




    ['__doc__',
     '__file__',
     '__loader__',
     '__name__',
     '__package__',
     '__spec__',
     'acos',
     'acosh',
     'asin',
     'asinh',
     'atan',
     'atan2',
     'atanh',
     'ceil',
     'copysign',
     'cos',
     'cosh',
     'degrees',
     'e',
     'erf',
     'erfc',
     'exp',
     'expm1',
     'fabs',
     'factorial',
     'floor',
     'fmod',
     'frexp',
     'fsum',
     'gamma',
     'gcd',
     'hypot',
     'inf',
     'isclose',
     'isfinite',
     'isinf',
     'isnan',
     'ldexp',
     'lgamma',
     'log',
     'log10',
     'log1p',
     'log2',
     'modf',
     'nan',
     'pi',
     'pow',
     'radians',
     'remainder',
     'sin',
     'sinh',
     'sqrt',
     'tan',
     'tanh',
     'tau',
     'trunc']



### 9.高级用法

#### 1. 用生成器(generators)写函数计算


```python
def double_numbers(iterable):
    for i in iterable:
        yield i + i
        
# 如果你想用一个Python的关键字当作变量名，可以加一个下划线来区分。
range_ = range(1, 900000000)
# 当找到一个 >=30 的结果就会停
# 这意味着 `double_numbers` 不会生成大于30的数。
for i in double_numbers(range_):
    print(i)
    if i >= 30:
        break
```

    2
    4
    6
    8
    10
    12
    14
    16
    18
    20
    22
    24
    26
    28
    30


#### 2. 装饰器


```python
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

    Can you buy me a beer?
    Can you buy me a beer? Please! I am poor :(


### 10. 常用内置库

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


## 四、深度学习常用Python库

### 1. Numpy

NumPy(Numerical Python) 是 Python 语言的一个扩展程序库，支持大量的维度数组与矩阵运算，此外也针对数组运算提供大量的数学函数库。

NumPy 的前身 Numeric 最早是由 Jim Hugunin 与其它协作者共同开发，2005 年，Travis Oliphant 在 Numeric 中结合了另一个同性质的程序库 Numarray 的特色，并加入了其它扩展而开发了 NumPy。NumPy 为开放源代码并且由许多协作者共同维护开发。

NumPy 是一个运行速度非常快的数学库，主要用于数组计算，包含：

- 一个强大的N维数组对象 ndarray
- 广播功能函数
- 整合 C/C++/Fortran 代码的工具
- 线性代数、傅里叶变换、随机数生成等功能

### ![NumPy速查表](https://tva1.sinaimg.cn/large/008i3skNgy1gpsyiqhazvj313y0u0kf1.jpg)

### 2.Scipy


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```

### 3.Matplotlib


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```

## 五、几个练习小技巧

1. 找一些难度适中的题目去练习，熟悉Python语法
2. 刷Leetcode，选择Python语言，尝试各种不同的解法
3. 在工作中用Python，不断优化代码，提升处理效率
4. 尝试用Python去复现机器学习/深度学习的模型，能够帮助你深入理解算法，提升代码能力

## 六、参考文献

1.[Look Ma, No For-Loops: Array Programming With NumPy](https://realpython.com/numpy-array-programming)

2.[Python基础教程](https://www.runoob.com/python/python-variable-types.html)

3.[LearnXInYMinutes](https://learnxinyminutes.com/docs/zh-cn/python-cn/)

4.https://oi-wiki.org/lang/python/#_12


```python

```
