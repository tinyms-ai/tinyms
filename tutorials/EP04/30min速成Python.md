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
 


