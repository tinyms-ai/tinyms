# 30min速成Python指南

[toc]

## 一、为什么要学习Python

原因很简单：简单，好上手。对于一个初学者来说，大家需要一门非常容易上手实现代码的语言来入门深度学习，Python是一种解释性语言，即使大家没有丰富的计算机知识，也可以快速学习。整体的代码非常清晰，易于阅读，并且功能强大，可以应用到多种场景。目前深度学习很多代码，尤其是发的Paper里的代码大多都有python代码实现，非常方便大家复现和实验。

## 二、如何安装Python

打开[Python官网](https://www.python.org)，进入[下载页面](https://www.python.org/downloads/)：

![image-20210422010607373](https://tva1.sinaimg.cn/large/008i3skNgy1gprxgc8miuj31tq0u0drp.jpg)

下载时，可以选择具体的版本号，这里我们以Python 3.7.5为例![image-20210422010836481](https://tva1.sinaimg.cn/large/008i3skNgy1gprxgkh3vkj31t60p247s.jpg)

点进去根据你的系统（Windows/Mac OS X/Ubuntu等），选择相应的版本包下载安装即可

![image-20210422011032662](https://tva1.sinaimg.cn/large/008i3skNgy1gprxgp5kurj31t20siwpd.jpg)

下载完成后，在终端输入python，输入'Hello,TinyMS'：

![image-20210422011827582](https://tva1.sinaimg.cn/large/008i3skNgy1gprxgugew4j32260h041l.jpg)

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



