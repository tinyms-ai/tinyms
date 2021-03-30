# 30min速成Shell脚本

[toc]

## 一、shell是什么？

我们常说的shell脚本，是一种为shell编写的脚本程序，shell编程非常简洁，只需要在有一个能写代码的文本编辑器，比如上文提到的vim，和能解释执行的脚本解释器就行。先看一下你的电脑里装了哪些shell呢？

输入：cat  /etc/shells

```bash
# List of acceptable shells for chpass(1).
# Ftpd will not allow users to connect who are not using
# one of these shells.

/bin/bash
/bin/csh
/bin/dash
/bin/ksh
/bin/sh
/bin/tcsh
/bin/zsh
```

一般我们常用的是Bash，也就是Bourne Again Shell，也是大多数Linux系统默认的shell，不过我们基本不会可以区分Bourne Shell 和 Bourne Again Shell，所以像 **#!/bin/sh**，它同样也可以改为 **#!/bin/bash**。

>  **#!** 意思是告诉系统其后路径所指定的程序即是解释此脚本文件的 Shell 程序

### 如何运行shell文件？

- **1.**作为可执行程序

vi hello.sh

```bash
chmod +x ./test.sh #使脚本具有执行权限

./test.sh #执行脚本

```

> 注意执行的时候一定要写成 **./test.sh**，而不是 **test.sh**，运行其它二进制的程序也一样，直接写 test.sh，linux 系统会去 PATH 里寻找有没有叫 test.sh 的，而只有 /bin, /sbin, /usr/bin，/usr/sbin 等在 PATH 里，你的当前目录通常不在 PATH 里，所以写成 test.sh 是会找不到命令的，要用 ./test.sh 告诉系统说，就在当前目录找。

- **2**.作为解释器参数

这种运行方式是，直接运行解释器，其参数就是 shell 脚本的文件名，如：

```bash
sh test.sh
bash test.sh
```

## 二、变量和参数

### 2.1 变量

```bash
#!/bin/bash
# 脚本的第一行叫 shebang，用来告知系统如何执行该脚本:

# 显示 “Hello world!”
echo Hello world!

# 每一句指令以换行或分号隔开：
echo 'This is the Day One Tutorial'; echo 'This is Day Two Tutorial'

# 声明一个变量：
a="Hello TinyMS"

# 下面是错误的做法：
a = "Hello TinyMS"
# Bash 会把 a 当做一个指令，由于找不到该指令，因此这里会报错。

# 也不可以这样：
a= 'Hello TinyMS'
# Bash 会认为 'Hello，TinyMS' 是一条指令，由于找不到该指令，这里再次报错。
# （这个例子中 'a=' 这部分会被当作仅对 'Hello，TinyMS' 起作用的赋值。）

# 使用变量：
echo $a
echo "$a"
echo '$a'
# 当你赋值 (assign) 、导出 (export)，或者以其他方式使用变量时，变量名前不加 $。
# 如果要使用变量的值， 则要加 $。
# 注意： ' (单引号) 不会展开变量（即会屏蔽掉变量）。

# 在变量内部进行字符串代换
echo ${a/Hello/Hi}
# 会把 a 中首次出现的 "some" 替换成 “A”。

# 变量的截取
Length=6
echo ${a:0:Length}
# 这样会仅返回变量值的前7个字符

# 变量的默认值
echo ${a:-"DefaultValueIfFooIsMissingOrEmpty"}
# 对 null (Foo=) 和空串 (Foo="") 起作用； 零（Foo=0）时返回0
# 注意这仅返回默认值而不是改变变量的值

```

几个内置变量

| 参数处理 | 说明                                                         |
| :------- | :----------------------------------------------------------- |
| $#       | 传递到脚本的参数个数                                         |
| $*       | 以一个单字符串显示所有向脚本传递的参数。 如"$*"用「"」括起来的情况、以"$1 $2 … $n"的形式输出所有参数。 |
| $$       | 脚本运行的当前进程ID号                                       |
| $!       | 后台运行的最后一个进程的ID号                                 |
| $@       | 与$*相同，但是使用时加引号，并在引号中返回每个参数。 如"$@"用「"」括起来的情况、以"$1" "$2" … "$n" 的形式输出所有参数。 |
| $-       | 显示Shell使用的当前选项，与[set命令](https://www.runoob.com/linux/linux-comm-set.html)功能相同。 |
| $?       | 显示最后命令的退出状态。0表示没有错误，其他任何值表明有错误。 |

```bash
# 内置变量：
echo "Last program return value: $?"
echo "Script's PID: $$"
echo "Number of arguments: $#"
echo "Scripts arguments: $@"
echo "Scripts arguments separated in different as: $1 $2..."
```

$* 与 $@ 区别：

- 相同点：都是引用所有参数。
- 不同点：只有在双引号中体现出来。假设在脚本运行时写了三个参数 1、2、3，，则 " * " 等价于 "1 2 3"（传递了一个参数），而 "@" 等价于 "1" "2" "3"（传递了三个参数）。

```bash
#!/bin/bash

echo "-- \$* example ---"
for i in "$*"; do
    echo $i
done

echo "-- \$@ example ---"
for i in "$@"; do
    echo $i
done
```

执行脚本，输出结果如下所示：

```
$ chmod +x test.sh 
$ ./test.sh 1 2 3
-- $* example ---
1 2 3
-- $@ example ---
1
2
3
```

### 2.2 传递参数

- 读取输入：

```bash
echo "What's your name?"
read Name # 这里不需要声明新变量，输入自己的名字
# 在屏幕输入你的名字，这里我输入的是Charlotte
echo Hello, $Name!
```
输出结果：
```bash
Hello,Charlotte
```

- 传递参数

```bash
#!/bin/bash

echo "Shell 传递参数！";
echo "执行的文件名：$0";
echo "第一个参数为：$1";
echo "第二个参数为：$2";
echo "第三个参数为：$3";
```
输出结果：
```bash
$ sh test.sh 1 2 3
Shell 传递参数实例！
执行的文件名：./test.sh
第一个参数为：1
第二个参数为：2
第三个参数为：3

```

## 三、shell数组

Bash Shell 只支持一维数组（不支持多维数组），初始化时不需要定义数组大小（与 PHP 类似）。

- 创建数组

格式：

```bash
array_name=(value1 value2 ... valuen)
```

举例：

```bash
test_array=(this is shell tutorial)
```

- 读取数组

格式：

```bash
${array_name[index]}
```

举例：

```bash
#!/bin/bash

test_array=(this is shell tutorial)

echo "第一个元素为: ${test_array[0]}"
echo "第二个元素为: ${test_array[1]}"
echo "第三个元素为: ${test_array[2]}"
echo "第四个元素为: ${test_array[3]}"
```

输出结果：

```bash
root@73df419e7539:~# sh array.sh
第一个元素为: this
第二个元素为: is
第三个元素为: shell
第四个元素为: tutorial
```

- 获取数组中的所有元素

```bash
#!/bin/bash

test_array[0]=this
test_array[1]=is
test_array[2]=shell
test_array[3]=tutorial

echo "数组的元素为: ${test_array[*]}"
echo "数组的元素为: ${test_array[@]}"
```

- 获取数组的长度

```bash
#!/bin/bash

test_array[0]=this
test_array[1]=is
test_array[2]=shell
test_array[3]=tutorial

echo "数组的元素为: ${#test_array[*]}"
echo "数组的元素为: ${#test_array[@]}"
```

- 用for 循环遍历数组

```bash
#!/bin/bash
  arr=(1 2 3 4 5 6 7 8 9 10)
  for a in ${arr[*]}
  do
    echo $a
  done
```

- 用while循环输出数组

```bash
arr=(1 2 3 4 5 6 7 8 9 10)
i=0
while [ $i -lt ${#arr[@]} ]
do
  echo ${arr[$i]}
  let i++
done
```

## 四、shell运算符

Shell 和其他编程语言一样，支持多种运算符，包括：

- 算数运算符
- 关系运算符
- 布尔运算符
- 字符串运算符
- 文件测试运算符

例如两数相加：

```bash
#!/bin/bash

val=`expr 2 + 2` 
val_1=$(expr 10 + 20)
echo "两数之和为 : $val"
echo "两数之和为 : $val_1"
```

> 两点注意：
>
> - 表达式和运算符之间要有空格，例如 2+2 是不对的，必须写成 2 + 2。
> - 完整的表达式要被 *``* 包含，注意这个字符不是常用的单引号，在 Esc 键下边。




-  ### 算术运算符

  下表列出了常用的算术运算符，假定变量 a 为 10，变量 b 为 20：

  | 运算符 | 说明                                          | 举例                          |
  | :----- | :-------------------------------------------- | :---------------------------- |
  | +      | 加法                                          | `expr $a + $b` 结果为 30。    |
  | -      | 减法                                          | `expr $a - $b` 结果为 -10。   |
  | *      | 乘法                                          | `expr $a \* $b` 结果为  200。 |
  | /      | 除法                                          | `expr $b / $a` 结果为 2。     |
  | %      | 取余                                          | `expr $b % $a` 结果为 0。     |
  | =      | 赋值                                          | a=$b 将把变量 b 的值赋给 a。  |
  | ==     | 相等。用于比较两个数字，相同则返回 true。     | [ $a == $b ] 返回 false。     |
  | !=     | 不相等。用于比较两个数字，不相同则返回 true。 | [ $a != $b ] 返回 true。      |

  **注意：**条件表达式要放在方括号之间，并且要有空格，例如: **[$a==$b]** 是错误的，必须写成 **[ $a == $b ]**。

```bash
#!/bin/bash

a=10
b=20

val=`expr $a + $b`
echo "a + b : $val"

val=`expr $a - $b`
echo "a - b : $val"

val=`expr $a \* $b`
echo "a * b : $val"

val=`expr $b / $a`
echo "b / a : $val"

val=`expr $b % $a`
echo "b % a : $val"

if [ $a == $b ]
then
   echo "a 等于 b"
fi
if [ $a != $b ]
then
   echo "a 不等于 b"
fi
```

输出结果：

```bash
a + b : 30
a - b : -10
a * b : 200
b / a : 2
b % a : 0
a 不等于 b
```



### 关系运算符

关系运算符只支持数字，不支持字符串，除非字符串的值是数字。

下表列出了常用的关系运算符，假定变量 a 为 10，变量 b 为 20：

| 运算符 | 说明                                                  | 举例                       |
| :----- | :---------------------------------------------------- | :------------------------- |
| -eq    | 检测两个数是否相等，相等返回 true。                   | [ $a -eq $b ] 返回 false。 |
| -ne    | 检测两个数是否不相等，不相等返回 true。               | [ $a -ne $b ] 返回 true。  |
| -gt    | 检测左边的数是否大于右边的，如果是，则返回 true。     | [ $a -gt $b ] 返回 false。 |
| -lt    | 检测左边的数是否小于右边的，如果是，则返回 true。     | [ $a -lt $b ] 返回 true。  |
| -ge    | 检测左边的数是否大于等于右边的，如果是，则返回 true。 | [ $a -ge $b ] 返回 false。 |
| -le    | 检测左边的数是否小于等于右边的，如果是，则返回 true。 | [ $a -le $b ] 返回 true。  |

> EQ 就是 EQUAL等于
>
> NE 就是 NOT EQUAL不等于 
>
> GT 就是 GREATER THAN大于　 
>
> LT 就是 LESS THAN小于 
>
> GE 就是 GREATER THAN OR EQUAL 大于等于 
>
> LE 就是 LESS THAN OR EQUAL 小于等于

示例：

```bash
#!/bin/bash

a=10
b=20

if [ $a -eq $b ]
then
   echo "$a -eq $b : a 等于 b"
else
   echo "$a -eq $b: a 不等于 b"
fi
if [ $a -ne $b ]
then
   echo "$a -ne $b: a 不等于 b"
else
   echo "$a -ne $b : a 等于 b"
fi
if [ $a -gt $b ]
then
   echo "$a -gt $b: a 大于 b"
else
   echo "$a -gt $b: a 不大于 b"
fi
if [ $a -lt $b ]
then
   echo "$a -lt $b: a 小于 b"
else
   echo "$a -lt $b: a 不小于 b"
fi
if [ $a -ge $b ]
then
   echo "$a -ge $b: a 大于或等于 b"
else
   echo "$a -ge $b: a 小于 b"
fi
if [ $a -le $b ]
then
   echo "$a -le $b: a 小于或等于 b"
else
   echo "$a -le $b: a 大于 b"
fi
```



输出结果：

```bash
10 -eq 20: a 不等于 b
10 -ne 20: a 不等于 b
10 -gt 20: a 不大于 b
10 -lt 20: a 小于 b
10 -ge 20: a 小于 b
10 -le 20: a 小于或等于 b
```



### 布尔运算符

下表列出了常用的布尔运算符，假定变量 a 为 10，变量 b 为 20：

| 运算符 | 说明                                                | 举例                                     |
| :----- | :-------------------------------------------------- | :--------------------------------------- |
| !      | 非运算，表达式为 true 则返回 false，否则返回 true。 | [ ! false ] 返回 true。                  |
| -o     | 或运算，有一个表达式为 true 则返回 true。           | [ $a -lt 20 -o $b -gt 100 ] 返回 true。  |
| -a     | 与运算，两个表达式都为 true 才返回 true。           | [ $a -lt 20 -a $b -gt 100 ] 返回 false。 |

示例

```bash
#!/bin/bash

a=10
b=20

if [ $a != $b ]
then
   echo "$a != $b : a 不等于 b"
else
   echo "$a == $b: a 等于 b"
fi
if [ $a -lt 100 -a $b -gt 15 ]
then
   echo "$a 小于 100 且 $b 大于 15 : 返回 true"
else
   echo "$a 小于 100 且 $b 大于 15 : 返回 false"
fi
if [ $a -lt 100 -o $b -gt 100 ]
then
   echo "$a 小于 100 或 $b 大于 100 : 返回 true"
else
   echo "$a 小于 100 或 $b 大于 100 : 返回 false"
fi
if [ $a -lt 5 -o $b -gt 100 ]
then
   echo "$a 小于 5 或 $b 大于 100 : 返回 true"
else
   echo "$a 小于 5 或 $b 大于 100 : 返回 false"
fi

```
输出结果：

```bash
10 != 20 : a 不等于 b
10 小于 100 且 20 大于 15 : 返回 true
10 小于 100 或 20 大于 100 : 返回 true
10 小于 5 或 20 大于 100 : 返回 false
```



### 逻辑运算符

| 运算符 | 说明       | 举例                                       |
| :----- | :--------- | :----------------------------------------- |
| &&     | 逻辑的 AND | [[ $a -lt 100 && $b -gt 100 ]] 返回 false  |
| \|\|   | 逻辑的 OR  | [[ $a -lt 100 \|\| $b -gt 100 ]] 返回 true |

示例

```bash
#!/bin/bash

a=10
b=20

if [[ $a -lt 100 && $b -gt 100 ]]
then
   echo "返回 true"
else
   echo "返回 false"
fi

if [[ $a -lt 100 || $b -gt 100 ]]
then
   echo "返回 true"
else
   echo "返回 false"
fi
```



输出结果：

```bash
返回 false
返回 true
```



### 字符串运算符

下表列出了常用的字符串运算符，假定变量 a 为 "abc"，变量 b 为 "efg"：

| 运算符 | 说明                                         | 举例                     |
| :----- | :------------------------------------------- | :----------------------- |
| =      | 检测两个字符串是否相等，相等返回 true。      | [ $a = $b ] 返回 false。 |
| !=     | 检测两个字符串是否不相等，不相等返回 true。  | [ $a != $b ] 返回 true。 |
| -z     | 检测字符串长度是否为0，为0返回 true。        | [ -z $a ] 返回 false。   |
| -n     | 检测字符串长度是否不为 0，不为 0 返回 true。 | [ -n "$a" ] 返回 true。  |
| $      | 检测字符串是否为空，不为空返回 true。        | [ $a ] 返回 true。       |

示例：

```bash
#!/bin/bash

a="abc"
b="efg"

if [ $a = $b ]
then
   echo "$a = $b : a 等于 b"
else
   echo "$a = $b: a 不等于 b"
fi
if [ $a != $b ]
then
   echo "$a != $b : a 不等于 b"
else
   echo "$a != $b: a 等于 b"
fi
if [ -z $a ]
then
   echo "-z $a : 字符串长度为 0"
else
   echo "-z $a : 字符串长度不为 0"
fi
if [ -n "$a" ]
then
   echo "-n $a : 字符串长度不为 0"
else
   echo "-n $a : 字符串长度为 0"
fi
if [ $a ]
then
   echo "$a : 字符串不为空"
else
   echo "$a : 字符串为空"
fi
```

输出结果：

```bash
abc = efg: a 不等于 b
abc != efg : a 不等于 b
-z abc : 字符串长度不为 0
-n abc : 字符串长度不为 0
abc : 字符串不为空
```

## 五、Shell test命令

Shell中的 test 命令用于检查某个条件是否成立，它可以进行数值、字符和文件三个方面的测试。

### 数值测试

| 参数 | 说明           |
| :--- | :------------- |
| -eq  | 等于则为真     |
| -ne  | 不等于则为真   |
| -gt  | 大于则为真     |
| -ge  | 大于等于则为真 |
| -lt  | 小于则为真     |
| -le  | 小于等于则为真 |

示例

```bash
num1=100
num2=100
if test $[num1] -eq $[num2]
then
    echo '两个数相等！'
else
    echo '两个数不相等！'
fi
```

输出结果：

```
两个数相等！
```

### 判断文件夹/文件是否存在

文件夹不存在则创建

```bash
if [ ! -d "/root/test" ];then
  mkdir /root/test
  else
  echo "文件夹已经存在"
fi
```

文件存在则删除

```bash
if [ ! -f "/root/test1.sh" ];then
  echo "文件不存在"
  else
  rm -f /root/test1.sh
fi
```

判断文件夹是否存在

```bash
if [ -d "/data/" ];then
  echo "文件夹存在"
  else
  echo "文件夹不存在"
fi
```

判断文件是否存在

```bash
if [ -f "/data/filename" ];then
  echo "文件存在"
  else
  echo "文件不存在"
fi
```

## 六、使用函数

定义函数

```bash
function foo ()
{
    echo "Arguments work just like script arguments: $@"
    echo "And: $1 $2..."
    echo "This is a function"
    return 0
}

```



更简单的方法：

```bash
# 更简单的方法
bar ()
{
    echo "Another way to declare functions!"
    return 0
}
```

调用函数

```bash
foo "My name is" $Name
```


