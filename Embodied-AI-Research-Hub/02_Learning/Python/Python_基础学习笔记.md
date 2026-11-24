---
title: Python 基础学习笔记
type: course-note
source: 926150.docx
source_sha256: 05c1e7311cbc24a274dbb2fae12bf405b5e30ffae9145bb6f0ffc6286536d1e0
source_paragraphs: P0050-P0414; P0468-P0471
original_date: 无法从源文档确定
status: 学习中，待自测
updated: 2026-11-06
---

# Python 基础学习笔记

> **来源**：`D:\Desktop\926150.docx`（SHA256 `05c1e731…6d1e0`），段落 P0050–P0414，另有 P0468–P0471 的 parameter/argument 补记。  
> **原始日期**：源文档在数学部分有 `9.26.2026` 与 `10.6.2026`，但 Python 段没有独立日期；不能把文档保存时间 2026-10-21 当作学习日期。  
> **性质**：这是课程摘录 + 代码片段的整理，**不是独立项目记录**，也不能证明已经掌握。  
> **原文**：[`926150_原始笔记逐段提取.md`](../../90_Archive/Original_Documents/926150_原始笔记逐段提取.md)

## 阅读图例

- **【原始笔记】**：源文档中的内容；保留原意，部分代码只修复了被 Word/WPS 丢失的换行。
- **【整理重排】**：标题、分组、代码块语言标识。
- **【AI解释】**：为理解补的说明，不是用户当时写的。
- **【待核实】**：源文档不足以确认，或示例输出可能不完整。
- **【待纠正】**：源文档中明确不准确/不完整的地方；原文保留，更正单列。

## 0. 这份笔记覆盖什么

字符串与常用方法 → 数字运算 → 条件/布尔 → 函数/作用域 → 列表 → 元组 → 循环 → 推导式/filter/map/lambda → 字典 → 集合 → 模块/import → 正则 → `is`/真值 → `__main__`。

**不在这里**：Jupyter 快捷键（在吴恩达 ML 笔记）、NumPy/Matplotlib（单独笔记）。  
**源文档后部的 P0468–P0471** 是 parameter/argument 补记，并入本笔记“函数”一节。

## 1. 字符串：不可变序列

【原始笔记】源段落 P0050–P0097。

### 1.1 引号、转义、成员判断

```python
msg = "It's a sunny day"
msg = 'It\'s a sunny day'
```

- `in` 操作符返回布尔值，表示某个字符/子串是否存在于字符串中。
- `len()` 获取字符串长度；索引从 0 开始：`my_str[0]`。
- 字符串**不可变**，不能直接改某个位置：

```python
greeting = 'hi'
greeting[0] = 'H'
# TypeError: 'str' object does not support item assignment
```

### 1.2 拼接、重复、插值、切片

```python
first = 'Hello'
second = 'world'
sentence = first + ' ' + second   # 字符串拼接
echo = 'ha' * 3                   # 'hahaha'

name = 'John Doe'
age = 26
name_and_age = f'My name is {name} and I am {age} years old'
print(name_and_age)  # My name is John Doe and I am 26 years old
```

> 【整理重排】原文 P0065 把赋值和 `print` 挤在同一行（Word 换行丢失）；这里只拆行，语义不变。

```python
my_str = 'Hello world'
print(my_str[1:4])   # ell
print(my_str[::-1])  # dlrow olleH
```

【AI解释】切片 `[start:stop:step]` 含 `start`、不含 `stop`；`[::-1]` 表示从头到尾、步长 -1，因此反转。源笔记已写“切片/反转”，这里补的是索引边界说明。

### 1.3 常用字符串方法

| 方法 | 源笔记中的意思 | 最小示例 |
|---|---|---|
| `upper()` | 返回全大写的新字符串 | `'hi'.upper()` → `'HI'` |
| `lower()` | 返回全小写的新字符串 | `'HI'.lower()` → `'hi'` |
| `strip()` | 去掉前导/尾随指定字符；无参数去空白 | `'  hi  '.strip()` → `'hi'` |
| `replace(old,new)` | 把所有 `old` 换成 `new` | `'hi hi'.replace('hi','yo')` → `'yo yo'` |
| `split(sep)` | 按分隔符拆成列表；无参数按空白 | `'a b'.split()` → `['a','b']` |
| `join(iterable)` | 用分隔符把字符串集合连接 | `' '.join(['a','b'])` → `'a b'` |
| `startswith(prefix)` | 是否以某前缀开头 | `'abc'.startswith('ab')` → `True` |
| `endswith(suffix)` | 是否以某后缀结尾 | `'abc'.endswith('bc')` → `True` |
| `find(sub)` | 第一次出现的索引；找不到返回 -1 | `'abc'.find('b')` → `1` |
| `count(sub)` | 不重叠子串出现次数 | `'aaa'.count('aa')` → `1` |
| `capitalize()` | 首字符大写，其余小写 | `'hELLO'.capitalize()` → `'Hello'` |
| `isupper()` / `islower()` | 至少一个有大小写字符，且全部大写/小写 | `'ABC'.isupper()` → `True` |
| `title()` | 每个单词首字母大写 | `'hello world'.title()` → `'Hello World'` |

```python
my_str = '  hello world  '
trimmed_my_str = my_str.strip()
print(trimmed_my_str)  # hello world

my_str = 'hello world'
split_words = my_str.split()
print(split_words)  # ['hello', 'world']

my_list = ['hello', 'world']
joined_my_str = ' '.join(my_list)
print(joined_my_str)  # hello world
```

### 1.4 `maketrans` / `translate`

【原始笔记】P0121–P0125：

- `str.maketrans()` 接受两个等长字符串，返回翻译表；源笔记说“每个字符以 Unicode 序号保存”。
- `translate()` 用翻译表替换字符，返回新字符串。

```python
t = str.maketrans('lk', 'br')
sentence = 'The tent gave in to the leaks.'
print(sentence.translate(t))
# The tent gave in to the bears.
```

### 1.5 REPL

【原始笔记】P0126：Python 解释器遵循 read–eval–print loop（REPL）：读取、求值、打印，然后回到 `>>>`。

> 【待核实】源笔记的 `translate` 例子输出以实际运行为准；`maketrans` 的“Unicode 序号”表述可理解为内部映射，不影响使用。

## 2. 数字与运算

【原始笔记】P0098–P0105。

- 地板除法 `//`：相除后取小于或等于结果的最大整数。
- 取模 `%`：返回余数。
- 指数 `**`。
- `round(x, n)`：四舍五入到 n 位小数；恰好居中时通常取偶数（banker's rounding），例如 `round(2.5)` 是 2，`round(3.5)` 是 4；浮点表示会影响小数结果。
- `abs(x)`：绝对值。
- `pow(a,b)`：幂；`pow(a,b,m)`：模幂。

```python
result_2 = pow(2, 3, 5)  # (2 ** 3) % 5
print(result_2)          # 3
```

【AI解释】`round` 的“取偶数”不是 Python 独有，但浮点数如 `2.675` 可能因二进制表示得不到预期的 `2.68`；涉及金额/评测阈值时不要依赖默认 round。

## 3. 条件、布尔与输入

【原始笔记】P0106–P0115。

```python
if condition:
    pass  # Code to execute if condition is True
```

- `and`：第一个操作数为 falsy 就返回它，否则返回第二个；两个都 truthy 才 truthy。
- `or`：第一个 truthy 就返回它，否则返回第二个。
- `not`：反转布尔值。
- `input()`：提示用户输入。

```python
name = input('What is your name?')
print('Hello', name)
# 用户输入 Kolade 时输出 Hello Kolade
```

> 【整理重排】原文 P0115 把 `input` 和 `print` 挤在一行；这里拆开。  
> 【AI解释】`input()` 返回字符串；要参与数值计算需显式 `int()`/`float()`，并处理非法输入。源笔记没有展开这一点。

## 4. 函数、作用域、参数

【原始笔记】P0116–P0120；P0468–P0471。

```python
def calculate_sum(a, b):
    print(a + b)
```

- 函数外创建的变量是全局作用域；函数内创建的是局部作用域；函数参数也是局部变量。
- 源笔记定义了 `calculate_sum` 但只 `print`，没有 `return`。

> 【AI解释】`print` 是把值显示出来；`return` 才是把值交给调用者。一个只 print 的函数返回 `None`。这是旧 V3 自测题之一，必须能区分。

### parameter 与 argument（源文档后部补记）

【原始笔记】P0468–P0471：

- **parameter（形参）**：函数定义时括号里的变量名。
- **argument（实参）**：调用函数时传入的具体数值/变量。

```python
def func(x, w, b):   # x, w, b 是 parameters（形参）
    ...

# x_train, 100, 100 是 arguments（实参）的示例
```

> 【待核实】P0471 的原文在“是”后中断，无法确定完整例子；不要补造成用户当时写过的内容。

## 5. 列表：可变有序序列

【原始笔记】P0127–P0205。

### 5.1 创建、索引、修改

```python
cities = ['Los Angeles', 'London', 'Tokyo']
developer = 'Jessica'
list(developer)  # ['J', 'e', 's', 's', 'i', 'c', 'a']
```

> 【整理重排】原文 P0131 把 `developer = 'Jessica'` 和 `list(developer)` 挤在同一行；这里拆开。

- 列表由字符串、数字甚至其他列表组成；有序、可变、零基索引。
- 可更新元素；`del` 删除；`in` 检查成员；可嵌套；可解包。

```python
programming_languages[0] = 'JavaScript'

developer = ['Jane Doe', 23, 'Python Developer']
del developer[1]
print(developer)  # ['Jane Doe', 'Python Developer']

developer = ['Alice', 25, ['Python', 'Rust', 'C++']]
developer[2][1]  # 'Rust'

developer = ['Alice', 34, 'Rust Developer']
name, age, job = developer

name, *rest = developer
print(name)  # 'Alice'
print(rest)  # [34, 'Rust Developer']
```

### 5.2 切片

```python
numbers = [1, 2, 3, 4, 5, 6]
numbers[1::2]  # [2, 4, 6]
```

### 5.3 增删改查

```python
numbers = [1, 2, 3, 4, 5]
numbers.append(6)
# [1, 2, 3, 4, 5, 6]

numbers = [1, 2, 3, 4, 5]
even_numbers = [6, 8, 10]
numbers.append(even_numbers)
# [1, 2, 3, 4, 5, [6, 8, 10]]

numbers = [1, 2, 3, 4, 5]
even_numbers = [6, 8, 10]
numbers.extend(even_numbers)
# [1, 2, 3, 4, 5, 6, 8, 10]

numbers = [1, 2, 3, 4, 5]
numbers.insert(2, 2.5)
# [1, 2, 2.5, 3, 4, 5]

numbers = [10, 20, 30, 40, 50, 50]
numbers.remove(50)
# [10, 20, 30, 40, 50]  # 只移除第一处

numbers = [1, 2, 3, 4, 5]
numbers.pop(1)  # 返回 2；不传参数移除最后一个
numbers.clear() # []
```

### 5.4 排序、反转、查找

```python
numbers = [19, 2, 35, 1, 67, 41]
numbers.sort()
print(numbers)  # [1, 2, 19, 35, 41, 67]

numbers = [19, 2, 35, 1, 67, 41]
sorted_numbers = sorted(numbers)
print(numbers)        # 原列表不变
print(sorted_numbers) # [1, 2, 19, 35, 41, 67]

numbers = [6, 5, 4, 3, 2, 1]
numbers.reverse()
print(numbers)  # [1, 2, 3, 4, 5, 6]

programming_languages = ['Rust', 'Java', 'Python', 'C++']
programming_languages.index('Java')  # 1
```

【AI解释】`list.sort()` 就地排序，返回值是 `None`；`sorted()` 返回新列表。`list.reverse()` 就地反转，也返回 `None`。源笔记没有明确“返回 None”，但这是常见错误点。

> 【待核实】源笔记未覆盖列表浅拷贝/深拷贝。旧 V3 的 20 题里有“列表复制后修改元素是否影响原列表”，需另做自测；不要凭本段推断已掌握。

## 6. 元组：不可变有序序列

【原始笔记】P0206–P0236。

```python
developer = 'Jessica'
tuple(developer)  # ('J', 'e', 's', 's', 'i', 'c', 'a')

desserts = ('cake', 'pie', 'cookies', 'ice cream')
desserts[1:3]  # ('pie', 'cookies')
```

- 元组创建后元素不能更改；不能直接“移除一项”。
- `in`、解包、`*rest`、切片均可用。
- `count()` 统计出现次数；`index()` 查索引，找不到 `ValueError`。
- `index(x, start)` / `index(x, start, end)` 可指定范围。

```python
programming_languages = ('Rust', 'Java', 'Python', 'C++', 'Rust', 'Python')
programming_languages.index('Python', 3)     # 5
programming_languages.index('Python', 2, 5)  # 2

numbers = (13, 2, 78, 3, 45, 67, 18, 7)
sorted(numbers)  # [2, 3, 7, 13, 18, 45, 67, 78]
# sorted() 总是返回新列表，不返回元组
```

```python
programming_languages = ('Rust', 'Java', 'Python', 'C++', 'Rust', 'Python')
sorted(programming_languages, key=len)
# ['C++', 'Rust', 'Java', 'Rust', 'Python', 'Python']

print(sorted(programming_languages, reverse=True))
# ['Rust', 'Rust', 'Python', 'Python', 'Java', 'C++']
```

> 【整理重排】原文 P0230 把赋值和 `sorted(...)` 挤在一行；这里拆开。  
> 【待核实】`key=len` 的稳定排序在等长元素间保持原顺序；示例输出以实际 Python 版本为准。

## 7. 循环：for、while、range、enumerate、zip

【原始笔记】P0237–P0285。

```python
programming_languages = ['Rust', 'Java', 'Python', 'C++']
for language in programming_languages:
    print(language)
```

```python
categories = ['Fruit', 'Vegetable']
foods = ['Apple', 'Carrot', 'Banana']
for category in categories:
    for food in foods:
        print(category, food)
# Fruit Apple
# Fruit Carrot
# Fruit Banana
# Vegetable Apple
# Vegetable Carrot
# Vegetable Banana
```

```python
secret_number = 3
guess = 0
while guess != secret_number:
    guess = int(input('Guess the number (1-5): '))
    if guess != secret_number:
        print('Wrong! Try again.')
print('You got it!')
```

- `break` 停止整个循环；`continue` 跳过当前迭代，进入下一次。
- `range(start, stop, step)` 生成整数序列；`range(40, 0, -10)` 得到 40, 30, 20, 10。
- `enumerate()` 同时给索引和值；`start` 可指定起始计数。
- `zip()` 把多个可迭代对象按位置配对，返回元组迭代器。

```python
languages = ['Spanish', 'English', 'Russian', 'Chinese']
list(enumerate(languages))
# [(0, 'Spanish'), (1, 'English'), (2, 'Russian'), (3, 'Chinese')]

for index, language in enumerate(languages, 1):
    print(f'Index {index} and language {language}')

developers = ['Naomi', 'Dario', 'Jessica', 'Tom']
ids = [1, 2, 3, 4]
list(zip(developers, ids))
# [('Naomi', 1), ('Dario', 2), ('Jessica', 3), ('Tom', 4)]
```

> 【待核实】源笔记 P0262 说“用 continue 代替 break”，但没有给出对应代码；不能据此推断 `continue` 会终止循环。它只跳过当前迭代。

## 8. 推导式、filter、map、sum、lambda

【原始笔记】P0286–P0311。

```python
even_numbers = [num for num in range(21) if num % 2 == 0]
print(even_numbers)

numbers = [1, 2, 3, 4, 5]
result = [(num, 'Even') if num % 2 == 0 else (num, 'Odd') for num in numbers]
print(result)
# [(1, 'Odd'), (2, 'Even'), (3, 'Odd'), (4, 'Even'), (5, 'Odd')]
```

> 【整理重排】原文 P0288、P0290 把 `print` 与赋值挤在同一行；这里拆开。

```python
words = ['tree', 'sky', 'mountain', 'river', 'cloud', 'sun']
def is_long_word(word):
    return len(word) > 4
long_words = list(filter(is_long_word, words))
print(long_words)  # ['mountain', 'river', 'cloud']

celsius = [0, 10, 20, 30, 40]
def to_fahrenheit(temp):
    return (temp * 9/5) + 32
fahrenheit = list(map(to_fahrenheit, celsius))
print(fahrenheit)  # [32.0, 50.0, 68.0, 86.0, 104.0]

numbers = [5, 10, 15, 20]
total = sum(numbers)
print(total)          # 50
print(sum(numbers, 10))  # 60

lambda num: num ** 2
```

【AI解释】`filter`/`map` 返回的是惰性迭代器，源笔记用 `list(...)` 物化；`lambda` 是匿名函数，适合很短的回调，不要为了短而牺牲可读性。

## 9. 字典：键值对

【原始笔记】P0312–P0341。

```python
pizza = {
    'name': 'Margherita Pizza',
    'price': 8.9,
    'calories_per_slice': 250,
    'toppings': ['mozzarella', 'basil']
}

pizza = dict([
    ('name', 'Margherita Pizza'),
    ('price', 8.9),
    ('calories_per_slice', 250),
    ('toppings', ['mozzarella', 'basil'])
])

pizza['name']              # 'Margherita Pizza'
pizza.get('toppings', [])  # ['mozzarella', 'basil']
pizza.keys()
pizza.values()
pizza.items()
```

- `clear()` 移除全部键值对。
- `pop(key, default)` 移除并返回；无默认且键不存在 → `KeyError`。
- `popitem()` 移除最后插入的项。
- `update(other)` 用另一个字典更新。
- `**` 可解包字典为关键字参数。

```python
pizza.pop('price', 10)
pizza.pop('total_price')  # KeyError

pizza.update({'price': 15, 'total_time': 25})

def sum(a, b, c):
    return a + b + c

nums = {'a': 2, 'b': 4, 'c': 1}
print(sum(**nums))  # 7
```

> 【待核实】源笔记 P0322 的 `pizza.keys()` 注释只列了 `name/price/calories_per_slice`，与完整字典（还含 `toppings`）不一致；应以实际运行输出为准。  
> 【AI解释】自定义 `sum` 会遮蔽内置 `sum`；示例只为演示 `**` 解包，真实代码应避免这种命名。

## 10. 集合：去重、无序、集合运算

【原始笔记】P0342–P0355。

```python
my_set = {1, 2, 3, 4, 5}
set()   # 空集合
{}      # 空字典，不是空集合
```

- 集合不保存重复值；可变、无序；元素必须是不可变类型（数字、字符串、元组等）。
- 支持并集 `|`、交集 `&`、差集 `-`、对称差 `^`。
- `add()`、`remove()`/`discard()`、`clear()`、`issubset()`、`issuperset()`、`isdisjoint()`。

> 【待纠正】源笔记 P0346 的排版 `set() # Set{}    # Dictionary` 容易误读。正确关系是：`set()` 创建空集合；`{}` 创建空字典。原文保留，此处单列澄清。  
> 【AI解释】`remove` 删除不存在元素会 `KeyError`，`discard` 不会；源笔记未展开，使用时查文档。

## 11. 模块与 import

【原始笔记】P0356–P0380。

```python
import module_name
module_name.function_name()

import module_name as module_alias
from module_name import name1, name2

from math import radians, sin, cos
angle_degrees = 40
angle_radians = radians(angle_degrees)
sine_value = sin(angle_radians)
cos_value = cos(angle_radians)
print(sine_value)  # 0.6427876096865393
print(cos_value)   # 0.766044443118978
```

```python
from module_name import *
from math import *
print(sqrt(36))  # 6.0
print(pow(5, 2)) # 25.0
print(exp(1))    # 2.718281828459045
```

- `import` 通常写在文件顶部；`as` 起别名；`from ... import ...` 只取需要的名字。
- `from ... import *` 会把名字全部导入，易污染命名空间；源笔记只作语法介绍。

### `if __name__ == '__main__'`

【原始笔记】P0379–P0411。

```python
if __name__ == '__main__':
    pass  # Code
```

> 【整理重排】原文只有 `# Code`，没有可执行语句；这里补 `pass` 使代码块语法完整，不改变原意。

- 直接运行文件时，`__name__` 是 `"__main__"`。
- 作为模块被导入时，`__name__` 是模块名（通常不含 `.py`）。
- 因此该条件块只在“作为主程序运行”时执行；被导入时不执行。

## 12. 正则表达式

【原始笔记】P0381–P0397。

- 正则（regex）是描述字符串规则的小型语言，用于查找、匹配、筛选、替换。
- `re.search(pattern, string)` 找到第一个匹配返回 match 对象，否则 `None`。
- `\d` 匹配一个十进制数字。
- 原始字符串 `r'\d'` 让反斜杠保持字面意义，适合正则。
- `+` 匹配前一个字符一次或多次。
- `re.fullmatch(pattern, string)` 要求整个字符串匹配。

```python
import re

greeting = "Hello there!"
print(re.search('Hi', greeting))     # None
print(re.search('Hello', greeting))  # <re.Match object; span=(0, 5), match='Hello'>

book = "Fahrenheit 451"
print(re.search(r'\d', book))
# <re.Match object; span=(11, 12), match='4'>

print(re.fullmatch(r'\d+', book))            # None
print(re.fullmatch(r'Fahrenheit \d+', book)) # match 'Fahrenheit 451'
```

> 【整理重排】原文 P0386 把 `greeting = ...` 和 `print` 挤在一行；这里拆开。  
> 【AI解释】正则很容易过度使用；解析 CSV/JSON 优先用标准库 `csv`/`json`，不要用正则硬拆嵌套结构。

## 13. `is`、`None`、`all`

【原始笔记】P0398–P0404。

```python
value = None
print(value is None)      # True
print(value is not None)  # False

truthy = [1, 2, 3]
print(all(truthy))  # True

falsy = [0, 1, 2, 3]
print(all(falsy))   # False
```

> 【整理重排】原文 P0398 只列了 `is` / `is not` 的用法短语；这里补成可运行的 `None` 示例，语义不变。

- `is` / `is not` 比较两个值是否引用同一个对象（同一性），不是值相等。
- `is None` / `is not None` 是检查 `None` 的惯用法。
- `all(iterable)` 所有元素 truthy 才返回 `True`。

【AI解释】判断数值/字符串相等用 `==`；`is` 适合 `None`、单例或明确的身份判断。不要用 `is` 比较小整数/字符串并依赖实现细节。

## 14. 用户当时的理解

【原始笔记】在 926150 的 Python 段中，**没有发现明确的第一人称理解、疑问或总结**；大部分是课程解释与代码摘录。  
因此本笔记不把任何教程句子标注为“用户自己的理解”。如果你当时确实有口头/其他笔记，请以后补录并注明日期。

## 15. 待解决问题与待核实

| 类型 | 内容 | 处理 |
|---|---|---|
| 待自测 | `print` 与 `return` 的区别；空列表求均值的接口 | 写小函数 + 测试，不先看答案 |
| 待自测 | 列表别名、浅拷贝、切片复制的区别 | 用 `id()` 和修改实验验证 |
| 待自测 | `is` 与 `==`；`set()` 与 `{}` | 口述 + 小代码 |
| 待核实 | 源文档多处代码行被 Word 合并（P0065、P0115、P0131、P0230、P0288、P0290、P0303、P0386 等） | 已做最小断行；原始文本见机械提取 |
| 待核实 | `pizza.keys()` 注释与完整字典不一致 | 实际运行确认 |
| 待核实 | P0471 的 argument 示例在“是”后中断 | 不补造；等原始上下文 |
| 待补 | 文件读写、异常处理、类/对象 | 源笔记有 import/函数，但没有完整程序；旧计划 D9–D13 是计划，不是完成 |

## 16. 自测（先不给答案）

1. 写一个函数，接收列表，返回 `(count, sum, mean)`；空列表时你选择抛异常还是返回 `None`？为什么？
2. 写一个函数，`print` 出结果但不 `return`；调用者得到什么？如何改？
3. `a = [1,2]; b = a; b.append(3)` 后 `a` 是什么？如果 `b = a[:]` 呢？
4. `x = [1,2]; y = x;` 与 `y = x.copy()` 的区别是什么？用 `id()` 验证。
5. `set()` 与 `{}` 分别是什么？如何创建空集合？
6. `list.sort()` 与 `sorted(list)` 的返回值/原列表变化有何不同？
7. `is` 与 `==` 各比较什么？什么时候必须用 `is None`？
8. 写一个 `for` 循环读列表并累加；再用 `sum()`/列表推导式重写。哪种更可读？
9. 用 `re.fullmatch(r'\d+', s)` 判断字符串是否全为数字；空字符串、负号、空格分别怎样？
10. 把一段 20 行的脚本拆成 3 个函数，写 3 个 `assert`；解释每个函数的输入/输出。

**通过标准**：不查完整答案，60–90 分钟完成 1、3、5、6、8；其余能口述。通过后才考虑把 Python 状态改为“已通过独立验收”。

## 17. 关联课程与知识点

- freeCodeCamp Python Basics（旧 V3 记录 145/578 步；只代表课程步骤，不代表科研路线进度）。
- 旧 V3 第 1–2 周：列表、字典、CSV、异常、函数组合。
- 旧 V3 第 6 周：Classes and Objects 基础（本笔记未覆盖；需另学）。
- CS231n Python/NumPy 教程（数组、广播；见 NumPy 笔记）。
- 后续：文件/异常/类 → PyTorch Dataset/Module；正则 → 日志解析（谨慎使用）。

## 18. 来源映射与完整性

| 主题 | 源段落 |
|---|---|
| 字符串 | P0050–P0097 |
| 数字 | P0098–P0105 |
| 条件/输入 | P0106–P0115 |
| 函数/作用域 | P0116–P0120；P0468–P0471 |
| 列表 | P0127–P0205 |
| 元组 | P0206–P0236 |
| 循环 | P0237–P0285 |
| 推导式/filter/map/lambda | P0286–P0311 |
| 字典 | P0312–P0341 |
| 集合 | P0342–P0355 |
| 模块/import | P0356–P0380 |
| 正则 | P0381–P0397 |
| `is`/真值 | P0398–P0404 |
| `__main__` | P0405–P0411 |

**完整性**：源文档 Python 段的有效概念均已覆盖；重复的示例只保留一份并标注整理。代码块均带 `python` 标识；被 Word 合并的行已拆开并标注。图片：本段无内嵌图。
