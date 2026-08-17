# Closure

Lua 的 Closure 有两种：

- LClosure: Lua Closure, Lua 源码编译出来的函数
- CClosure: C Closure, C 语言实现、注册给 Lua 的函数

```c
typedef union Closure {
  CClosure c;
  LClosure l;
} Closure;
```

## 1. 共用头部 `ClosureHeader`

用于辅助 gc 过程。

```c
#define ClosureHeader \
  CommonHeader; \
  lu_byte nupvalues; \
  GCObject *gclist
```

CommonHeader 见 gc 部分介绍。

### `nupvalues`

```c
lu_byte nupvalues;
```

记录闭包实际拥有多少个 upvalue。

因为结构末尾用了动态长度数组，所以必须单独保存数量。

### `gclist`

```c
GCObject *gclist;
```

垃圾回收器在标记、传播对象引用时，需要把闭包挂入 gray、grayagain 等内部工作链表。

它和 `CommonHeader` 里的 `next` 用途不同：

- `next`：对象长期所属的 GC 对象链
- `gclist`：GC 当前阶段的临时工作链

---

## 2. LClosure

> At compile time: when a function is compiled, it generates a **prototype** containing the virtual machine instructions for the function, its constant values (numbers, literal strings, etc.), and some debug information.
>
> At run time: whenever Lua executes a `function...end` expression, it creates a new _closure_. Each closure has a reference to its corresponding **prototype**, a reference to its **environment** (a table wherein it looks for global variables), and an array of references to **upvalues**, which are used to access outer local variables.
>
> Functions and Closures - The Implementation of Lua 5.0

由此可知，lua 的 closure 由三部分组成：

- Proto：函数源码编译出的 VM 机器码、常量表等信息
- Upvalues：函数内引用的外层函数临时变量
- Env：全局变量

在 Lua 5.1 中：`LClosure` 确实拥有一个独立的 `struct Table *env` 字段来专门存储当前函数所处的环境。在 Lua 5.2 及之后：Lua 移除了独立的 env 字段，取消了全局环境的概念，转而引入了 _ENV 机制。全局环境变成了该函数的第一个 Upvalue（即 `upvals[0]`）。

对 Lua 函数来说，closure 基本就是 `Proto + upvalues`；但 Lua 同时支持用 C 实现的函数，因此源码里有 `LClosure` 和 `CClosure` 两种闭包。

```c
typedef struct LClosure {
    ClosureHeader;
    struct Proto *p;
    UpVal *upvals[1];
} LClosure;
```

它表示 Lua 源码中的函数闭包。

例如：

```lua
local x = 10

local function f()
    return x
end
```

`f` 对应一个 `LClosure`：

```text
LClosure f
├── p          → 函数的 Proto
├── nupvalues  = 1
└── upvals[0]  → 保存或引用变量 x 的 UpVal
```

### Proto

```c
struct Proto *p;
```

`Proto` 是 Lua 函数编译后的静态描述，也可以理解为“函数原型”或“字节码模板”。

它通常包含：

- 字节码指令
- 常量表
- 嵌套函数的 Proto
- 局部变量调试信息
- 源码行号信息
- 参数数量
- 是否为可变参数函数
- 所需最大栈空间
- upvalue 描述信息

例如：

```lua
local function add(a, b)
    return a + b
end
```

它的 `Proto` 大致表达：

```text
参数数量：2
最大寄存器数：若干
常量表：可能为空
字节码：
    ADD
    RETURN
```

但是 `Proto` 不表示一次具体的闭包实例，因为它没有绑定本次捕获的外部变量。

### UpVal

为什么是 `UpVal *upvals[1];` 长度为 1 的指针数组？其实 `LClosure` 是个可变长结构体，实际 `upvals` 数组长度为 `nupvalues`。写成这样是为了兼容不同 C 标准和编译器。

Lua 中，一个函数引用的、定义在外层 lexical scope 中的局部变量，就是它的 upvalue。

例如：

```lua
function outer()
    local x = 42

    local function inner()
        return x
    end

    return inner
end
```

对 `inner` 来说：`x` 是它的 upvalue。

upvalue 有两种状态：

- open：`outer` 函数还没返回，`x` 还在 `outer` 的栈上，upvalue 直接指向栈上内存。
- closed：`outer` 函数返回后，`x` 的栈内存失效，失效前把值搬到 upvalue 自己的内部空间。

```c
/*
** Upvalues for Lua closures
*/
typedef struct UpVal {
  CommonHeader;
  union {
    TValue *p;  /* points to stack or to its own value */
    ptrdiff_t offset;  /* used while the stack is being reallocated */
  } v;
  union {
    struct {  /* (when open) */
      struct UpVal *next;  /* linked list */
      struct UpVal **previous;
    } open;
    TValue value;  /* the value (when closed) */
  } u;
} UpVal;
```

通常访问 upvalue 内的 TValue(Typed Value) 的方式是：

```c
LClosure *cl = ...;  // 当前闭包
int i = ...;       // upvalue 索引, 从指令中解码
TValue *upval = cl->upvals[i]->v.p;
```

`inner` 函数的 `Proto.upvalues[]` 是 `struct Upvaldesc []` 类型，记录了 upvalue 的编译期信息：

```c
/*
** Description of an upvalue for function prototypes
*/
typedef struct Upvaldesc {
  TString *name;  /* upvalue name (for debug information) */
  lu_byte instack;  /* whether it is in stack (register) */
  lu_byte idx;  /* index of upvalue (in stack or in outer function's list) */
  lu_byte kind;  /* kind of corresponding variable */
} Upvaldesc;
```

在运行时执行 `OP_CLOSURE` 指令实例化 `inner`，代码位于 `luaV_execute`：

```c
vmcase(OP_CLOSURE) {
  StkId ra = RA(i);
  Proto *p = cl->p->p[GETARG_Bx(i)];
  halfProtect(pushclosure(L, p, cl->upvals, base, ra));
  checkGC(L, ra + 1);
  vmbreak;
}
```

---

## 3. CClosure

```c
typedef struct CClosure {
    ClosureHeader;
    lua_CFunction f;
    TValue upvalue[1];
} CClosure;
```

它表示一个由 C 函数实现的 Lua 闭包。

### `lua_CFunction f`

```c
lua_CFunction f;
```

它是实际被调用的 C 函数指针。

`lua_CFunction` 通常定义为：

```c
typedef int (*lua_CFunction)(lua_State *L);
```

例如：

```c
static int add(lua_State *L) {
    lua_Integer a = luaL_checkinteger(L, 1);
    lua_Integer b = luaL_checkinteger(L, 2);

    lua_pushinteger(L, a + b);
    return 1;
}
```

把它注册进 Lua：

```c
lua_pushcfunction(L, add);
lua_setglobal(L, "add");
```

Lua 中就可以调用：

```lua
print(add(10, 20))
```

这时底层创建的是一个 `CClosure`：

```text
CClosure
├── f = add
├── nupvalues = 0
└── upvalue = 无
```

### C 函数也可以有 upvalue

例如：

```c
lua_pushinteger(L, 100);
lua_pushcclosure(L, add_base, 1);
```

这里创建一个带一个 upvalue 的 C closure：

```text
CClosure
├── f = add_base
├── nupvalues = 1
└── upvalue[0] = integer 100
```

C 函数中可以这样访问：

```c
static int add_base(lua_State *L) {
    lua_Integer base =
        lua_tointeger(L, lua_upvalueindex(1));

    lua_Integer value =
        luaL_checkinteger(L, 1);

    lua_pushinteger(L, base + value);
    return 1;
}
```

Lua 中：

```lua
add_base(23)  --> 123
```

所以 C closure 的概念也是：

```text
C 函数指针 + 捕获的值
```

也就是：

```text
f + upvalues
```

这与 Lua closure 的：

```text
Proto + upvalues
```

其实是对应关系。

### 为什么两种 upvalue 类型不同

这是最值得注意的差异：

```c
CClosure:
    TValue upvalue[1];

LClosure:
    UpVal *upvals[1];
```

C closure 直接存储 `TValue`，Lua closure 则存储 `UpVal *`。

#### CClosure 直接保存值

```c
TValue upvalue[1];
```

C closure 的 upvalue 创建时直接复制到闭包对象中：

```text
Lua stack 中的值
       │
       │ 创建 C closure 时复制
       ▼
CClosure.upvalue[0]
```

之后这个值就属于闭包对象。

它不需要保持与某个 Lua 局部变量栈槽的共享关系。

#### LClosure 保存 `UpVal *`

Lua 闭包捕获的是“变量”，不只是捕获当时的值。

考虑：

```lua
local x = 10

local function get()
    return x
end

local function set(v)
    x = v
end
```

`get` 和 `set` 必须共享同一个 `x`：

```text
get.upvals[0] ──┐
                ├──→ 同一个 UpVal → x
set.upvals[0] ──┘
```

执行：

```lua
set(20)
print(get())  --> 20
```

如果 `LClosure` 直接保存 `TValue`，`get` 和 `set` 就会各自拥有一份副本，无法共享变量。

因此中间需要一层 `UpVal`：

```text
LClosure
   │
   │ UpVal *
   ▼
UpVal
   │
   ├── open：指向 Lua 栈槽
   │
   └── closed：值保存在 UpVal 自己内部
```

这也是 CClosure 和 LClosure 结构不同的根本原因。
