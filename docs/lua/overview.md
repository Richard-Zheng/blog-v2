# Lua VM Overview

使用一段最简单的 Lua 代码来说明 Lua VM 的工作原理：

```lua
local a=10; local b=20; print(a+b)
```

## 调试技巧

写 gdb 脚本，这样不用每次都手动输入：

```gdb
set pagination off
set debuginfod enabled off

source lua_gdb.py

break lua_newstate
break preinit_thread
run -e "local a=10; local b=20; print(a+b)"
```

Python 脚本 `lua_gdb.py`：

```python
import gdb


OPNAMES = [
    "MOVE", "LOADI", "LOADF", "LOADK", "LOADKX", "LOADFALSE",
    "LFALSESKIP", "LOADTRUE", "LOADNIL", "GETUPVAL", "SETUPVAL",
    "GETTABUP", "GETTABLE", "GETI", "GETFIELD", "SETTABUP",
    "SETTABLE", "SETI", "SETFIELD", "NEWTABLE", "SELF", "ADDI",
    "ADDK", "SUBK", "MULK", "MODK", "POWK", "DIVK", "IDIVK",
    "BANDK", "BORK", "BXORK", "SHLI", "SHRI", "ADD", "SUB",
    "MUL", "MOD", "POW", "DIV", "IDIV", "BAND", "BOR", "BXOR",
    "SHL", "SHR", "MMBIN", "MMBINI", "MMBINK", "UNM", "BNOT",
    "NOT", "LEN", "CONCAT", "CLOSE", "TBC", "JMP", "EQ", "LT",
    "LE", "EQK", "EQI", "LTI", "LEI", "GTI", "GEI", "TEST",
    "TESTSET", "CALL", "TAILCALL", "RETURN", "RETURN0", "RETURN1",
    "FORLOOP", "FORPREP", "TFORPREP", "TFORCALL", "TFORLOOP",
    "SETLIST", "CLOSURE", "VARARG", "GETVARG", "ERRNNIL",
    "VARARGPREP", "EXTRAARG",
]

MODE_NAMES = ["ABC", "vABC", "ABx", "AsBx", "Ax", "sJ"]


def _field(expr):
    return gdb.parse_and_eval(expr)


def _decode_instruction(raw):
    op = raw & 0x7f
    a = (raw >> 7) & 0xff
    k = (raw >> 15) & 1
    b = (raw >> 16) & 0xff
    c = (raw >> 24) & 0xff
    vb = (raw >> 16) & 0x3f
    vc = (raw >> 22) & 0x3ff
    bx = (raw >> 15) & 0x1ffff
    ax = (raw >> 7) & 0x1ffffff
    mode = int(_field("luaP_opmodes")[op]) & 7
    name = OPNAMES[op] if op < len(OPNAMES) else "OP?"
    if mode == 0:
        args = f"A={a} B={b} C={c} k={k}"
    elif mode == 1:
        args = f"A={a} vB={vb} vC={vc} k={k}"
    elif mode == 2:
        args = f"A={a} Bx={bx}"
    elif mode == 3:
        args = f"A={a} sBx={bx - 65535}"
    elif mode == 4:
        args = f"Ax={ax}"
    else:
        args = f"sJ={ax - 16777215}"
    return name, MODE_NAMES[mode], args


def _tvalue_text(tv):
    tag = int(tv["tt_"])
    kind = tag & 0x0f
    variant = (tag >> 4) & 3
    value = tv["value_"]
    try:
        if kind == 0:
            return "nil"
        if kind == 1:
            return "true" if variant == 1 else "false"
        if kind == 2:
            return f"lightuserdata({value['p']})"
        if kind == 3:
            return str(value["n"] if variant == 1 else value["i"])
        if kind == 4:
            ts = value["gc"].cast(gdb.lookup_type("TString").pointer())
            return '"' + ts["contents"].string(errors="replace") + '"'
        if kind == 5:
            return f"table({value['gc']})"
        if kind == 6:
            names = {0: "LuaClosure", 1: "CFunction", 2: "CClosure"}
            ptr = value["f"] if variant == 1 else value["gc"]
            return f"{names.get(variant, 'function')}({ptr})"
        if kind == 7:
            return f"userdata({value['gc']})"
        if kind == 8:
            return f"thread({value['gc']})"
    except gdb.error:
        pass
    return f"tag=0x{tag:x}"


class LuaInsn(gdb.Command):
    def __init__(self):
        super().__init__("lua-insn", gdb.COMMAND_DATA)

    def invoke(self, arg, from_tty):
        try:
            raw = int(_field("i"))
            pc = _field("pc")
            code = _field("cl->p->code")
            index = int(pc - code) - 1
            name, mode, args = _decode_instruction(raw)
            gdb.write(f"bytecode pc={index + 1}  {name:<12} {args}  [{mode}]\n")
        except gdb.error as exc:
            gdb.write(f"lua-insn: stop inside luaV_execute after vmfetch ({exc})\n")


class LuaStack(gdb.Command):
    def __init__(self):
        super().__init__("lua-stack", gdb.COMMAND_DATA)

    def invoke(self, arg, from_tty):
        try:
            base = _field("base")
            size = int(_field("cl->p->maxstacksize"))
            top = _field("L->top.p")
            gdb.write(f"registers: base={base}, logical top={top}, max={size}\n")
            for index in range(size):
                slot = (base + index).dereference()["val"]
                marker = " <top" if base + index == top else ""
                gdb.write(f"  R[{index:<2}] {_tvalue_text(slot)}{marker}\n")
        except gdb.error as exc:
            gdb.write(f"lua-stack: no active Lua VM frame ({exc})\n")


class LuaFrames(gdb.Command):
    def __init__(self):
        super().__init__("lua-frames", gdb.COMMAND_DATA)

    def invoke(self, arg, from_tty):
        try:
            ci = _field("L->ci")
            depth = 0
            while int(ci) != 0 and depth < 32:
                frame = ci.dereference()
                status = int(frame["callstatus"])
                flavor = "C" if status & (1 << 15) else "Lua"
                funcslot = frame["func"]["p"].dereference()["val"]
                gdb.write(
                    f"  #{depth} {flavor:<3} ci={ci} func={_tvalue_text(funcslot)}\n"
                )
                ci = frame["previous"]
                depth += 1
        except gdb.error as exc:
            gdb.write(f"lua-frames: cannot inspect frames ({exc})\n")


class LuaState(gdb.Command):
    def __init__(self):
        super().__init__("lua-state", gdb.COMMAND_DATA)

    def invoke(self, arg, from_tty):
        gdb.execute("lua-insn")
        gdb.execute("lua-frames")
        gdb.execute("lua-stack")


LuaInsn()
LuaStack()
LuaFrames()
LuaState()
gdb.write("Loaded Lua helpers: lua-insn, lua-stack, lua-frames, lua-state\n")
```

编译命令：

```bash
make clean
make linux MYCFLAGS='-O0 -g3 -fno-inline -fno-omit-frame-pointer -fno-optimize-sibling-calls -DLUA_USE_JUMPTABLE=0'
```

| 参数 | 作用 |
|---|---|
| `-O0` | 关闭 GCC 优化，源码行和执行顺序更直观 |
| `-g3` | 生成完整调试信息，包括宏信息 |
| `-fno-inline` | 尽量避免函数被内联 |
| `-fno-omit-frame-pointer` | 保留帧指针，让 C 调用栈更稳定 |
| `-fno-optimize-sibling-calls` | 避免 GCC 自己把尾调用优化掉 |
| `-DLUA_USE_JUMPTABLE=0` | 让 VM 使用普通 `switch` 分派 |

## 初始化

从 lua.c 的 `main` 开始：

```c
int main (int argc, char **argv) {
  int status, result;
  lua_State *L = luaL_newstate();  /* create state */
  if (L == NULL) {
    l_message(argv[0], "cannot create state: not enough memory");
    return EXIT_FAILURE;
  }
  lua_gc(L, LUA_GCSTOP);  /* stop GC while building state */
  lua_pushcfunction(L, &pmain);  /* to call 'pmain' in protected mode */
  lua_pushinteger(L, argc);  /* 1st argument */
  lua_pushlightuserdata(L, argv); /* 2nd argument */
  status = lua_pcall(L, 2, 1, 0);  /* do the call */
  result = lua_toboolean(L, -1);  /* get result */
  report(L, status);
  lua_close(L);
  return (result && status == LUA_OK) ? EXIT_SUCCESS : EXIT_FAILURE;
}
```

lua_State 是一条 Lua thread 的上下文，包含了执行 Lua 字节码所需的所有状态信息。几乎所有 Lua 函数第一个参数都是它。Lua 官方并不支持真正的多线程，但是可以在单个操作系统线程上协作式地并发执行多个 Lua thread.

`luaL_newstate` 是 `lua_newstate` 的包装函数，用于在没内存分配 state 的时候进行错误处理。(普通的错误处理依赖 state)

```c
LUA_API lua_State *lua_newstate (lua_Alloc f, void *ud, unsigned seed) {
  int i;
  lua_State *L;
  // 在堆上分配一个 global_State + lua_State 的内存块
  // struct global_State 包含一个 LX, 也即 lua_State + 前面 8 bytes 额外空间
  global_State *g = cast(global_State*,
                       (*f)(ud, NULL, LUA_TTHREAD, sizeof(global_State)));
  if (g == NULL) return NULL;
  L = &g->mainth.l;
  L->tt = LUA_VTHREAD;

  /* 设置一大堆字段的默认值... */

  if (luaD_rawrunprotected(L, f_luaopen, NULL) != LUA_OK) {
    /* memory allocation error: free partial state */
    close_state(L);
    L = NULL;
  }
  return L;
}
```

`luaD_rawrunprotected` 保护调用了函数 `f_luaopen`，保护调用是什么呢？就是在出现错误的时候可以统一跳转到一段代码进行错误处理。类似于 try-catch.

```c
TStatus luaD_rawrunprotected (lua_State *L, Pfunc f, void *ud) {
  l_uint32 oldnCcalls = L->nCcalls;
  lua_longjmp lj;
  lj.status = LUA_OK;
  // L->errorJmp 是一个链表，每次保护调用都会把本次上下文放在最前面
  lj.previous = L->errorJmp;  /* chain new error handler */
  L->errorJmp = &lj;
  LUAI_TRY(L, &lj, f, ud);  /* call 'f' catching errors */
  // 结束调用，弹出 lj，恢复上一个上下文
  L->errorJmp = lj.previous;  /* restore old error handler */
  L->nCcalls = oldnCcalls;
  return lj.status;
}
```

这里的关键就在 LUAI_TRY

```c
/* ISO C handling with long jumps */
#define LUAI_THROW(L,c)		longjmp((c)->b, 1)
#define LUAI_TRY(L,c,f,ud)	if (setjmp((c)->b) == 0) ((f)(L, ud))
```

这里用 setjmp 把当前的 C 上下文保存到了 `lua_longjmp lj;` 中，以 RISC-V 为例，包括：

- 被调用者保存寄存器（Call-preserved Registers）`s0` - `s11`
- 栈指针 `sp` 和帧指针 `fp`
- 返回地址 `ra`
- 最后，设置返回值为 0 并返回

这时会正常执行 `Pfunc f` 也即 `f_luaopen`，它会初始化 Lua VM 的各种状态。

```c
/*
** open parts of the state that may cause memory-allocation errors.
*/
static void f_luaopen (lua_State *L, void *ud) {
  global_State *g = G(L);  /* 取得当前 Lua 实例中所有线程共享的全局状态。 */
  UNUSED(ud);  /* 本次初始化不需要回调上下文，仅用于匹配 Pfunc 函数签名。 */
  stack_init(L, L);  /* 为主线程分配值栈，并初始化 base_ci 基础调用帧。 */
  init_registry(L, g);  /* 创建 Registry 表，并注册主线程和全局变量表。 */
  luaS_init(L);  /* 初始化全局字符串表，并创建内存错误消息等固定字符串。 */
  luaT_init(L);  /* 创建并固定 __index、__add 等元方法名称字符串。 */
  luaX_init(L);  /* 创建并固定词法分析器使用的 Lua 保留字字符串。 */
  g->gcstp = 0;  /* 清除 GC 停止标志，从现在开始允许垃圾回收运行。 */
  setnilvalue(&g->nilvalue);  /* 用 nil 标记状态已完整初始化，可进行正常清理。 */
  luai_userstateopen(L);  /* 调用用户可自定义的状态创建完成钩子，默认什么也不做。 */
}
```

如最上面的函数注释所说，此函数有可能触发内存分配错误（如内存不足），发生错误的时候会调用 `luaD_throw`

```c
l_noret luaD_throw (lua_State *L, TStatus errcode) {
  if (L->errorJmp) {  /* thread has an error handler? */
    L->errorJmp->status = errcode;  /* set status */
    LUAI_THROW(L, L->errorJmp);  /* 展开为 longjmp(L->errorJmp->b, 1) */
  } else { /* 省略 */ }
}
```

此时会恢复前面保存的上下文跳转回 `luaD_rawrunprotected`，不同的是 setjmp 返回值不再是 0，而是 1，表示发生了错误，返回 `errcode`。

这些初始化函数里比较值得拿出来说的是 `stack_init(L1, L)`, 它会 call `L`(的全局状态) 中的 `frealloc` 函数为 `L1->stack` 分配堆内存，失败时使用 `L->errorJmp` 进行错误处理，并初始化基础调用帧。

```c
static void stack_init (lua_State *L1, lua_State *L) {
  int i;
  /* initialize stack array */
  L1->stack.p = luaM_newvector(L, BASIC_STACK_SIZE + EXTRA_STACK, StackValue);
  L1->tbclist.p = L1->stack.p;
  for (i = 0; i < BASIC_STACK_SIZE + EXTRA_STACK; i++)
    setnilvalue2s(L1->stack.p + i);  /* erase new stack */
  L1->stack_last.p = L1->stack.p + BASIC_STACK_SIZE;
  /* initialize first ci */
  resetCI(L1);
  L1->top.p = L1->stack.p + 1;  /* +1 for 'function' entry */
}
```