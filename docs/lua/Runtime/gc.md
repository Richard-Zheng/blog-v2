# GC

lua 的 gc 负责自动回收不再使用的堆内存（？是吗）

## CommonHeader

CommonHeader 是所有可回收对象的公共头部，定义在 `lobject.h` 中：

```c
/*
** Common Header for all collectable objects (in macro form, to be
** included in other objects)
*/
#define CommonHeader	struct GCObject *next; lu_byte tt; lu_byte marked
```

展开来就是

```c
struct GCObject *next;
lu_byte tt;
lu_byte marked;
```

- `next`：连接到所有 GC 对象链表
- `tt`：说明这是 Lua closure 还是 C closure
- `marked`：GC 标记状态以及分代年龄

所有可回收对象开头都有这三个字段，因此都可以被视作 `GCObject` 对象。

```c
/* Common type for all collectable objects */
typedef struct GCObject {
  CommonHeader;
} GCObject;
```

lua VM 内部会维护一个超大的 `GCObject` 链表，所有当前存活的可回收对象都会被挂在这个链表上。

lua 的垃圾回收有两种算法：incremental mark-and-sweep 和 generational。前者是增量标记清除，后者是分代回收。
