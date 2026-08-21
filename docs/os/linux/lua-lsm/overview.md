# 源码阅读

## lsm.c

首先看最底部

```c
DEFINE_LSM(lua) = {
	.id = &lua_lsmid,
	.enabled = &lua_lsm_enabled,
	.blobs = &lua_lsm_blob_sizes, /* 给一些内核对象多加一个字段，给 Lua-LSM 存放对象专属数据用，叫 security blob */
	.init = lua_lsm_init,
	.initcall_fs = lua_lsm_securityfs_init,
};
```

这是 LSM 的注册信息。往上看，就是 `lua_lsm_init`. 出于简化需要，统计计数相关的代码就省略了。

```c
struct lvm_state {
	lua_State *L;
	atomic_t refcount;
	struct lvm_state *next;
	bool dirty;
};

static int __init lua_lsm_init(void)
{
	for_each_possible_cpu(cpu)
		lvm_pool_init_cpu(cpu); /* 初始化每个 CPU 的 lvm(lua vm) pool */

	err = task_blob_init(current); /* 初始化当前 task 自带的 security blob */
	if (err)
		return err;

	for_each_possible_cpu(cpu) {
		lvm = kzalloc(sizeof(struct lvm_state), GFP_KERNEL);
		if (!lvm)
			return -ENOMEM;

		err = lua_state_alloc(lvm); /* 最终调用 lvm_build_lua_state */
		if (err)
			return err;

		/* 把刚初始化好的 lvm 指针放进 per-CPU 变量 irq_lvms 中。之后可以按 CPU 获取 */
		per_cpu(irq_lvms, cpu) = lvm;
	}

	/* Register only the hooks that Lua-LSM exposes to modules. */
	for (i = 0; i < ARRAY_SIZE(lua_lsm_hooks); i++) {
		if (!lua_lsm_hook_supported(i))
			continue;
		/* 注册 LSM hook */
		security_add_hooks(&lua_lsm_hooks[i], 1, &lua_lsmid);
	}

	/* Report that Lua-LSM successfully initialized */
	lua_lsm_initialized = 1;

	pr_info("Lua based LSM initialized (lvm_pool_max=%u)\n",
		READ_ONCE(lua_lvm_pool_max));
	return 0;
}
```

LVM (Lua VM) 就是 `lua_State`, Lua VM 的上下文，包含 Lua 栈和调用链等。

如果是中断处理（softirq）时进入 Lua-LSM hook, 那就使用 per-CPU 的 LVM, 其余情况下使用当前 task 的 LVM，从 lvm pool 中获取。

`lua_State` 是如何创建的呢？

```c
/*
 * Build a fully-initialized lua_State without publishing it into any
 * lvm_state. The returned state has openlibs, lualibs, shdict, and the
 * _MODULES table installed; the caller is responsible for installing it.
 */
static lua_State *lvm_build_lua_state(struct lvm_state *lvm)
{
	lua_State *L;
	int status;

	L = lua_newstate(lvm_alloc, lvm);
	if (!L)
		return NULL;

	lua_atpanic(L, lvm_panic);
	lua_gc(L, LUA_GCSTOP, 0);

	lua_pushcfunction(L, lvm_pmain);
	status = lua_pcall(L, 0, 1, 0);
	if (status != 0) {
		__log_err("pcall: status = %d, top = %d, %s\n",
			  status, lua_gettop(L), lua_tostring(L, -1));
		lua_close(L);
		return NULL;
	}
	if (!lua_toboolean(L, -1) && lua_gettop(L) != 1) {
		__log_err("lvm_pmain: top = %d, stack[top] = [%s]\n",
			  lua_gettop(L), luaL_typename(L, -1));
		lua_close(L);
		return NULL;
	}
	lua_pop(L, 1);

	lvm_stats_vmalloc();
	return L;
}
```

Side note: `lua_newstate` 会分配 `lua_State` + `global_State` 的内存空间。

现在来看初始化 `lua_State` 的函数 `lvm_pmain`:

```c
static int lvm_pmain(lua_State *L)
{
	luaL_openlibs(L);

	/* open builtin libraries */
	lualibs_openall(L);

	/* shared dict init */
	shdict_init(L);

	lua_gc(L, LUA_GCRESTART, 0);

	/* _G._G = nil, remove global variable _G */
	lua_pushnil(L);
	lua_setfield(L, LUA_GLOBALSINDEX, "_G");

	lua_pushcfunction(L, ll_require);
	lua_setglobal(L, "require");

	/* build _MODULES table with metatable */
	lua_newtable(L);			/* _MODULES table */
	lua_createtable(L, 0, 1);		/* metatable */
	lua_pushcfunction(L, lua_modules_index); /* TODO: pass it as args */
	lua_setfield(L, -2, "__index");		/* metatable.__index = func */
	/* setmetatable(_MODULES, metatable) */
	lua_setmetatable(L, -2);
	lua_setfield(L, LUA_REGISTRYINDEX, "_MODULES");

	/* return true */
	lua_pushboolean(L, 1);
	return 1;
}
```