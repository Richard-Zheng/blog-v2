# 从源码编译 Lua

为了调试 Lua 源码，建议使用以下命令编译：

```bash
make clean
make linux MYCFLAGS="-O0 -g3 -fno-omit-frame-pointer"
```

如果系统安装了 readline，也可以：

```bash
make clean
make linux-readline MYCFLAGS="-O0 -g3 -fno-omit-frame-pointer"
```
