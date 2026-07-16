# OpenSBI

[OpenSBI - GitHub](https://github.com/riscv-software-src/opensbi)

Clone the repository and add some args to Makefile,

```diff
diff --git a/Makefile b/Makefile
index 37793aaf..6224f186 100644
--- a/Makefile
+++ b/Makefile
@@ -1,3 +1,8 @@
+# From https://jyywiki.cn/OS/demos/intro/opensbi/Makefile
+export PLATFORM := generic
+export CROSS_COMPILE := riscv64-linux-gnu-
+export DEBUG := 1
+
 #
 # SPDX-License-Identifier: BSD-2-Clause
 #
```

then let make output the build command.

```bash
make -nB > a.log
```

Use sed to make it more readable (?). This trick also comes from jyy's OS course.

```bash
sed -i "s#$(PWD)#.#g" a.log
sed -i "s|mkdir.*echo|echo|g" a.log
sed -i "s/ /\r  /g" a.log
```

It uses the `fw_jump.elf.ld` linker script to build the final file.

```bash
echo " ELF       platform/generic/firmware/fw_jump.elf"; 
riscv64-linux-gnu-gcc
  ...
  -Wl,-T./build/platform/generic/firmware/fw_jump.elf.ld
  -o
  ./build/platform/generic/firmware/fw_jump.elf
```

`fw_jump.elf.ld` is preprocessed `fw_jump.elf.ldS`.

```bash
echo " CPP       platform/generic/firmware/fw_jump.elf.ld"; 
riscv64-linux-gnu-gcc
  ...
  
  -DOPENSBI_DEBUG
  -x
  c
  ./firmware/fw_jump.elf.ldS
  |
  grep -v "#"
  >
  ./build/platform/generic/firmware/fw_jump.elf.ld
```