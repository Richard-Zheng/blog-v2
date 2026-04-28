# 阿里云 Ubuntu 配置

配置选的 2C2G, 开机一看

```
# free
               total        used        free      shared  buff/cache   available
Mem:         1651696      393420      418248        2652     1012996     1258276
Swap:              0           0           0
```

内存才 1.6 GiB！搜索了一下看到 [V2EX](https://v2ex.com/t/998120) 有人说是 crashkernel 占用的. 修改 `/etc/default/grub` 删除相关内核选项后，

```
grub-mkconfig -o /boot/grub/grub.cfg
```

然后重启。现在再来看内存：

```
# free
               total        used        free      shared  buff/cache   available
Mem:         1913840      404420     1308076        2672      353696     1509420
Swap:              0           0           0
```

约为 1.825 GiB, 使用 `dmesg | grep -i "Memory:"` 查看实际物理内存：

```
# dmesg | grep -i "Memory:"
[    0.045721] Memory: 1830108K/1995668K available (22528K kernel code, 4436K rwdata, 14408K rodata, 4920K init, 4792K bss, 165300K reserved, 0K cma-reserved)
[    0.151181] Freeing SMP alternatives memory: 48K
[    0.597818] Freeing initrd memory: 54708K
[    0.854832] Freeing unused decrypted memory: 2028K
[    0.860629] Freeing unused kernel image (initmem) memory: 4920K
[    0.867992] Freeing unused kernel image (rodata/data gap) memory: 1976K
# free
               total        used        free      shared  buff/cache   available
Mem:         1913840      361872     1305672        2664      399288     1551968
Swap:              0           0           0
```

1995668 KiB 和 2048 MiB 相比还是有 96.7 MiB 左右的差值，可能是 BIOS 等占的内存。最终可用内存为 1.825 GiB.

另外可以关掉 aliyun-service

```
systemctl disable --now aliyun.service
```
