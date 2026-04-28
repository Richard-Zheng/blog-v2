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

将近 1.9 GiB.

另外可以关掉 aliyun-service

```
systemctl disable --now aliyun.service
```
