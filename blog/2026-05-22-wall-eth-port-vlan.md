---
title: 墙壁网口单线复用小记
tags: ['OpenWrt']
---

为什么要折腾这个：单个 MAC 只能分到一个前缀 /128 的公网 IPv6，如果网口下接路由器，那就只有路由器能有一个 IPv6. 而我需要路由器和连到路由器的 archlinux 电脑主机都有 IPv6，于是需要电脑主机通过某种方式和墙壁网口桥接，然后在电脑主机上 DHCP 拿地址。

全程通过远程连接实现，人不在现场。能这么做有很大一部分原因是用了 Tailscale, 它会智能切换到当前可用的接口和线路，只要保证电脑和路由器总是能连接到互联网就能连上。

<!-- truncate -->

## 第 0 步：连接 Wi-Fi 确保连通性

首先为了保险起见，先在 archlinux 侧额外连一个 wifi 来提供互联网连通性备份

```
sudo nmcli device wifi connect "SZU_WLAN" name "szuwifi"
```

由于这个 wifi 需要 web portal 登录，所以先不使用默认路由

```
sudo nmcli connection modify szuwifi ipv4.never-default yes
```

然后为了登录 wifi, 给它加上连接 wifi 登录页的路由

```
sudo nmcli connection modify szuwifi -ipv4.routes "172.31.0.0/16 172.28.112.1 40"
```

为了能打开 web portal 登录页，再加一个公网地址用于跳转登录页

```
dig @192.168.247.6 +noedns mirrors.nju.edu.cn # 210.28.130.3
sudo nmcli connection modify szuwifi +ipv4.routes "210.28.0.0/16 172.28.112.1 40"
```

这时浏览器访问 http://mirrors.nju.edu.cn 应该可以跳转到登录页…… 不对，怎么直接打开了。哦原来走 IPv6 了。只有 Firefox 能关 IPv6, 打开 `about:config` 搜索 `ipv6` 关掉就行。这时候可以跳转登录页了。

成功登录认证后恢复路由

```
sudo nmcli connection modify szuwifi -ipv4.routes "172.31.0.0/16 172.28.112.1 40"
sudo nmcli connection modify szuwifi -ipv4.routes "210.28.0.0/16 172.28.112.1 40"
sudo nmcli connection modify szuwifi ipv4.never-default no
```

## 第 1 步：在 OpenWrt 上的配置

然后现在来改路由器配置。用 luci 网页端改配置，好处是有个超时 90 秒连不上 luci 自动恢复配置的功能。

wan 接口修改前对应设备为 `eth1`，电脑 lan 口为 `lan2`.

1. 记录 MAC：在 **网络 -> 接口** 中，记下 `wan` 接口 (`eth1`) 现在的 MAC 地址。这步是因为墙壁网口要登录，所以要保持 MAC 不变。
2. 创建 VLAN 设备：
   - 去 **网络 -> 接口 -> 设备**，点击 **添加设备配置**。
   - **类型**：VLAN (802.1q)
   - **基础设备**：连电脑的口，比如 `lan2`
   - **VLAN ID**：填 `10`（或其他数字）
   - 保存。此时多了一个叫 `lan2.10` 的设备。
3. 创建外网桥接：
   - 再次点击 **添加设备配置**。
   - **类型**：网桥 (Bridge device)
   - **设备名**：`bbwan`
   - **网桥端口**：勾选 `eth1` 和刚才建好的 `lan2.10`
   - **MAC 地址**（高级设置里）：填入第一步记下的 `eth1` 原 MAC。
4. 绑定接口：
   - 回到 **网络 -> 接口**。把 `wan` 和 `wan6` 接口的底层设备都改成 `bbwan`。
5. 保存并应用

这里折腾了很久，因为我忘了之前做 MAC VLAN + NDP Proxy 创建的接口会和网桥冲突，删掉以后好了。

## 第二步：在 Arch Linux 上的配置

NetworkManager 命令：

```
sudo nmcli connection add type vlan con-name "wanvlan10" ifname "vlan10" vlan.parent "enp4s0" vlan.id 10
sudo nmcli connection modify wanvlan10 ipv4.never-default yes

sudo nmcli connection up "Wired connection 1"
sudo nmcli connection up wanvlan10
```

然后需要 web portal 登录，具体操作和上面 wifi 登录基本一致。
