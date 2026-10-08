# NAT66 on OpenWrt 21.02

OpenWrt 21.02 用的是 `fw3` + `iptables` (legacy)，不是 nftables 的 `fw4`。这个版本的 `fw3`
**不支持 `masq6`**，所以给 LAN 配 IPv6 NAT 要手工下发 `ip6tables` 规则。

先确认上游是否真的需要 NAT66。如果上游通过 DHCPv6-PD 下发前缀，直接给 LAN 分配全局地址
即可；只有当上游只给一个 `/128`（校园网、CPE 后面的二级路由等）时才需要 NAT66。

## 1. 检查上游是否提供前缀委派 (PD)

```
ifstatus wan6 | grep -A4 '"ipv6-prefix"'
```

非空（比如 `2001:db8:1:2::/64`）说明有 PD，走原生路由，本文档不适用。为空则抓一次
DHCPv6 报文确认上游的回复：

```
opkg install tcpdump
tcpdump -i eth1 -n -vvv -s0 'udp port 546 or udp port 547'
```

另一个终端触发重新申请：

```
ubus call network.interface.wan6 down
ubus call network.interface.wan6 up
```

看到下面这样就是上游明确拒绝前缀委派：

```
dhcp6 advertise (IA_NA IAID:1 ... (IA_ADDR 2001:db8::1234 ...))
                (IA_PD IAID:1 T1:0 T2:0 (status-code NoPrefixAvail))
```

记下 WAN 设备名（下面假设 `eth1`）和 LAN 的 ULA 前缀（`ip -6 addr show br-lan`）。

## 2. 关闭默认路由的源地址限制

没有 PD 时 `netifd` 只添加一条带源地址限制的默认路由：

```
default from 2001:db8::1234 via fe80::1 dev eth1 proto static metric 512
```

`from 2001:db8::1234` 表示只有源地址等于 WAN `/128` 的包才匹配。LAN 用的是 ULA
（`fdxx:...`），源地址对不上，包没有路由可用，`ping6 -I` 会报 `Permission denied` 或静默
失败。**这不是 NAT 的问题，是路由的问题。**

```
uci set network.wan6.sourcefilter='0'
uci commit network
```

`sourcefilter='0'` 让 `netifd` 生成不带源限制的默认路由。对应的处理在
`package/network/ipv6/odhcp6c/files/dhcpv6.sh`：

```sh
[ "$sourcefilter" = "0" ] && proto_export "NOSOURCEFILTER=1"
```

`dhcpv6.script` 据此决定调用 `proto_add_ipv6_route` 时是否传源地址参数。

## 3. 配置 LAN 的 IPv6 服务

LAN 需要作为 RA 服务器和 DHCPv6 服务器，把 ULA 前缀和默认网关通告给客户端。DNS 走 IPv4
的 dnsmasq 就够了，不需要本机作为 IPv6 DNS 服务器，也不需要 NDP 代理：

```
uci set dhcp.lan.ra='server'
uci set dhcp.lan.dhcpv6='server'
uci set dhcp.lan.dns_service='0'
uci set dhcp.lan.ndp='disabled'
uci set dhcp.lan.ra_slaac='1'
uci set dhcp.lan.ra_flags='managed-config'
uci commit dhcp
/etc/init.d/odhcpd restart
```

各选项的含义（来自 `odhcpd` 的 README）：

| 选项 | 值 | 含义 |
|---|---|---|
| `ra` | `server` | 作为 RA 服务器 |
| `dhcpv6` | `server` | 作为 DHCPv6 服务器 |
| `dns_service` | `0` | 不把本机地址作为 DNS 服务通告 |
| `ndp` | `disabled` | 关闭 NDP 代理 |
| `ra_slaac` | `1` | PIO 里置 A (autonomous) 标志，启用 SLAAC |
| `ra_flags` | `managed-config` | RA 置 M (managed) 标志，告诉客户端用 DHCPv6 拿地址 |

**注意**：`ra_management` 和 `ra_flags` / `ra_slaac` 是互斥的，见 `config.c`：

```c
if (!tb[IFACE_ATTR_RA_FLAGS] && !tb[IFACE_ATTR_RA_SLAAC] &&
    (c = tb[IFACE_ATTR_RA_MANAGEMENT])) {
```

只要显式设了 `ra_flags` 或 `ra_slaac`，`ra_management` 就会被整个忽略。LuCI 界面写的是
`ra_management`，手动配置时容易和 `ra_flags` 混用，两者不要同时设。`ra_management` 的映射
关系是 `0`: OTHER + SLAAC，`1`: OTHER|MANAGED + SLAAC，`2`: OTHER|MANAGED 无 SLAAC。

## 4. 让 LAN 的 RA 通告默认网关

LAN 只有 ULA 时，`odhcpd` 默认不会把路由器自己通告为默认网关，RA 里 router lifetime 是 0：

```
ICMP6, router advertisement
  hop limit 64, Flags [managed, other stateful], pref medium, router lifetime 0s
  prefix info option: fd16:811a:a19c::/64, Flags [onlink, auto], valid infinity
```

按 RFC 4861，router lifetime = 0 表示"我不是默认网关"，客户端因此不会用它建默认路由。
原因在 `odhcpd` 的 `src/router.c`：LAN 只有 ULA 时 `valid_prefix` 一直为 false，于是
`nd_ra_router_lifetime` 被强制为 0：

```c
if (!IN6_IS_ADDR_ULA(&addr->addr.in6) || iface->default_router)
        valid_prefix = true;
...
if (default_route) {
        if (!valid_prefix) {
                syslog(LOG_WARNING, "A default route is present but there is no public prefix "
                                "on %s thus we don't announce a default route!", iface->name);
                adv.h.nd_ra_router_lifetime = 0;
        } else
                adv.h.nd_ra_router_lifetime = htons(...);
}
```

`odhcpd` README 里 `ra_default` 的说明：

```
ra_default  integer  0  Override default route
          0: default, 1: ignore no public address, 2: ignore all
```

设成 `2`：

```
uci set dhcp.lan.ra_default='2'
uci commit dhcp
/etc/init.d/odhcpd restart
```

再抓 RA 验证，`router lifetime` 应该变成非 0（默认 1800s）。周期 RA 的默认最大间隔是
600s，直接抓可能等很久，重启 `odhcpd` 会立即发一次初始 RA：

```
tcpdump -i br-lan -n -vvv 'icmp6 and ip6[40] == 134'
```

## 5. 添加 NAT66 规则

`fw3` 不认识 `masq6`：

```
uci set firewall.@zone[1].masq6='1'
```

会打印 `Warning: Option @zone[1].masq6 is unknown` 并忽略它。运行时 `nat` 表里也不会有
任何 masquerade 规则：

```
ip6tables -t nat -S POSTROUTING
-P POSTROUTING ACCEPT
```

原因是 `fw3` 的 `zones.c` 里生成 NAT 规则的分支硬编码了 IPv4，IPv6 分支根本不存在：

```c
case FW3_TABLE_NAT:
        if (zone->masq && handle->family == FW3_FAMILY_V4)
```

所以 NAT66 规则只能手工下发。写到 `/etc/firewall.user`（先删后加，保证重复执行幂等）：

```
cat >> /etc/firewall.user <<'EOF'

# IPv6 NAT66
while ip6tables -t nat -D POSTROUTING -o eth1 -j MASQUERADE 2>/dev/null; do :; done
ip6tables -t nat -A POSTROUTING -o eth1 -j MASQUERADE
EOF
```

把 `eth1` 换成实际的 WAN 设备名。

## 6. 让 firewall.user 在 reload 时也执行

`fw3` 的 `reload` 和 `restart` 对 include 的处理不同，见 `includes.c`：

```c
void fw3_run_includes(struct fw3_state *state, bool reload)
{
        list_for_each_entry(include, &state->includes, list)
        {
                if (reload && !include->reload)
                        continue;
                ...
        }
}
```

`reload()` 传 `reload=true`，所以没有 `reload='1'` 的 include 会被跳过。默认的
`/etc/firewall.user` 段没设这个选项，`fw3 reload` 不会执行它（只有 `restart` 才会），
NAT66 规则在 reload 后就没了。

`miniupnpd` 的 include 段一直能正常执行，就是因为它设了 `reload='1'`：

```
uci set firewall.@include[0].reload='1'
uci commit firewall
```

## 7. 应用并验证

```
ifdown wan6; ifup wan6
fw3 reload
```

```
ip6tables -t nat -S POSTROUTING
# 应该有 -A POSTROUTING -o eth1 -j MASQUERADE

ip -6 route show default
# 应该没有 "from ..." 前缀
```

用一个 LAN 的 ULA 地址做源测连通性（`ping6 -I` 要传地址，不是前缀）：

```
ping6 -I fd16:811a:a19c::1 2400:3200::1
curl -6 --interface fd16:811a:a19c::1 -o /dev/null -w '%{http_code}\n' http://www.taobao.com
```

确认 NAT 会话已建立：

```
cat /proc/net/nf_conntrack | grep ipv6
# src=fd16:...  dst=<公网>
# src=<公网>    dst=2001:db8::1234   <- 回包被 masquerade 回 WAN 的 /128
```

## 8. 持久化

改动都已通过 `uci commit` 写入 `/etc/config/`，重启后自动生效：

| 配置 | 作用 |
|---|---|
| `network.wan6.sourcefilter='0'` | 去掉默认路由的源地址限制 |
| `dhcp.lan.ra='server'` / `dhcpv6='server'` | LAN 作为 RA / DHCPv6 服务器 |
| `dhcp.lan.dns_service='0'` | 不把本机地址作为 DNS 服务通告 |
| `dhcp.lan.ndp='disabled'` | 关闭 NDP 代理 |
| `dhcp.lan.ra_slaac='1'` | 启用 SLAAC |
| `dhcp.lan.ra_flags='managed-config'` | RA 置 M 标志 |
| `dhcp.lan.ra_default='2'` | RA 通告默认网关 |
| `firewall.@include[0].reload='1'` | `fw3 reload` 时执行 `firewall.user` |
| `/etc/firewall.user` 里的 `ip6tables ... MASQUERADE` | NAT66 规则本身 |

`/etc/firewall.user` 是持久化文件，不在 tmpfs 上。每次 `fw3` 启动 / reload 都会重新执行。

## 附：关于 fullcone

IPv4 fullcone 由两个 opkg 包提供：`kmod-ipt-fullconenat`（内核模块 `xt_FULLCONENAT.ko`）
和 `iptables-mod-fullconenat`（用户态 `libipt_FULLCONENAT.so`），代码来自
[Chion82/netfilter-full-cone-nat](https://github.com/Chion82/netfilter-full-cone-nat)。

上游 `fw3` 不认识 `fullcone` 选项，需要 `package/network/config/firewall` 里的
`100-fullconenat.patch` 支持。开了之后（LuCI 里是 `luci-app-turboacc` 的开关）会把
`MASQUERADE` target 换成 `FULLCONENAT`：

```
uci set firewall.@defaults[0].fullcone='1'
uci commit firewall
fw3 reload

# iptables -t nat -S
# -A zone_wan_postrouting -j FULLCONENAT
# -A zone_wan_prerouting  -j FULLCONENAT
```

**IPv6 没有 fullcone**。同一个模块的 target 注册写死了 IPv4：

```c
static struct xt_target tg_reg[] __read_mostly = {
 {
  .name       = "FULLCONENAT",
  .family     = NFPROTO_IPV4,
  .targetsize = sizeof(struct nf_nat_ipv4_multi_range_compat),
  ...
```

映射表存的是 `__be32`（32 位地址），用户态也没有 `libip6t_FULLCONENAT.so`，
`ip6tables -j FULLCONENAT` 会报 `Couldn't load target`。

所以本文档配出来的是**对称 NAT**，不是 fullcone。内网主动出网够用；外部主动访问内网设备需要
额外加 DNAT 规则。

## 附：DNS

不需要为了 IPv6 专门给 LAN 下发 IPv6 DNS。DNS 查询走的传输层（UDP/TCP）和它查询的记录
类型（A / AAAA）是独立的两件事，用 IPv4 的 DNS 服务器照样能拿到 AAAA 记录：

```
nslookup www.baidu.com 192.168.1.1
Server:  192.168.1.1
Address: 192.168.1.1#53
Name:    www.baidu.com
Address 1: 183.2.172.177
Address 2: 240e:ff:e020:99b:0:ff:b099:cff1   <- AAAA 正常返回
```

所以第 3 步里把 `dns_service` 设为 `0`（不把本机地址作为 IPv6 DNS 服务通告），客户端继续
用 DHCPv4 下发的 `192.168.1.1` 就行。

`dns_service` 的默认值其实是 `1`（见 `odhcpd` 的 `config.c`：`iface->dns_service = true;`），
即默认会把本机地址作为 DNS 服务通过 DHCPv6 / RA 下发。设成 `0` 只是去掉这个多余的通告。
