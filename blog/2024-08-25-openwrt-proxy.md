---
title: 在 OpenWrt 上配置透明代理
tags: ['OpenWrt']
---

记录了在 OpenWrt 上配置基于 nftables redirect 的透明代理的过程。

<!-- truncate -->

首先配置你要用的代理软件，这里以 mihomo 为例

设置开机自动启动

```bash
vi /etc/init.d/mihomo
```

内容如下

```bash
#!/bin/sh /etc/rc.common

START=90
STOP=15

USE_PROCD=1
#PROCD_DEBUG=1

CONF=/etc/mihomo
PROG=/usr/bin/mihomo

start_service() {
        procd_open_instance
        procd_set_param command $PROG -d $CONF
        procd_set_param stdout 1
        procd_set_param stderr 1
        procd_set_param respawn ${respawn_threshold:-3600} ${respawn_timeout:-5} ${respawn_retry:-9}
        procd_close_instance
}
```

设置权限，并使之生效

```bash
chmod 755 /etc/init.d/mihomo
service mihomo enable
```

启动 mihomo

```bash
service mihomo start
```

然后配置 redir 服务，注意要监听 `0.0.0.0` 因为 nftables redirect 后的 IP 是局域网中的本机 IP 如 192.168.1.1

```
listeners:
- name: redir-in
  type: redir
  port: 1101
  listen: 0.0.0.0
```

配置 nftables 规则

```bash
#!/bin/sh
curl -s -L https://ispip.clang.cn/all_cn_cidr.txt \
  | sed -e '$!s/$/,/' \
  | sed -e '1s/^/define CN_CIDR_4 = {\n/' \
  | sed -e '$a}' \
  > /etc/cn_cidr.nft
if ! [[ $? == 0 ]] ; then
  echo "ERROR: unable to download cn cidr list."
  nft delete table ip proxy4 2> /dev/null
  exit 1
fi ;

nft delete table ip proxy4 2> /dev/null
nft -f - << EOF
include "/root/cn_cidr.nft"

table ip proxy4 {
    set intranet {
        typeof ip daddr
        flags interval
        elements = {
            0.0.0.0/8,
            10.0.0.0/8,
            127.0.0.0/8,
            169.254.0.0/16,
            172.16.0.0/12,
            192.168.0.0/16,
            224.0.0.0/4,
            240.0.0.0/4,
        }
    }
    set chnroute {
        typeof ip daddr
        flags interval
        elements = \$CN_CIDR_4
    }
    set static_ip {
        typeof ip saddr
        flags interval
        comment "generated from openwrt dhcp"
    }
    set static_ip_mac {
        type ether_addr
        flags interval
        comment "generated from openwrt dhcp"
    }
    chain prerouting {
        type nat hook prerouting priority 0; policy accept;

        ip daddr @intranet return
        ip daddr @chnroute return
        ip saddr @static_ip return

        ip protocol tcp redirect to :1101
        #meta l4proto { tcp, udp } tproxy ip to :1536 accept
    }
}
EOF

i=0
while uci get dhcp.@host[$i] &> /dev/null ; do
  nft add element ip proxy4 static_ip_mac { $(uci get dhcp.@host[$i].mac) }
  nft add element ip proxy4 static_ip { $(uci get dhcp.@host[$i].ip) }
  i=$((i+1));
done
```

最后注意设置 OpenWrt 防火墙，放行 LAN 到路由器的流量。

然后再设置 mosdns

```bash
#!/bin/sh /etc/rc.common

START=75
USE_PROCD=1

PROG=/usr/bin/mosdns
PORT=5533

restore_setting() {
  rm -f /etc/mosdns/redirect.lock
  uci set dhcp.@dnsmasq[0].noresolv='0'
  uci del dhcp.@dnsmasq[0].cachesize
  uci del_list dhcp.@dnsmasq[0].server="127.0.0.1#$PORT"
  uci commit dhcp
}

redirect_setting() {
  uci add_list dhcp.@dnsmasq[0].server="127.0.0.1#$PORT"
  uci set dhcp.@dnsmasq[0].rebind_protection='0'
  uci set dhcp.@dnsmasq[0].noresolv="1"
  uci set dhcp.@dnsmasq[0].cachesize='0'
  uci commit dhcp
}

reload_dnsmasq() {
  /etc/init.d/dnsmasq reload
}

start_service() {
  procd_open_instance mosdns
  #procd_set_param env QUIC_GO_DISABLE_RECEIVE_BUFFER_WARNING=true
  procd_set_param command $PROG start
  procd_append_param command -d "/etc/mosdns"
  procd_set_param stdout 1
  procd_set_param stderr 1
  procd_set_param respawn
  procd_close_instance mosdns
  redirect_setting
  reload_dnsmasq
}

stop_service() {
  [ -f "/etc/mosdns/redirect.lock" ] && restore_setting
  reload_dnsmasq
}
```

mosdns 配置如下

```
log:
  level: error

include: []

plugins:
  - tag: lcache
    type: cache
    args:
      size: 10240
      lazy_cache_ttl: 86400

  ###### Data #######

  # ad and tracker domain list
  - tag: skk_reject_list
    type: domain_set
    args:
      files:
        - ./reject.conf

  # Felixonmars' chinese domain list
  - tag: felix_china_list
    type: domain_set
    args:
      files:
        - ./accelerated-domains.china.raw.txt

  - tag: gfwlist
    type: domain_set
    args:
      files:
        - ./proxy-list.txt
        - ./gfwlist_domain.txt

  - tag: domestic_ip
    type: ip_set
    args:
      files:
        - ./all_cn_cidr.txt

  ###### Forward DNS ######

  - tag: forward_domestic
    type: forward
    args:
      concurrent: 2
      upstreams:
        - addr: udp://223.5.5.5
        - addr: udp://223.6.6.6

  # use clash dns on localhost redir-host mode
  - tag: forward_global
    type: forward
    args:
      upstreams:
        - addr: udp://127.0.0.1:1053

  ####### Try 2 DNS and respond based on IP #####

  - tag: try_domestic
    type: sequence
    args:
      - exec: $forward_domestic
      - matches: resp_ip $domestic_ip
        exec: accept
      - exec: query_summary domestic query get non-chinese ip
      - exec: drop_resp

  - tag: try_global
    type: sequence
    args:
      - exec: prefer_ipv4
      - exec: $forward_global
      #- matches: "!resp_ip $domestic_ip"
      #  exec: "nftset inet,my_table,my_set,ipv4_addr,24 inet,my_table,my_set,ipv6_addr,48"
      - exec: accept

  - tag: auto_try_fallback
    type: fallback
    args:
      primary: try_domestic
      secondary: try_global
      threshold: 500
      always_standby: true

  - tag: fix_google_play
    type: redirect
    args:
      rules:
        - services.googleapis.cn services.googleapis.com

  - tag: main
    type: sequence
    args:
      - matches:
          - qname $skk_reject_list
        exec: reject 3

      - exec: cache 1024
      - matches:
          - has_resp
        exec: accept

      - exec: $fix_google_play

      - matches:
          - qname $felix_china_list
        exec: $forward_domestic
      - matches:
          - has_resp
        exec: accept

      - matches:
          - qname $gfwlist
        exec: $forward_global
      - matches:
          - has_resp
        exec: accept

      - exec: $auto_try_fallback

  - type: udp_server
    args:
      entry: main
      listen: 127.0.0.1:5533
  - type: tcp_server
    args:
      entry: main
      listen: 127.0.0.1:5533
```