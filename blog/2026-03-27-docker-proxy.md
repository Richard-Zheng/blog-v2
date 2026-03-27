---
title: 不依赖 TUN、路由表、防火墙规则配置透明代理
tags: ['Docker']
---

记录了在 Docker 容器中配置透明代理的过程。

容器为 rootful，但容器中仅有 `cap_net_bind_service` 和 `cap_net_raw` 两个权限，无法使用 TUN、路由表、防火墙规则等方式配置透明代理。

以下配置使用 [mihomo v1.19.21](https://github.com/MetaCubeX/mihomo/releases/tag/v1.19.21).

<!-- truncate -->

首先想到的是 SNI 代理。SNI 是 TLS 协议中的一个扩展，允许客户端在握手过程中告诉服务器它想要连接的主机名，且这个信息是明文传输的，因此只要我们能够通过某种方式拦截到这个信息，就可以实现透明代理。

首先配置 DNS 让其对于要代理的域名解析到环回地址。

```yaml
dns:
  enable: true
  listen: 0.0.0.0:53
  ipv6: false
  enhanced-mode: fake-ip
  fake-ip-range: 127.18.0.1/16
  fake-ip-filter-mode: rule
  fake-ip-filter:
    - GEOSITE,gfw,fake-ip
    - MATCH,real-ip
  default-nameserver:
    - tls://223.5.5.5
    - tls://223.6.6.6
  nameserver:
    - https://doh.pub/dns-query
    - https://dns.alidns.com/dns-query
```

然后监听 80、443 端口。

```yaml
tunnels:
  - network: [tcp]
    address: 0.0.0.0:443
    target: 127.0.0.1:443
  - network: [tcp]
    address: 0.0.0.0:80
    target: 127.0.0.1:80
```

流量进入 80、443 端口后会经过 sniffer, 在这里根据 SNI 替换目标地址/域名。

```yaml
sniffer:
  enable: true
  override-destination: true
  sniff:
    HTTP:
      ports: [80]
    TLS:
      ports: [443, 8443]
    QUIC:
      ports: [443, 8443]
  skip-domain:
    - "Mijia Cloud"
    - "+.push.apple.com"
```

如上应该已经可以正常代理。在折腾的时候发现也许可以使用 fakeip，其实上面已经配置了 fake-ip-range 为 `127.18.0.0/16`，但是 tunnel 入站的流量似乎不会经过 fakeip 处理（目的地址被设为 target 里写的 `127.0.0.1:443`）。尝试了一下 redir 入站不能正常工作，原因不明。但是 tproxy 入站是可以正常工作的。

```yaml
tproxy-port: 443
listeners:
  - name: tproxy-backup
    type: tproxy
    port: 80
    listen: 0.0.0.0
    udp: true
```

此时把 sniffer 关闭，删除 tunnels 片段，应该也可以正常代理了。

其余部分参考[文档](https://wiki.metacubex.one/example/conf/)配置即可。

最后还需要把 rules 里面的 IP 规则全部删掉，防止触发 DNS 解析。

完整配置如下：

```yaml
# url 里填写自己的订阅,名称不能重复
proxy-providers:
  provider1:
    url: ""
    type: http
    interval: 86400
    health-check: {enable: true,url: "https://www.gstatic.com/generate_204", interval: 300}
    override:
      additional-prefix: "[provider1]"

proxies: 
  - name: "直连"
    type: direct
    udp: true

tproxy-port: 443
listeners:
  - name: tproxy-backup
    type: tproxy
    port: 80
    listen: 0.0.0.0
    udp: true

mixed-port: 7890
ipv6: false
allow-lan: true
unified-delay: false
tcp-concurrent: true
external-controller: 0.0.0.0:9090
external-ui: ui
external-ui-url: "https://github.com/MetaCubeX/metacubexd/archive/refs/heads/gh-pages.zip"

log-level: debug

geodata-mode: true
geox-url:
  geoip: "https://github.com/MetaCubeX/meta-rules-dat/releases/download/latest/geoip-lite.dat"
  geosite: "https://github.com/MetaCubeX/meta-rules-dat/releases/download/latest/geosite.dat"
  mmdb: "https://github.com/MetaCubeX/meta-rules-dat/releases/download/latest/country-lite.mmdb"
  asn: "https://github.com/MetaCubeX/meta-rules-dat/releases/download/latest/GeoLite2-ASN.mmdb"

find-process-mode: strict
global-client-fingerprint: chrome

profile:
  store-selected: true
  store-fake-ip: false

sniffer:
  enable: false
  override-destination: true
  sniff:
    HTTP:
      ports: [80]
    TLS:
      ports: [443, 8443]
    QUIC:
      ports: [443, 8443]
  skip-domain:
    - "Mijia Cloud"
    - "+.push.apple.com"

tun:
  enable: false
  stack: mixed
  dns-hijack:
    - "any:53"
    - "tcp://any:53"
  auto-route: true
  auto-redirect: true
  auto-detect-interface: true

dns:
  enable: true
  listen: 0.0.0.0:53
  ipv6: false
  enhanced-mode: fake-ip
  fake-ip-range: 127.18.0.1/16
  fake-ip-filter-mode: rule
  fake-ip-filter:
    - GEOSITE,geolocation-!cn,fake-ip
    - GEOSITE,gfw,fake-ip
    - GEOSITE,CN,real-ip
    - MATCH,real-ip
  default-nameserver:
    - tls://223.5.5.5
    - tls://223.6.6.6
  nameserver:
    - https://doh.pub/dns-query
    - https://dns.alidns.com/dns-query

proxy-groups:

  - name: 默认
    type: select
    proxies: [自动选择,直连,香港,台湾,日本,新加坡,美国,其它地区,全部节点]

  - name: Google
    type: select
    proxies: [默认,香港,台湾,日本,新加坡,美国,其它地区,全部节点,自动选择,直连]

  - name: Telegram
    type: select
    proxies: [默认,香港,台湾,日本,新加坡,美国,其它地区,全部节点,自动选择,直连]

  - name: Twitter
    type: select
    proxies: [默认,香港,台湾,日本,新加坡,美国,其它地区,全部节点,自动选择,直连]

  - name: 哔哩哔哩
    type: select
    proxies: [默认,香港,台湾,日本,新加坡,美国,其它地区,全部节点,自动选择,直连]

  - name: 巴哈姆特
    type: select
    proxies: [默认,香港,台湾,日本,新加坡,美国,其它地区,全部节点,自动选择,直连]

  - name: YouTube
    type: select
    proxies: [默认,香港,台湾,日本,新加坡,美国,其它地区,全部节点,自动选择,直连]

  - name: NETFLIX
    type: select
    proxies: [默认,香港,台湾,日本,新加坡,美国,其它地区,全部节点,自动选择,直连]

  - name: Spotify
    type: select
    proxies:  [默认,香港,台湾,日本,新加坡,美国,其它地区,全部节点,自动选择,直连]

  - name: Github
    type: select
    proxies:  [默认,香港,台湾,日本,新加坡,美国,其它地区,全部节点,自动选择,直连]

  - name: 国内
    type: select
    proxies:  [直连,默认,香港,台湾,日本,新加坡,美国,其它地区,全部节点,自动选择]

  - name: 其他
    type: select
    proxies:  [默认,香港,台湾,日本,新加坡,美国,其它地区,全部节点,自动选择,直连]

  #分隔,下面是地区分组
  - name: 香港
    type: select
    include-all: true
    exclude-type: direct
    filter: "(?i)港|hk|hongkong|hong kong"

  - name: 台湾
    type: select
    include-all: true
    exclude-type: direct
    filter: "(?i)台|tw|taiwan"

  - name: 日本
    type: select
    include-all: true
    exclude-type: direct
    filter: "(?i)日|jp|japan"

  - name: 美国
    type: select
    include-all: true
    exclude-type: direct
    filter: "(?i)美|us|unitedstates|united states"

  - name: 新加坡
    type: select
    include-all: true
    exclude-type: direct
    filter: "(?i)(新|sg|singapore)"

  - name: 其它地区
    type: select
    include-all: true
    exclude-type: direct
    filter: "(?i)^(?!.*(?:🇭🇰|🇯🇵|🇺🇸|🇸🇬|🇨🇳|港|hk|hongkong|台|tw|taiwan|日|jp|japan|新|sg|singapore|美|us|unitedstates)).*"

  - name: 全部节点
    type: select
    include-all: true
    exclude-type: direct

  - name: 自动选择
    type: url-test
    include-all: true
    exclude-type: direct
    tolerance: 10

rules:
  - GEOSITE,github,Github
  - GEOSITE,twitter,Twitter
  - GEOSITE,youtube,YouTube
  - GEOSITE,google,Google
  - GEOSITE,telegram,Telegram
  - GEOSITE,netflix,NETFLIX
  - GEOSITE,bilibili,哔哩哔哩
  - GEOSITE,bahamut,巴哈姆特
  - GEOSITE,spotify,Spotify
  - GEOSITE,CN,国内
  - GEOSITE,geolocation-!cn,其他

  - MATCH,其他
```