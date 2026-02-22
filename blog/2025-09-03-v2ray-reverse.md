---
title: v2ray 反向代理
---

搭建 v2ray 反向代理，类似 frp, 但是可以使用 shadowsocks 协议。

参考：[反向代理 2 - 新 V2Ray 白话文指南](https://guide.v2fly.org/app/reverse2.html)

<!-- truncate -->

内网 bridge：

```json
{
  "log": {
    "access": "none",
    "loglevel": "error"
  },
  "reverse": {
    "bridges": [
      {
        "tag": "bridge",
        "domain": "private.cloud.com"
      }
    ]
  },
  "outbounds": [
    {
      "protocol": "shadowsocks",
      "settings": {
        "servers": [
          {
            "address": "rev.ip.com",
            "method": "aes-128-gcm",
            "password": "passwd",
            "port": 16823
          }
        ]
      },
      "tag": "interconn"
    },
    {
      "protocol": "freedom",
      "settings": {},
      "tag": "out"
    }
  ],
  "routing": {
    "rules": [
      {
        "type": "field",
        "inboundTag": [
          "bridge"
        ],
        "domain": [
          "full:private.cloud.com"
        ],
        "outboundTag": "interconn"
      },
      {
        "type": "field",
        "inboundTag": [
          "bridge"
        ],
        "outboundTag": "out"
      }
    ]
  }
}
```

注意 `"domain": [ "full:private.cloud.com" ]` 没写上就会回环，折腾一下午。

公网 portal：

```json
{
  "log": {
    "loglevel": "info"
  },
  "reverse": {
    "portals": [
      {
        "tag": "portal",
        "domain": "private.cloud.com"
      }
    ]
  },
  "inbounds": [
    {
      "tag":"ssh-external",
      "port":2233,
      "protocol":"dokodemo-door",
        "settings":{
          "address":"127.0.0.1",
          "port":22,
          "network":"tcp"
        }
    },
    {
      "tag": "external",
      "port": 1989,
      "protocol": "shadowsocks",
      "settings": {
        "method": "aes-128-gcm",
        "password": "passwd"
      }
    },
    {
      "tag": "interconn",
      "port": 16823,
      "protocol": "shadowsocks",
      "settings": {
        "method": "aes-128-gcm",
        "password": "passwd"
      }
    }
  ],
  "routing": {
    "rules": [
      {
        "type": "field",
        "inboundTag": [
          "ssh-external"
        ],
        "outboundTag": "portal"
      },
      {
        "type": "field",
        "inboundTag": [
          "external"
        ],
        "outboundTag": "portal"
      },
      {
        "type": "field",
        "inboundTag": [
          "interconn"
        ],
        "domain":[  
          "full:private.cloud.com"
        ],
        "outboundTag": "portal"
      }
    ]
  }
}
```
