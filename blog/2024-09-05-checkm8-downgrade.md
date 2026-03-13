---
title: Checkm8 设备刷入未签名版本 iOS
tags: ['iOS']
---

首先下载 [Legacy iOS Kit](https://github.com/LukeZGD/Legacy-iOS-Kit)，理论上好像直接用这个脚本就可以了。但是它对网络的要求比较高。

或者[手动设置 nonce](https://web.archive.org/web/20230505231139/https://gist.github.com/0xallie/aac55c97f7925cddcf5ec3167f85dfe8):

<!-- truncate -->

0. Put it in DFU
1. `wget https://alexia.lol/gaster/gaster-Linux.zip && unzip gaster-Linux.zip && chmod +x gaster`
2. `sudo ./gaster pwn && sudo ./gaster reset`
3. `sudo ./futurerestore -t <drag_blob_here> --use-pwndfu --set-nonce --latest-sep --latest-baseband <drag_ipsw_here>`

后使用 [Legacy iOS Kit](https://github.com/LukeZGD/Legacy-iOS-Kit) 附带的 FutureRestore 刷入。Option 选择 No RSEP (`--no-rsep`)