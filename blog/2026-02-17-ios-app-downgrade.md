---
title: iOS App 降级
---

简单记录一下 iOS App 降级需要用到的各种工具。

<!-- truncate -->

参考：[iOS 旧版 APP 推荐与降级方法（持续更新）](https://qianling.pw/ios-app/)

ipa 下载工具（Windows）：[ipaDown](https://www.52pojie.cn/thread-1863801-1-1.html)

iTunes 12.6.5.3(最后一个带App Store的版本)：

- macOS: https://secure-appldnld.apple.com/itunes12/091-33628-20170922-EF8F0FE4-9FEF-11E7-B113-91CF9A97A551/iTunes12.6.3.dmg
- Windows i386: https://secure-appldnld.apple.com/itunes12/091-33627-20170922-EF8CB708-9FEF-11E7-8504-92CF9A97A551/iTunesSetup.exe
- Windows x86_64: https://secure-appldnld.apple.com/itunes12/091-33626-20170922-F51D3530-A003-11E7-8324-03D19A97A551/iTunes64Setup.exe

抓包域名为 `p[xx]-buy.itunes.apple.com`

安装 ipa 可以用[AppManager](https://github.com/kawaiizenbo/AppManager)