---
title: Archlinux 安装 EPSON 老打印机驱动
tags: ['homelab']
---

打印机型号为 Epson M105 Series. 通过官网 http://download.ebz.epson.net/dsc/search/01/search/?OSC=LX 查询得驱动包名称为 epson-inkjet-printer-201215w，AUR 上也有对应的包。但是已经好几年不更新了。直接安装 AUR 上的 1.0.0 版无法正常编译，说是找不到 debug_msg 这个函数。我不想深究，正好有新版 1.0.1 的源码包，于是直接下载下来在本地开个服务器 serve 给 Arch 编译，对应修改 PKGBUILD 即可。fixbuild.patch 也不需要了。

<!-- truncate -->

安装后还得把 filter 可执行文件复制到 CUPS 的目录下

```
sudo cp /opt/epson-inkjet-printer-201215w/cups/lib/filter/epson_inkjet_printer_filter /usr/lib/cups/filter
```

完整 PKGBUILD 如下：

```
# Contributor: Andre Klitzing <andre () incubo () de>

pkgname=epson-inkjet-printer-201215w
_pkgname_filter=epson-inkjet-printer-filter
_suffix=1.src.rpm
pkgver=1.0.1
pkgrel=10
pkgdesc="Epson printer driver (M100, M105, M200, M205)"
arch=('i686' 'x86_64')
url="http://download.ebz.epson.net/dsc/search/01/search/?OSC=LX"
license=('LGPL' 'custom:Epson Licence Agreement')
depends=('cups' 'ghostscript')
#makedepends=('libtool' 'make' 'automake' 'autoconf')
source=(http://localhost:8000/${pkgname}-${pkgver}-${_suffix} fixbuild.patch)

build() {
  cd "$srcdir" || exit
  tar xzf $pkgname-$pkgver.tar.gz
  FILTER_FILE=$(ls $_pkgname_filter*.tar.gz)
  tar xzf $FILTER_FILE

  cd "${FILTER_FILE%.tar.gz}" || exit
  #patch -p1 -i "$srcdir"/fixbuild.patch
  autoreconf -f -i
  # if you have runtime problems: add "--enable-debug" and look into /tmp/epson-inkjet-printer-filter.txt
  ./configure LDFLAGS="$LDFLAGS -Wl,--no-as-needed" --prefix=/opt/$pkgname
  make
}

package() {
  cd "$srcdir/$pkgname-$pkgver" || exit
  install -d "$pkgdir/opt/$pkgname/"
  if [ "$CARCH" = "x86_64" ]; then
    cp -a --no-preserve=mode lib64 "$pkgdir/opt/$pkgname/"
  else
    cp -a --no-preserve=mode lib "$pkgdir/opt/$pkgname/"
  fi
  cp -a --no-preserve=mode resource "$pkgdir/opt/$pkgname/"

  if [ -e "watermark" ]; then
    cp -a --no-preserve=mode watermark "$pkgdir/opt/$pkgname/"
  fi
  install -d "$pkgdir/usr/share/cups/model/$pkgname"
  install -m 644 ppds/* "$pkgdir/usr/share/cups/model/$pkgname"

  cd "$srcdir" || exit
  FILTER_FILE=$(ls $_pkgname_filter*.tar.gz)
  cd "${FILTER_FILE%.tar.gz}" || exit
  install -d "$pkgdir/opt/$pkgname/cups/lib/filter/"
  install -m 755 src/epson_inkjet_printer_filter "$pkgdir/opt/$pkgname/cups/lib/filter/epson_inkjet_printer_filter"
}
sha256sums=('6b9252225c5e7210ed2acd36fb9c5fac4bb6f65aaf2ebfd3ce1b1690c63055ec'
            '85b0493972dcb92befd2bbf8d0ce705fc6280d54d83e985e9f7d0301bb01af50')
```