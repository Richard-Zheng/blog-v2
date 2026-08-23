# Git Internal

## Object

参考 [10.2 Git Objects - Pro Git book](https://git-scm.com/book/en/v2/Git-Internals-Git-Objects) 以及 [StackOverflow](https://stackoverflow.com/questions/20666331/how-git-branches-and-tags-are-stored-in-disks/20800756#20800756)

创建一个新 Git 仓库

```bash
cd /tmp
git init test
cd test
```

向 Git object 写入一个 Blob

```bash
echo 'test content' | git hash-object -w --stdin
```

会输出

```
d670460b4b4aece5915caf5c68d12f560a9fe3e4
```

这是 SHA1 hash，那么具体 hash 的是啥呢？这里有个 ruby 脚本

```ruby
content = "test content\n"
header = "blob #{content.bytesize}\0"
store = header + content
require 'digest/sha1'
sha1 = Digest::SHA1.hexdigest(store)
```

得到

```
irb(main):001> content = "test content\n"
irb(main):002> header = "blob #{content.bytesize}\0"
irb(main):003> store = header + content
=> "blob 13\u0000test content\n"
irb(main):004> require 'digest/sha1'
irb(main):005> sha1 = Digest::SHA1.hexdigest(store)
=> "d670460b4b4aece5915caf5c68d12f560a9fe3e4"
```

存在哪里呢

```
$ find .git/objects
.git/objects
.git/objects/d6
.git/objects/d6/70460b4b4aece5915caf5c68d12f560a9fe3e4
.git/objects/info
.git/objects/pack
```

存的是 `compress(store)` 写一个 Python 脚本查看：

```py
import zlib
import sys

with open(sys.argv[1], 'rb') as f:
    raw_data = zlib.decompress(f.read())
    printable_data = raw_data.replace(b'\0', b'\n')
    print(printable_data.decode('utf-8', errors='replace'))
```

保存为 `blobcat.py` 然后运行

```
$ python3 blobcat.py .git/objects/d6/70460b4b4aece5915caf5c68d12f560a9fe3e4
blob 13
test content

```

这里有个有些简化的 mental model: 你在 Git 存储库中跟踪的所有文件（以及它们每一次 git add 时的快照）都以 Blob Object 的形式存储在 `.git/objects` 下。注意存储的只是每个文件的二进制内容，而不存储文件名。由此也可以看出如果有两个内容完全一样的 Blob, 那它们只会在 `.git/objects` 下被存储一份（因为 SHA1 相同）。
