---
slug: word-workflow
title: Word 毕业论文自用工作流
tags: ['Word']
---

到了要写毕业论文的时候，首先面临一个选择：LaTeX 还是 Word ？这个问题你需要咨询你的指导老师，看查重系统能否提交 Word 以外的格式。我的学校只接受 Word 所以被迫选择 Word.

我写论文的思路是每章分开写，先写实验部分，再写理论部分，最后写绪论和总结。每章节的草稿使用 Markdown 格式，最后再合并成一个大 Markdown 然后使用 pandoc 转成 docx. 这样有个好处是我写公式可以直接在 Markdown 里写 LaTeX, 然后 pandoc 会自动帮我转成 MS Word 里的公式。

<!-- truncate -->

**MD to Word (pandoc):** pandoc 转换 word 时可以依据给定的模板。我用了 [Achuan-2/pandoc_docx_template](https://github.com/Achuan-2/pandoc_docx_template) 这个模板。它的格式总体过关，但是有个问题是标题 1 没有编号。这里有个坑：Word 的列表级别和样式是互相绑定的，改了列表级别后样式也跟着改。修改方法：搜索多级列表设置 - 定义新的多级列表 - 依次选中每个级别然后修改“将级别链接到样式”即可。

**Word to MD (pandoc):** 毕业论文里的源文件我都用 Git 管理。为了优化 Word 文档版本控制的效果，我采用了 [farhan-syah/git-docx](https://github.com/farhan-syah/git-docx) 里的方法，在 commit 的时候通过 hook 来生成每个 Word 对应的 Markdown，这样在 diff 不同版本的时候就能很容易看出内容的改动。至于格式变动，我改的不频繁且不好用纯文本表示，只能不管了。为了让 pandoc 正确处理 Word 中的公式，修改了一下脚本

```diff
diff --git a/hooks/pre-commit b/hooks/pre-commit
index 68c1e35..5dc1c0f 100755
--- a/hooks/pre-commit
+++ b/hooks/pre-commit
@@ -111,7 +111,7 @@ for file in $(git diff --cached --name-only --diff-filter=d | grep "\.docx$"); d
   #echo "$mdfile"
 
   # convert .docx file to Markdown
-  pandoc -t markdown_strict "$file" -o "$mdfile" || {
+  pandoc -t markdown+tex_math_dollars "$file" -o "$mdfile" || {
     echo "Conversion to Markdown failed"
     exit 1
   }
```

pandoc 生成出来的 docx 还是塞了很多乱糟糟的样式，为了省心我还是找到学校模板然后一段一段地复制进去了......现在看来可能不复制也行？

**Windows 虚拟机:** 为了确保兼容性以及后续的 VBA 脚本需求，不得不用 Windows 上的 MS Word （之前我一直用的是 Archlinux + WPS Office）。为此我用 libvirt 开了一个 Windows 10 虚拟机并安装 MS Word，然后使用 virtofs 把 Linux 下的目录共享进虚拟机里，省得再装 Git 和 pandoc 了，效果还行。为了确保 virtofs 正常工作，需要启用共享内存

```xml
<memoryBacking>
  <source type="memfd"/>
  <access mode="shared"/>
</memoryBacking>
```

然后启用 virtofs

```xml
<filesystem type="mount" accessmode="passthrough">
  <driver type="virtiofs"/>
  <binary path="/usr/lib/virtiofsd"/>
  <source dir="/home/yourusername/vshare"/>
  <target dir="vshare"/>
  <alias name="fs0"/>
  <address type="pci" domain="0x0000" bus="0x06" slot="0x00" function="0x0"/>
</filesystem>
```

在 Windows 虚拟机中安装 WinFSP 和 virtio 驱动，在服务中启用 VirtIO-FS Service 即可。

然后就要处理编号问题。我在写 markdown 的时候自然是没有人工编号的，因为我随时可能增删内容。现在需要用脚本批量给图片、表格、公式加编号。

**公式加编号:** 这个应该是最难搞的。学校要求是每个单独一行的公式右侧都得有 `(1.1)` 这样的编号。一种方案是 [如何利用 Word 徒手写公式？ - 知乎](https://zhuanlan.zhihu.com/p/22199318) 这篇文章里的建表格。但是有人提到用表格给公式编号会导致被格式检测系统标红，它会既认为你没给那个用于排版对齐的不可见的表格编号，也没给公式编号......所以用制表符的方案。首先按照 [这篇知乎回答](https://www.zhihu.com/question/19689000/answer/48425000) 新建/修改一个公式样式。

然后需要给所有公式前后都加一个制表符，再在后面附上编号就好啦。但是我尝试用 VBA 脚本的时候怎么也不能正常给公式前/后加制表符，因为 Word 总是默认你是要在公式内部加......最后我手动给所有公式加了制表符和括号`()`. 然后是自动给公式维护编号的代码：

Prompt 如下：

我现在需要给毕业论文里面的公式加编号，只用编号不用任何交叉引用

我现在新加了个`公式`样式

```
缩进:
    制表位:  18 字符, 居中 +  36 字符, 右对齐
    首行缩进:  0 字符, 样式: 链接, 在样式库中显示
    基于: 正文
```

然后我需要处理这两种情况

```
[Tab]公式[Tab]()
[Tab]公式[Tab](3.2)
```

更新为：

```
[Tab]公式[Tab](3.1)
[Tab]公式[Tab](3.2)
```

写VBA脚本自动化这个。我的标题样式名：`标题 1,chapter`

```vb
Sub UpdateEquationNumbersOnly()
    Dim para As Paragraph
    Dim styleName As String
    Dim currentChapter As String
    Dim eqCount As Integer
    Dim headingCount As Integer
    Dim numberText As String

    currentChapter = ""
    eqCount = 1
    headingCount = 0

    For Each para In ActiveDocument.Paragraphs
        styleName = CStr(para.Style)

        ' 1. 识别一级标题
        If styleName = "标题 1,chapter" Then
            headingCount = headingCount + 1

            currentChapter = GetFirstNumber(para.Range.ListFormat.ListString)

            ' 如果标题没有自动编号，就按标题出现顺序作为章号
            If currentChapter = "" Then
                currentChapter = CStr(headingCount)
            End If

            eqCount = 1
        End If

        ' 2. 维护“公式”样式段落末尾编号
        If styleName = "公式" And currentChapter <> "" Then
            numberText = currentChapter & "." & eqCount

            If ReplaceEquationNumberByFind(para, numberText) Then
                eqCount = eqCount + 1
            End If
        End If
    Next para

    MsgBox "公式编号更新完成！"
End Sub


Function GetFirstNumber(ByVal s As String) As String
    Dim i As Integer
    Dim ch As String
    Dim result As String

    result = ""

    For i = 1 To Len(s)
        ch = Mid(s, i, 1)

        If ch >= "0" And ch <= "9" Then
            result = result & ch
        ElseIf result <> "" Then
            Exit For
        End If
    Next i

    GetFirstNumber = result
End Function


Function ReplaceEquationNumberByFind(ByVal para As Paragraph, ByVal numberText As String) As Boolean
    Dim r As Range

    ' 情况 1：已有编号，例如 Tab + (3.2)
    Set r = para.Range.Duplicate
    r.End = r.End - 1

    With r.Find
        .ClearFormatting
        .Replacement.ClearFormatting

        .Text = "^t\([0-9.]{1,}\)"
        .Replacement.Text = "^t(" & numberText & ")"

        .Forward = True
        .Wrap = wdFindStop
        .Format = False
        .MatchWildcards = True

        If .Execute(Replace:=wdReplaceOne) Then
            ReplaceEquationNumberByFind = True
            Exit Function
        End If
    End With

    ' 情况 2：空括号，例如 Tab + ()
    Set r = para.Range.Duplicate
    r.End = r.End - 1

    With r.Find
        .ClearFormatting
        .Replacement.ClearFormatting

        .Text = "^t()"
        .Replacement.Text = "^t(" & numberText & ")"

        .Forward = True
        .Wrap = wdFindStop
        .Format = False
        .MatchWildcards = False

        If .Execute(Replace:=wdReplaceOne) Then
            ReplaceEquationNumberByFind = True
            Exit Function
        End If
    End With

    ReplaceEquationNumberByFind = False
End Function
```

**图片加编号:** 这个就简单很多，下面是 prompt

写VBA宏自动找到docx里所有以图【xxx】 xxx这样格式的段落，修改样式为`图-caption`，并且把【xxx】替换成编号1.2其中1是章节号，每章标题样式`标题 1,chapter`首行缩进:  0 字符, 多级符号 + 级别: 1 + 编号样式: 1, 2, 3, … + 起始编号: 1 + 对齐方式: 左侧 + 对齐位置:  0 厘米 + 缩进位置:  0 厘米

下面这个宏会做三件事：

1. 遍历全文段落；
2. 遇到样式为 `标题 1,chapter` 的一级标题时，读取当前章节号；
3. 找到形如 `图【xxx】 xxx` 的段落，把样式改成 `图-caption`，并把 `【xxx】` 替换成 `章节号.图序号`，例如 `图【abc】 xxx` → `图【1.2】 xxx`。

```vb
Sub AutoNumberFigureCaptions()
    Dim para As Paragraph
    Dim styleName As String
    Dim currentChapter As String
    Dim figCount As Long
    Dim txt As String
    
    Dim re As Object
    Dim m As Object
    
    Set re = CreateObject("VBScript.RegExp")
    re.Global = False
    re.IgnoreCase = False
    
    ' 匹配：图【任意内容】 后面可以有空格和标题
    re.Pattern = "^图【[^】]*】"
    
    currentChapter = ""
    figCount = 0
    
    For Each para In ActiveDocument.Paragraphs
        styleName = para.Style
        
        ' 1. 识别一级标题，更新章节号
        If styleName = "标题 1,chapter" Then
            currentChapter = para.Range.ListFormat.ListString
            
            ' 防止读到类似 "第1章" 或 "1."，只保留数字部分
            currentChapter = ExtractFirstNumber(currentChapter)
            
            figCount = 0
        
        Else
            txt = para.Range.Text
            
            ' 去掉段落末尾的回车符
            If Len(txt) > 0 Then
                txt = Left(txt, Len(txt) - 1)
            End If
            
            ' 2. 匹配图【xxx】开头的段落
            If re.Test(txt) Then
                If currentChapter <> "" Then
                    figCount = figCount + 1
                    
                    ' 3. 设置样式
                    para.Style = ActiveDocument.Styles("图-caption")
                    
                    ' 4. 替换【xxx】里的内容为 章节号.图号
                    Set m = re.Execute(txt)(0)
                    
                    Dim r As Range
                    Set r = para.Range
                    r.End = r.Start + m.Length
                    
                    r.Text = "图【" & currentChapter & "." & figCount & "】"
                End If
            End If
        End If
    Next para
    
    MsgBox "图片题注编号更新完成。", vbInformation
End Sub


Function ExtractFirstNumber(ByVal s As String) As String
    Dim re As Object
    Dim ms As Object
    
    Set re = CreateObject("VBScript.RegExp")
    re.Global = False
    re.Pattern = "\d+"
    
    If re.Test(s) Then
        Set ms = re.Execute(s)
        ExtractFirstNumber = ms(0).Value
    Else
        ExtractFirstNumber = s
    End If
End Function
```

最后，似乎用 LaTeX 写然后闲鱼/淘宝买个 PDF 转 Word 的服务也可以......不过我没试过。
