# Windows 全新安装后设置

1. 设置硬件时钟为 UTC；
2. 关闭 Windows 时间自动同步（包括网络时间和自动设置时间）；
3. 手动与 NTP 服务器同步一次；
4. 关闭“快速启动”。

你可以将下面的内容保存为 `setup.ps1`，右键用管理员身份运行 PowerShell，然后执行此脚本。

```powershell
# 1. 设置硬件时钟为 UTC
Set-ItemProperty -Path "HKLM:\SYSTEM\CurrentControlSet\Control\TimeZoneInformation" -Name "RealTimeIsUniversal" -Value 1

# 2. 关闭 Windows 自动时间同步
Set-Service -Name w32time -StartupType Disabled
Stop-Service -Name w32time

# 3. 手动与 NTP 服务器同步一次
Set-Service -Name w32time -StartupType Manual
Start-Service -Name w32time
w32tm /resync

# 4. 关闭快速启动
$regPath = "HKLM:\SYSTEM\CurrentControlSet\Control\Session Manager\Power"
Set-ItemProperty -Path $regPath -Name "HiberbootEnabled" -Value 0

Write-Host "全部设置完成，请重启电脑以生效。"
```

**注意事项：**
- 需要以管理员权限运行 PowerShell。
- 第 1 步设置后，Windows 与 Linux 双系统可以共享 UTC 硬件时间。
- 第 2 步关闭了自动时间同步，如需恢复可将 `w32time` 服务设置为自动并启动。
- 第 3 步会手动与默认 NTP 服务器同步一次时间。
- 第 4 步关闭了快速启动，重启后生效。

如有需要调整 NTP 服务器，可在 `w32tm /config /manualpeerlist:"ntp.example.com" /update` 里更换服务器地址。