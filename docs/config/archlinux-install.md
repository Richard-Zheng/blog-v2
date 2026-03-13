# 安装 ArchLinux

LiveCD 下分区、挂载、pacstrap：

```bash
#!/bin/bash

# Variables
ESP=/dev/nvme0n1p1
ROOT=/dev/nvme0n1p2
SWAPFILESIZE=2G
# Optimization: Added discard=async for NVMe, removed redundant nodiratime
MOUNT_OPTIONS="noatime,compress=zstd,space_cache=v2,discard=async,ssd"

# 1. Format the partition
mkfs.btrfs -L ROOT $ROOT -f

# 2. Create Flat Subvolume Layout
mount $ROOT /mnt
btrfs sub create /mnt/@
btrfs sub create /mnt/@home
btrfs sub create /mnt/@pkg
btrfs sub create /mnt/@snapshots
btrfs sub create /mnt/@vm
btrfs sub create /mnt/@log
btrfs sub create /mnt/@swap
umount /mnt

# 3. Mount Root Subvolume
mount -o $MOUNT_OPTIONS,subvol=@ $ROOT /mnt

# 4. Create Mount Points
mkdir -p /mnt/{boot,home,var/cache/pacman/pkg,var/log,var/lib/libvirt,.snapshots,btrfs}

# 5. Mount Other Subvolumes
mount -o $MOUNT_OPTIONS,subvol=@home      $ROOT /mnt/home
mount -o $MOUNT_OPTIONS,subvol=@pkg       $ROOT /mnt/var/cache/pacman/pkg
mount -o $MOUNT_OPTIONS,subvol=@snapshots $ROOT /mnt/.snapshots
mount -o $MOUNT_OPTIONS,subvol=@log       $ROOT /mnt/var/log
mount -o $MOUNT_OPTIONS,subvolid=5        $ROOT /mnt/btrfs

# 6. Specialized Subvolume Mounting (NoCOW for VMs)
mount -o $MOUNT_OPTIONS,subvol=@vm        $ROOT /mnt/var/lib/libvirt
chattr +C /mnt/var/lib/libvirt # Disable Copy-on-Write for VM performance

# 7. Mount ESP
mount $ESP /mnt/boot

# 8. Create Swapfile on Btrfs
# btrfs-progs automatically handles NoCOW and compression exclusion for swapfiles
btrfs filesystem mkswapfile --size $SWAPFILESIZE --uuid clear /mnt/btrfs/@swap/swapfile
swapon /mnt/btrfs/@swap/swapfile

pacstrap -K /mnt base linux linux-firmware intel-ucode sof-firmware networkmanager man-db man-pages neovim grub efibootmgr zsh sudo
```

完成之后 chroot 进去

```bash
# Replace hostname with the name for your host
export HOST=hostname
# Replace Europe/London with your Region/City
export TZ="Asia/Shanghai"

echo $HOST > /etc/hostname
cat << EOF >> /etc/hosts
127.0.0.1 localhost
::1       localhost
127.0.1.1 $HOST.localdomain $HOST
EOF

ln -sf /usr/share/zoneinfo/$TZ /etc/localtime
# Sync system time to hardware clock
hwclock --systohc

# - set locale
echo "en_US.UTF-8 UTF-8" >> /etc/locale.gen
#echo "zh_CN.UTF-8 UTF-8" >> /etc/locale.gen
locale-gen
echo 'LANG=en_US.UTF-8'  > /etc/locale.conf

grub-install --target=x86_64-efi --efi-directory=/boot --bootloader-id=GRUB
grub-mkconfig -o /boot/grub/grub.cfg

systemctl enable NetworkManager
# enable ssh root login
#sed -i 's/PermitRootLogin prohibit-password/PermitRootLogin yes/g' /etc/ssh/sshd_config

passwd root
```

重启进入刚安装好的 ArchLinux

```bash
echo "export EDITOR=nvim" >> ~/.bash_profile
source ~/.bash_profile
useradd -m -G wheel -s /bin/bash frain
passwd frain
visudo

su frain
sudo pacman -S chezmoi
chezmoi init git@codeup.aliyun.com:6957efbbba2d7df023d9a849/dotfiles.git
chezmoi apply
```