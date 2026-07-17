# xv6 Shell

xv6 shell's source code is a good starting point for understanding xv6 and Unix's syscall interface design.

[sh.c - mit-pdos/xv6-riscv](https://github.com/mit-pdos/xv6-riscv/blob/riscv/user/sh.c)

Shell is essentially an interpreter. It takes user input, parses it, and use relevant system calls to finish the job. It behaves just like any  other normal user program.

We begin from `main()` function.

```c
int
main(void)
{
  static char buf[100];
  int fd;

  // Ensure that three file descriptors are open.
  while ((fd = open("console", O_RDWR)) >= 0) {
    if (fd >= 3) {
      close(fd);
      break;
    }
  }
  ...
}
```

The `while` loop is to ensure that the first three file descriptors (stdin, stdout, stderr) are open. The operating system assigns fd by increasing order. When fd 0, 1, 2 are accidentally closed, what the shell opens next might be assigned to fd 0, 1, or 2, leading to unexpected write/read behavior, which could be a security issue.

Then it reads a line from stdin, stores it in `buf` like [xxx\n\0], and strips the leading blank characters. If the command is `cd`, it calls `chdir()`, otherwise it does a `fork` syscall.

```c
int
main(void)
{
  ...
  // Read and run input commands.
  while (getcmd(buf, sizeof(buf)) >= 0) {
    char *cmd = buf;
    while (*cmd == ' ' || *cmd == '\t')
      cmd++;
    if (*cmd == '\n') // is a blank command
      continue;
    if (cmd[0] == 'c' && cmd[1] == 'd' && cmd[2] == ' ') {
      // Chdir must be called by the parent, not the child.
      cmd[strlen(cmd) - 1] = 0; // chop \n
      if (chdir(cmd + 3) < 0)
        fprintf(2, "cannot cd %s\n", cmd + 3);
    } else {
      if (fork1() == 0)
        runcmd(parsecmd(cmd));
      wait(0);
    }
  }
  exit(0);
}
```

In the child process, `fork` returns 0, and it calls `runcmd(parsecmd(cmd))` to parse the command and execute it. In the parent process, `fork` returns the child's pid, and it calls `wait(0)` to wait for the child process to finish.

So far, `cmd` is a string, `parsecmd(cmd)` turns it into a `struct cmd *` by parsing the string.