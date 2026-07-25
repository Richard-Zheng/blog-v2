# xv6 Shell

xv6 shell's source code is a good starting point for understanding xv6 and Unix's syscall interface design.

[sh.c - mit-pdos/xv6-riscv](https://github.com/mit-pdos/xv6-riscv/blob/riscv/user/sh.c)

Shell is essentially an interpreter. It takes user input, parses it, and use relevant system calls to finish the job. It behaves just like any  other normal user program.

## main function

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

The `while` loop is to ensure that the first three file descriptors (stdin, stdout, stderr) are open. The operating system assigns fd using **next minimal available number**. When fd 0, 1, 2 are accidentally closed, what the shell opens next might be assigned to fd 0, 1, or 2, leading to unexpected read/write behavior, which could be a security issue.

Then it reads a line from stdin, stores it in `buf` like `['l', 's', '\n', '\0']`, and strips the leading blank characters. If the command is `cd`, it calls `chdir()`, otherwise it does a `fork` syscall.

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

## Parsing

So far, `cmd` is a string, `parsecmd(cmd)` turns it into a `struct cmd *` by parsing the string.

It uses the following EBNF grammar:

```
line        ::= pipe { "&" } [ ";" line ] ;

pipe        ::= exec [ "|" pipe ] ;

exec        ::= block
              | { redirection } { WORD { redirection } } ;

block       ::= "(" line ")" { redirection } ;

redirection ::= "<" WORD
              | ">" WORD
              | ">>" WORD ;
```

in which:

```
{ X }  represents zero or more occurrences of X
[ X ]  represents zero or one occurrence of X
|      represents choice
```

FIRST sets:

```
FIRST(redirection) = { "<", ">", ">>" }

FIRST(block)       = { "(" }

FIRST(exec)        = { WORD, "<", ">", ">>", "(" }

FIRST(pipe)        = FIRST(exec)

FIRST(line)        = FIRST(pipe)
```

## Evaluation

After parsing we get a `struct cmd *` AST, then we feed it into `runcmd(struct cmd *cmd)` to traverse the AST. Most of the `cmd` types are obvious, but `PIPE` is a bit tricky.

```c
// Execute cmd.  Never returns.
void
runcmd(struct cmd *cmd)
{
  int p[2];
  struct pipecmd *pcmd;
  ...

  switch (cmd->type) {
  default:
    panic("runcmd");

  ...

  case PIPE:
    pcmd = (struct pipecmd *)cmd;
    if (pipe(p) < 0)
      panic("pipe");
    if (fork1() == 0) {
      close(1);
      dup(p[1]);
      close(p[0]);
      close(p[1]);
      runcmd(pcmd->left);
    }
    if (fork1() == 0) {
      close(0);
      dup(p[0]);
      close(p[0]);
      close(p[1]);
      runcmd(pcmd->right);
    }
    close(p[0]);
    close(p[1]);
    wait(0);
    wait(0);
    break;
  }
  exit(0);
}
```

`pipe(p)` creates two file descriptors:

```
p[0]: read end
p[1]: write end
```

You can think of it as:

```
write to p[1] -----> read from p[0]
```

For example:

```
cat file | grep hello
```

What we need to implement is:

```
cat stdout (fd 1) -> pipe -> grep stdin (fd 0)
```

### 1. Create the left-side process

```c
if (fork1() == 0) {
    close(1);
    dup(p[1]);

    close(p[0]);
    close(p[1]);

    runcmd(pcmd->left);
}
```

The key part is:

```c
close(1);
dup(p[1]);
```

`dup(oldfd)` duplicates a file descriptor and uses the smallest available fd.

After closing fd 1:

```c
dup(p[1]);
```

it returns 1.

So now:

```text
fd 1 -> pipe write end
```

When the left command later calls:

```c
write(1, ...)
```

the data goes into the pipe.

Then:

```c
close(p[0]);
close(p[1]);
```

closes the original pipe descriptors.

Even though fd 1 still points to the same pipe write end, the original `p[1]` is no longer needed.

Finally:

```c
runcmd(pcmd->left);
```

executes the left command.

### 2. Create the right-side process

```c
if (fork1() == 0) {
    close(0);
    dup(p[0]);

    close(p[0]);
    close(p[1]);

    runcmd(pcmd->right);
}
```

Here:

```c
close(0);
dup(p[0]);
```

So:

```text
fd 0 -> pipe read end
```

When the right program calls:

```c
read(0, ...)
```

it reads the left side's output from the pipe.

### 3. Parent process closes the pipe

```c
close(p[0]);
close(p[1]);
```

This cannot be omitted.

The parent process does not participate in data transfer, so it must close both ends.

Especially the write end `p[1]`.

The condition for the pipe read end to receive EOF is:

> all file descriptors pointing to that pipe write end must be closed.

Suppose the left command exits, but the parent still keeps `p[1]` open:

```text
left write end is closed
parent write end is still open
```

Then the right side still thinks more data may come, so:

```c
read(...)
```

it may keep waiting and never receive EOF.

### 4. Wait for both child processes

```c
wait(0);
wait(0);
```

Because two child processes were created:

```text
left process
right process
```

we need to wait twice.

The full process relationship is:

```text
PIPE execution process
│
├── left child process
│   ├── stdout -> pipe write
│   └── exec left command
│
├── right child process
│   ├── stdin <- pipe read
│   └── exec right command
│
├── close pipe read/write
├── wait for left/right
└── exit
```
