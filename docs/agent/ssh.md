# SSH Remote

## Problem

When I ask Codex to log in my server and check something for me, it did this:

```js
const r = await tools.exec_command({cmd:"ssh -o ConnectTimeout=10 myvps-xxx.top 'echo \"=== ACTIVE SERVICES ===\"; systemctl list-units --type=service --state=running --no-pager --no-legend; echo; echo \"=== LISTENING SOCKETS ===\"; ss -tulpn","workdir":"/home/frain/blog-v2","yield_time_ms":15000,"max_output_tokens":10000,"sandbox_permissions":"require_escalated","justification":"May I inspect active services and listening ports over SSH?","prefix_rule":["ssh"]});
text(r.output);
```

**Output**

```
Script completed
Wall time 26.9 seconds
Output:
zsh:1: unmatched '
```

And it tried once more, this time get it right. But I don't want to waste tokens on this.

So I wrote a `ssh` wrapper to `~/.local/bin/ssh-run`

```bash
#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "usage: ssh-run HOST [ssh options...]" >&2
  exit 2
fi

exec ssh \
  -o ConnectTimeout=10 \
  "$@" \
  'bash -se'
```

And added the following to `~/.codex/AGENTS.md`

```
## SSH / remote command execution

For non-interactive remote shell commands, prefer `ssh-run` instead of
embedding the remote script in an `ssh` command-line argument.

Usage:

    ssh-run [SSH_OPTIONS...] destination <<'EOF'
    remote commands...
    EOF

SSH options are passed through directly to `ssh`, so options such as
`-p`, `-i`, `-J`, `-4`, and `-o ...` may be used when necessary.

Do not put substantial remote shell scripts directly in an SSH
command argument like:

    ssh HOST 'command1; command2; ...'

Use plain `ssh` when you specifically need SSH features such as
interactive sessions, tunneling, port forwarding, or `-N`.
```