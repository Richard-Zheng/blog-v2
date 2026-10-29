// MMTLS 记录 / 握手解析（纯函数，无 DOM、无依赖，可在 SSR 下跑）。
// 输出是一棵「Wireshark 包详情」风格的树，每个节点带原始字节区间 [off, off+len)。

export const RECORD_TYPES = {
  0x15: 'Alert / Close',
  0x16: 'Handshake',
  0x17: 'Application Data',
  0x19: 'Early Data',
}

const HS_TYPES = {
  0: 'Alert',
  1: 'ClientHello',
  2: 'ServerHello',
  4: 'NewSessionTicket',
  8: 'EncryptedExtensions',
  15: 'Signature',
  20: 'Finished',
}

const CIPHERS = {
  0xc02b: 'TLS_ECDHE_ECDSA_WITH_AES_128_GCM_SHA256',
  0x00a8: 'TLS_PSK_WITH_AES_128_GCM_SHA256',
}

const EXT_TYPES = {
  0x0f: 'PreSharedKey',
  0x10: 'ClientKeyShare',
  0x11: 'ServerKeyShare',
  0x12: 'EarlyDataTimestamp',
}

const isVersion = (v) => v === 0xf103 || v === 0xf104

// ------------------------------------------------------------------ hex 输入

/**
 * 接受 Wireshark 的 Copy → as a Hex Dump / Hex Stream / Escaped String，
 * 以及 xxd -p、任意空白分隔。返回紧凑小写 hex。
 */
export function normalizeHex(input) {
  if (typeof input !== 'string') return ''
  const text = input.trim()
  if (!text) return ''

  if (/\\x[0-9a-fA-F]{2}/.test(text)) {
    let out = ''
    for (const m of text.matchAll(/\\x([0-9a-fA-F]{2})/g)) out += m[1]
    return out.toLowerCase()
  }

  let hex = ''
  for (const rawLine of text.split(/\r?\n/)) {
    let line = rawLine.trim()
    if (!line) continue
    // 去掉 "0000  " / "00000000: " 这类偏移列
    line = line.replace(/^(?:0x)?[0-9a-fA-F]{4,16}:\s+/, '').replace(/^(?:0x)?[0-9a-fA-F]{4,16}\s{2,}/, '')
    // hex 列与 ascii 列之间是 2 个以上空格
    const head = line.split(/\s{2,}/)[0]
    const compact = head.replace(/[\s:]/g, '').replace(/^0x/i, '')
    if (/^[0-9a-fA-F]*$/.test(compact)) hex += compact
  }
  return hex.toLowerCase()
}

const toBytes = (hex) => {
  const out = new Uint8Array(hex.length / 2)
  for (let i = 0; i < out.length; i++) out[i] = parseInt(hex.substr(i * 2, 2), 16)
  return out
}

export const hexOf = (u8) => Array.from(u8, (b) => b.toString(16).padStart(2, '0')).join('')

const asciiOf = (u8) =>
  Array.from(u8, (b) => (b >= 0x20 && b < 0x7f ? String.fromCharCode(b) : '.')).join('')

const be16 = (u8, i = 0) => (u8[i] << 8) | u8[i + 1]
const be32 = (u8, i = 0) =>
  ((u8[i] << 24) >>> 0) + (u8[i + 1] << 16) + (u8[i + 2] << 8) + u8[i + 3]

const clip = (bytes, max = 16) =>
  bytes.length <= max ? hexOf(bytes) : `${hexOf(bytes.subarray(0, max))}…`

const timeOf = (ts) => {
  const d = new Date(ts * 1000)
  return Number.isNaN(d.getTime()) ? '' : `${d.toISOString().replace('T', ' ').replace('Z', '')} UTC`
}

// ------------------------------------------------------------------ 游标

class Cur {
  constructor(u8, start, end) {
    if (start < 0 || end > u8.length || start > end) throw new Error('容器越界')
    this.buf = u8
    this.pos = start
    this.end = end
  }

  remaining() {
    return this.end - this.pos
  }

  take(n) {
    if (n < 0 || n > this.remaining()) {
      throw new Error(`字段越界：需要 ${n}B，剩 ${this.remaining()}B`)
    }
    const off = this.pos
    this.pos += n
    return { off, len: n, bytes: this.buf.subarray(off, off + n) }
  }

  u8() {
    return this.take(1).bytes[0]
  }

  u16be() {
    return be16(this.take(2).bytes)
  }

  u16le() {
    const b = this.take(2).bytes
    return b[0] | (b[1] << 8)
  }

  u32be() {
    return be32(this.take(4).bytes)
  }
}

// 节点：label + value + 可选字节区间；gen = Wireshark 里那种 [方括号] 生成字段
const mk = (label, { value = '', span, gen = false, children = null, mono = false, crlf = false } = {}) => ({
  label,
  value,
  off: span ? span.off : undefined,
  len: span ? span.len : undefined,
  gen,
  mono,
  crlf,
  children,
})

// ------------------------------------------------------------------ HTTP

const HTTP_REQ = /^(?:GET|POST|PUT|HEAD|DELETE|CONNECT|OPTIONS|TRACE|PATCH) \S+ HTTP\/1\.[01]/
const HTTP_RESP = /^HTTP\/1\.[01] \d{3}/

function looksLikeHttp(u8, i) {
  if (i + 16 > u8.length) return false
  const head = String.fromCharCode(...u8.subarray(i, Math.min(u8.length, i + 96)))
  return HTTP_REQ.test(head) || HTTP_RESP.test(head)
}

function parseHttp(u8, start) {
  const children = []
  let p = start
  let firstLine = ''
  // 首行 + 头，逐行给区间
  for (;;) {
    let eol = p
    while (eol + 1 < u8.length && !(u8[eol] === 13 && u8[eol + 1] === 10)) eol++
    if (eol + 1 >= u8.length) return null
    const lineEnd = eol + 2
    const text = String.fromCharCode(...u8.subarray(p, eol))
    const span = { off: p, len: lineEnd - p }
    if (p === start) {
      firstLine = text
      children.push(mk(text, { span, mono: true, crlf: true }))
    } else if (text === '') {
      children.push(mk('\\r\\n', { span }))
      p = lineEnd
      break
    } else {
      const idx = text.indexOf(':')
      children.push(mk(idx > 0 ? text.slice(0, idx) : text, {
        value: idx > 0 ? text.slice(idx + 1).trim() : '',
        span,
        crlf: true,
      }))
    }
    p = lineEnd
  }
  const bodyStart = p

  // 从首行/头里凑一个 URI
  const host = (() => {
    const h = children.find((c) => c.label.toLowerCase() === 'host')
    return h ? h.value : ''
  })()
  const target = firstLine.split(' ')[1] || ''
  if (host && target.startsWith('/')) {
    children.push(mk(`Full request URI: http://${host}${target}`, { gen: true }))
  }
  const lenHeader = children.find((c) => c.label.toLowerCase() === 'content-length')
  const declared = lenHeader ? parseInt(lenHeader.value, 10) : NaN
  const bodyLen = Number.isFinite(declared) ? declared : u8.length - bodyStart
  children.push(mk('File Data', { value: `${bodyLen} bytes`, span: { off: bodyStart, len: bodyLen } }))

  return {
    node: mk('Hypertext Transfer Protocol', { children, span: { off: start, len: bodyStart - start } }),
    bodyStart,
  }
}

// ------------------------------------------------------------------ 握手

function pskNodes(c) {
  const out = []
  const t = c.take(1)
  const type = t.bytes[0]
  out.push(mk('PSK Type', {
    value: type === 1 ? 'Access (Linux) (1)' : type === 2 ? 'Refresh (Linux) (2)' : String(type),
    span: t,
  }))
  const life = c.take(4)
  out.push(mk('PSK Lifetime (seconds)', { value: String(be32(life.bytes)), span: life }))
  const idLen = c.take(2)
  out.push(mk('PSK Identity Length', { value: String(be16(idLen.bytes)), span: idLen }))
  if (be16(idLen.bytes) > 0) {
    const id = c.take(be16(idLen.bytes))
    out.push(mk('PSK Identity / MAC', { value: clip(id.bytes), span: id, mono: true }))
  }
  const age = c.take(4)
  out.push(mk('PSK Key Version / Age Slot', { value: String(be32(age.bytes)), span: age }))
  const nLen = c.take(2)
  out.push(mk('Ticket Nonce Length', { value: String(be16(nLen.bytes)), span: nLen }))
  if (be16(nLen.bytes) > 0) {
    const nonce = c.take(be16(nLen.bytes))
    out.push(mk('Ticket Nonce', { value: hexOf(nonce.bytes), span: nonce, mono: true }))
  }
  const tkLen = c.take(2)
  out.push(mk('Session Ticket Length', { value: String(be16(tkLen.bytes)), span: tkLen }))
  if (be16(tkLen.bytes) > 0) {
    const tk = c.take(be16(tkLen.bytes))
    out.push(mk('Session Ticket (opaque, not the PSK secret)', {
      value: clip(tk.bytes), span: tk, mono: true,
    }))
  }
  if (c.remaining() > 0) {
    const tail = c.take(c.remaining())
    out.push(mk('Trailing', { value: clip(tail.bytes), span: tail, gen: true, mono: true }))
  }
  return out
}

function publicKeyNodes(c, prefix) {
  const kN = c.take(2)
  const n = be16(kN.bytes)
  const out = [mk(`${prefix} Public Key Length`, { value: String(n), span: kN })]
  const key = c.take(n)
  const kids = []
  if (n === 65 && key.bytes[0] === 4) {
    kids.push(mk('EC Point Format', { value: 'Uncompressed (0x04)', span: { off: key.off, len: 1 } }))
    kids.push(mk('EC Point X', { value: hexOf(key.bytes.subarray(1, 33)), span: { off: key.off + 1, len: 32 }, mono: true }))
    kids.push(mk('EC Point Y', { value: hexOf(key.bytes.subarray(33, 65)), span: { off: key.off + 33, len: 32 }, mono: true }))
  }
  out.push(mk(`${prefix} Public Key`, { value: kids.length ? '' : clip(key.bytes), span: key, mono: true, children: kids.length ? kids : null }))
  return out
}

function extensionNodes(type) {
  const out = []
  const handle = (C) => {
    if (type === 0x000f) {
      const f = C.take(1)
      out.push(mk('PSK Presence Flag', { value: String(f.bytes[0]), span: f }))
      if (f.bytes[0] === 1) {
        const l = C.take(4)
        out.push(mk('Serialized PSK Length', { value: String(be32(l.bytes)), span: l }))
        const body = C.take(be32(l.bytes))
        out.push(mk('Pre-Shared Key Ticket', { span: body, children: pskNodes(new Cur(C.buf, body.off, body.off + body.len)) }))
      }
    } else if (type === 0x0010) {
      const n = C.take(1)
      out.push(mk('Key Share Count', { value: String(n.bytes[0]), span: n }))
      for (let i = 0; i < n.bytes[0]; i++) {
        const g = C.take(4)
        const v = C.take(4)
        const kids = [
          mk('Key Group', { value: `P-256 (0x${be32(g.bytes).toString(16)})`, span: g }),
          mk('Key Version / Selection ID', { value: String(be32(v.bytes)), span: v }),
          ...publicKeyNodes(C, 'Key'),
        ]
        out.push(mk(`Key Share ${i + 1}`, { children: kids }))
      }
    } else if (type === 0x0011) {
      const v = C.take(4)
      out.push(mk('Key Version / Selection ID', { value: String(be32(v.bytes)), span: v }))
      out.push(...publicKeyNodes(C, 'Server'))
    } else if (type === 0x0012) {
      const t = C.take(4)
      out.push(mk('Early Data Timestamp', { value: `${be32(t.bytes)} (${timeOf(be32(t.bytes))})`, span: t }))
    }
    if (C.remaining() > 0) {
      const tail = C.take(C.remaining())
      out.push(mk('Data', { value: clip(tail.bytes), span: tail, gen: type === 0x000f, mono: true }))
    }
  }
  return { out, handle }
}

function extensionBlockNodes(c) {
  const out = []
  const blk = c.take(4)
  const total = be32(blk.bytes)
  out.push(mk('Extension Block Length', { value: String(total), span: blk }))
  const blockSpan = c.take(total)
  const start = blockSpan.off
  const block = mk('Hello Extensions', { span: blockSpan, children: [] })
  const inner = new Cur(c.buf, start, start + total)
  const cnt = inner.take(1)
  block.children.push(mk('Extension Count', { value: String(cnt.bytes[0]), span: cnt }))
  for (let i = 0; i < cnt.bytes[0]; i++) {
    const lenF = inner.take(4)
    const n = be32(lenF.bytes)
    const extStart = inner.pos
    const body = inner.take(n) // 长度包含 2 字节 type
    const ec = new Cur(c.buf, extStart, extStart + n)
    const tp = ec.take(2)
    const type = be16(tp.bytes)
    const name = EXT_TYPES[type] || `0x${type.toString(16).padStart(4, '0')}`
    const kids = [
      mk('Extension Length', { value: String(n), span: lenF }),
      mk('Extension Type', { value: `${name} (0x${type.toString(16).padStart(4, '0')})`, span: tp }),
    ]
    const handler = extensionNodes(type)
    try {
      handler.handle(ec)
      kids.push(...handler.out)
    } catch (e) {
      kids.push(mk(`解析中断：${e.message}`, { gen: true }))
    }
    block.children.push(mk(`Extension ${i + 1}: ${name}`, {
      span: { off: lenF.off, len: 4 + n }, children: kids,
    }))
  }
  if (inner.remaining() > 0) {
    const tail = inner.take(inner.remaining())
    block.children.push(mk('Extensions Trailing', { value: clip(tail.bytes), span: tail, gen: true, mono: true }))
  }
  out.push(block)
  return out
}

function helloNodes(c, isClient) {
  const out = []
  const lenF = c.take(4)
  out.push(mk('Handshake Message Length', { value: String(be32(lenF.bytes)), span: lenF }))
  const tp = c.take(1)
  out.push(mk('Handshake Type', { value: `${HS_TYPES[tp.bytes[0]]} (0x${tp.bytes[0].toString(16).padStart(2, '0')})`, span: tp }))
  const ver = c.take(2)
  const innerVer = (ver.bytes[0] | (ver.bytes[1] << 8)).toString(16)
  out.push(mk('Inner Version (little endian)', { value: `MMTLS 0x${innerVer} (0x${innerVer})`, span: ver }))
  if (isClient) {
    const cc = c.take(1)
    out.push(mk('Cipher Suite Count', { value: String(cc.bytes[0]), span: cc }))
    for (let i = 0; i < cc.bytes[0]; i++) {
      const cs = c.take(2)
      const v = be16(cs.bytes)
      out.push(mk(`Cipher Suite${cc.bytes[0] > 1 ? ` ${i + 1}` : ''}`, {
        value: `${CIPHERS[v] || 'unknown'} (0x${v.toString(16).padStart(4, '0')})`,
        span: cs,
      }))
    }
  } else {
    const cs = c.take(2)
    const v = be16(cs.bytes)
    out.push(mk('Cipher Suite', {
      value: `${CIPHERS[v] || 'unknown'} (0x${v.toString(16).padStart(4, '0')})`,
      span: cs,
    }))
  }
  const rnd = c.take(32)
  out.push(mk('Hello Random (32 bytes)', { value: clip(rnd.bytes), span: rnd, mono: true }))
  if (isClient) {
    const ts = c.take(4)
    const v = be32(ts.bytes)
    out.push(mk('Client Unix Timestamp', { value: String(v), span: ts }))
    out.push(mk('Client Time', { value: timeOf(v), span: ts }))
  }
  out.push(...extensionBlockNodes(c))
  if (c.remaining() > 0) {
    const tail = c.take(c.remaining())
    out.push(mk('Trailing', { value: clip(tail.bytes), span: tail, gen: true, mono: true }))
  }
  return out
}

function otherHandshakeNodes(c, direction) {
  const out = []
  const lenF = c.take(4)
  out.push(mk('Handshake Message Length', { value: String(be32(lenF.bytes)), span: lenF }))
  const tp = c.take(1)
  const type = tp.bytes[0]
  let name = HS_TYPES[type] || `unknown (${type})`
  if (type === 20 && direction) name = direction === 's2c' ? 'ServerFinished' : 'ClientFinished'
  out.push(mk('Handshake Type', { value: `${name} (0x${type.toString(16).padStart(2, '0')})`, span: tp }))
  if (type === 4) {
    const cnt = c.take(1)
    out.push(mk('Session Ticket Count', { value: String(cnt.bytes[0]), span: cnt }))
    for (let i = 0; i < cnt.bytes[0]; i++) {
      const l = c.take(4)
      out.push(mk(`Session Ticket Length ${i + 1}`, { value: String(be32(l.bytes)), span: l }))
      const body = c.take(be32(l.bytes))
      out.push(mk(`Session Ticket ${i + 1}`, {
        span: body, children: pskNodes(new Cur(c.buf, body.off, body.off + body.len)),
      }))
    }
  } else if (type === 8) {
    out.push(...extensionBlockNodes(c))
  } else if (type === 15) {
    const l = c.take(2)
    out.push(mk('Signature Length', { value: String(be16(l.bytes)), span: l }))
    const sig = c.take(be16(l.bytes))
    out.push(mk('Signature (DER ECDSA)', { value: clip(sig.bytes), span: sig, mono: true }))
  } else if (type === 20) {
    const l = c.take(2)
    out.push(mk('Verify Data Length', { value: String(be16(l.bytes)), span: l }))
    const v = c.take(be16(l.bytes))
    out.push(mk('Verify Data', { value: clip(v.bytes), span: v, mono: true }))
  } else if (type === 0) {
    const d = c.take(1)
    out.push(mk('Alert Description', {
      value: `${d.bytes[0]}${d.bytes[0] === 1 ? ' (end of early data)' : ''}`, span: d,
    }))
    const lv = c.take(1)
    out.push(mk('Alert Level', {
      value: `${lv.bytes[0]}${lv.bytes[0] === 1 ? ' (warning)' : lv.bytes[0] === 2 ? ' (fatal)' : ''}`,
      span: lv,
    }))
  }
  if (c.remaining() > 0) {
    const tail = c.take(c.remaining())
    out.push(mk('Trailing', { value: clip(tail.bytes), span: tail, gen: true, mono: true }))
  }
  return out
}

function interpretHandshake(u8, start, end, direction) {
  if (end - start < 5) return null
  const declared = be32(u8, start)
  if (declared !== end - start - 4) return null
  const type = u8[start + 4]
  if (type !== 1 && type !== 2 && HS_TYPES[type] === undefined) return null
  const c = new Cur(u8, start, end)
  try {
    if (type === 1 || type === 2) {
      if (end - start < 11) return null
      const ver = u8[start + 5] | (u8[start + 6] << 8)
      if (!isVersion(ver)) return null
      return { name: type === 1 ? 'ClientHello' : 'ServerHello', nodes: helloNodes(c, type === 1) }
    }
    return { name: type === 8 ? 'EncryptedExtensions' : HS_TYPES[type], nodes: otherHandshakeNodes(c, direction) }
  } catch (e) {
    return { name: `${HS_TYPES[type] || 'Handshake'} (解析中断)`, nodes: [mk(`[${e.message}]`, { gen: true })] }
  }
}

// ------------------------------------------------------------------ 记录

const isHeader = (u8, i) =>
  i + 5 <= u8.length &&
  RECORD_TYPES[u8[i]] !== undefined &&
  isVersion((u8[i + 1] << 8) | u8[i + 2])

function recordNode(u8, i, direction) {
  const type = u8[i]
  const ver = be16(u8, i + 1)
  const len = be16(u8, i + 3)
  if (i + 5 + len > u8.length) return { node: null, next: u8.length }

  const hdr = { off: i, len: 5 }
  const body = { off: i + 5, len }
  const inner = (type === 0x16 || type === 0x19 || type === 0x15)
    ? interpretHandshake(u8, i + 5, i + 5 + len, direction)
    : null

  const headerKids = [
    mk('Record Length (including header)', { value: String(len + 5), gen: true, span: { off: i + 3, len: 2 } }),
    mk('Record Version', { value: `MMTLS 0x${ver.toString(16)}`, gen: true, span: { off: i + 1, len: 2 } }),
  ]
  const kids = [
    mk('Record Header', { value: hexOf(u8.subarray(i, i + 5)), span: hdr, mono: true, children: headerKids }),
  ]
  if (inner) {
    headerKids.push(mk('Body Interpretation: Plaintext ' + inner.name, { gen: true, span: body }))
    kids.push(mk(inner.name, { span: body, children: inner.nodes }))
  } else {
    headerKids.push(mk(
      len >= 16 ? `Body: ${len - 16} B ciphertext + 16 B GCM tag` : 'Body: opaque',
      { gen: true, span: body },
    ))
  }

  const label = RECORD_TYPES[type]
  const suffix = inner ? `${label} / ${inner.name}` : label
  const node = mk('MMTLS Record', { value: suffix, span: { off: i, len: 5 + len }, children: kids })
  node.recType = type
  return { node, next: i + 5 + len }
}

// ------------------------------------------------------------------ 入口

export function parseMmtls(input, opts = {}) {
  const direction = opts.direction === 's2c' ? 's2c' : 'c2s'
  const hex = normalizeHex(input)
  if (!hex) return { error: '没有识别到 hex 内容', nodes: [] }
  if (hex.length % 2) return { error: `hex 长度为奇数（${hex.length} 个字符）`, nodes: [] }
  const u8 = toBytes(hex)

  // 跳过 Frame/Ethernet/IPv4/TCP 之类前缀：找最早的 HTTP 起始或 MMTLS 记录头
  let start = -1
  const httpWindow = Math.min(u8.length, 8192)
  for (let i = 0; i < u8.length; i++) {
    if (isHeader(u8, i)) {
      start = i
      break
    }
    if (i < httpWindow) {
      const c = u8[i]
      // 'G' 'P' 'H' 'D' 'O' 'T' 'C' 才可能是方法/响应首字母
      if ((c === 0x47 || c === 0x50 || c === 0x48 || c === 0x44 || c === 0x4f || c === 0x54 || c === 0x43) &&
          looksLikeHttp(u8, i)) {
        start = i
        break
      }
    }
  }
  if (start < 0) {
    return { error: '没找到 HTTP 起始行或 MMTLS 记录头（应形如 19 f1 04 .. 或 "POST /mmtls/.. HTTP/1.1"）', nodes: [] }
  }

  const nodes = []
  let p = start
  if (looksLikeHttp(u8, p)) {
    const http = parseHttp(u8, p)
    if (http) {
      nodes.push(http.node)
      p = http.bodyStart
    }
  }
  while (p + 5 <= u8.length) {
    if (isHeader(u8, p)) {
      const { node, next } = recordNode(u8, p, direction)
      if (node) nodes.push(node)
      p = next > p ? next : p + 1
      continue
    }
    p++
  }

  if (nodes.length === 0) return { error: '没解析出任何记录', nodes: [] }
  return { nodes, total: u8.length, skipped: start, bytes: u8 }
}

export const hexDump = (u8, baseOffset = 0) => {
  const lines = []
  for (let i = 0; i < u8.length; i += 16) {
    const chunk = u8.subarray(i, i + 16)
    lines.push({
      off: baseOffset + i,
      hex: Array.from(chunk, (b) => b.toString(16).padStart(2, '0')).join(' '),
      ascii: asciiOf(chunk),
    })
  }
  return lines
}

export { asciiOf }
