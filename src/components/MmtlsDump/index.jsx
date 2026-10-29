import React, { useDeferredValue, useMemo, useState } from 'react'
import { parseMmtls, hexDump, hexOf } from './parse'
import styles from './styles.module.css'

function Line({ node, hover, onHover, bytes }) {
  const hasSpan = node.off !== undefined && node.len > 0
  const hovering = hover && hover.id === node.id
  return (
    <span
      className={styles.line}
      onMouseEnter={hasSpan ? () => onHover(node) : undefined}
      onMouseLeave={hasSpan ? () => onHover(null) : undefined}
    >
      {node.gen ? (
        <span className={styles.gen}>
          [{node.label}
          {node.value ? `: ${node.value}` : ''}]
        </span>
      ) : (
        <>
          <span className={styles.label}>{node.label}</span>
          {node.value ? (
            <>
              <span className={styles.colon}>: </span>
              <span className={node.mono ? styles.mono : styles.value}>{node.value}</span>
              {node.crlf && <span className={styles.crlf}>\r\n</span>}
            </>
          ) : null}
        </>
      )}
      {hovering && hasSpan && (
        <span className={styles.chip}>
          {bytes ? hexOf(bytes.subarray(node.off, node.off + Math.min(node.len, 12))) : ''}
          {node.len > 12 ? '…' : ''}
          <span className={styles.chipRange}>@{node.off}</span>
        </span>
      )}
    </span>
  )
}

function HexView({ bytes, base, hover }) {
  const MAX = 4096
  const shown = bytes.length > MAX ? bytes.subarray(0, MAX) : bytes
  const lines = useMemo(() => hexDump(shown, base), [shown, base])
  const big = bytes.length > 4096
  const from = hover ? hover.off : -1
  const to = hover ? hover.off + hover.len : -1
  return (
    <details open className={styles.hexbox}>
      <summary>原始字节（{bytes.length} B）</summary>
      <pre className={styles.dumppre}>
        {lines.map((l) => (
          <div key={l.off} className={styles.dumpline}>
            <span className={styles.offset}>{l.off.toString(16).padStart(4, '0')}</span>
            {big ? (
              <span className={styles.hex}>{l.hex}</span>
            ) : (
              <span className={styles.hex}>
                {Array.from(bytes.subarray(l.off - base, l.off - base + 16), (b, i) => {
                  const abs = l.off + i
                  const on = abs >= from && abs < to
                  return (
                    <span key={i} className={on ? styles.hl : undefined}>
                      {b.toString(16).padStart(2, '0')}
                      {i < 15 ? ' ' : ''}
                    </span>
                  )
                })}
              </span>
            )}
            <span className={styles.ascii}>{l.ascii}</span>
          </div>
        ))}
        {bytes.length > MAX && (
          <div className={styles.more}>… 只显示前 {MAX} 字节（共 {bytes.length} B）</div>
        )}
      </pre>
    </details>
  )
}

function Node({ node, level, ctx, isTop }) {
  const padding = { paddingLeft: `${level * 14 + 14}px` }
  const children = node.children || []

  const row = (
    <span className={styles.row} style={padding}>
      <span className={styles.spacer} />
      <Line node={node} hover={ctx.hover} onHover={ctx.setHover} bytes={ctx.bytes} />
    </span>
  )

  if (children.length === 0) {
    return <div className={styles.leaf}>{row}</div>
  }

  return (
    <details
      open
      className={styles.branch}
      data-rec={isTop && node.recType !== undefined ? node.recType.toString(16) : undefined}
    >
      <summary className={styles.summary} style={padding}>
        <Line node={node} hover={ctx.hover} onHover={ctx.setHover} bytes={ctx.bytes} />
      </summary>
      <div className={styles.kids}>
        {children.map((child, i) => (
          <Node key={`${child.label}-${i}`} node={child} level={level + 1} ctx={ctx} />
        ))}
        {isTop && node.off !== undefined && ctx.bytes && (
          <div style={{ paddingLeft: '14px' }}>
            <HexView bytes={ctx.bytes.subarray(node.off, node.off + node.len)} base={node.off} hover={ctx.hover} />
          </div>
        )}
      </div>
    </details>
  )
}

/**
 * 把 Wireshark 复制出来的 hex 解析成「包详情」树。
 *   <MmtlsDump initial={hexText} caption="..." direction="c2s" />
 */
export default function MmtlsDump({ children, hex, initial, direction, caption }) {
  const start = typeof initial === 'string' ? initial : typeof hex === 'string' ? hex : typeof children === 'string' ? children : ''
  const [text, setText] = useState(start)
  const [hover, setHover] = useState(null)

  const deferred = useDeferredValue(text)
  const parsed = useMemo(
    () => parseMmtls(deferred, { direction: direction === 's2c' ? 's2c' : 'c2s' }),
    [deferred, direction],
  )

  // 给每个节点一个 id（用于 hover 高亮判定）
  const nodes = useMemo(() => {
    let n = 0
    const walk = (list) => list.map((node) => {
      const copy = { ...node, id: `${n++}` }
      if (node.children) copy.children = walk(node.children)
      return copy
    })
    return parsed.nodes ? walk(parsed.nodes) : []
  }, [parsed])

  const ctx = { hover, setHover, bytes: parsed.bytes }

  return (
    <figure className={styles.wrap}>
      {caption && <figcaption className={styles.caption}>{caption}</figcaption>}

      <div className={styles.toolbar}>
        <textarea
          className={styles.input}
          value={text}
          spellCheck={false}
          onChange={(e) => setText(e.target.value)}
          placeholder={'把 Wireshark 里的 hex 粘到这里：\nCopy → as a Hex Dump / as Hex Stream / as Escaped String 都可以'}
        />
        <div className={styles.actions}>
          <button type="button" className={styles.btn} onClick={() => setText(start)}>示例数据</button>
          <button type="button" className={styles.btn} onClick={() => setText('')}>清空</button>
          <span className={styles.hint}>
            自动跳过 Frame / Ethernet / IPv4 / TCP 前缀；鼠标停在字段上会高亮下面的原始字节
          </span>
        </div>
      </div>

      {parsed.error ? (
        <p className={styles.error}>无法解析：{parsed.error}</p>
      ) : (
        <>
          <p className={styles.summary}>
            {nodes.length} 个顶层节点 / {parsed.total} 字节
            {parsed.skipped > 0 && `（跳过前置 ${parsed.skipped} 字节）`}
          </p>
          <div className={styles.tree}>
            {nodes.map((node) => (
              <Node key={node.id} node={node} level={0} ctx={ctx} isTop />
            ))}
          </div>
        </>
      )}
    </figure>
  )
}
