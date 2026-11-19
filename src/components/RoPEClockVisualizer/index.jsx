import React, {useId, useState} from 'react';
import styles from './styles.module.css';

const DIMENSION = 16;
const BASE = 10000;
const MAX_POSITION = 10000;
const TAU = 2 * Math.PI;

export default function RoPEClockVisualizer() {
  const [position, setPosition] = useState(0);
  const sliderId = useId();

  return (
    <div className="card margin-vert--lg">
      <div className="card__header">
        <h3>RoPE 的旋转时钟</h3>
        <p className="margin-bottom--none">
          d<sub>h</sub> = {DIMENSION}，base = {BASE}，每个时钟对应一对通道。
        </p>
      </div>
      <div className="card__body">
        <div className={styles.controls}>
          <label htmlFor={sliderId}>
            位置 m = <strong>{position}</strong>
          </label>
          <button
            type="button"
            className="button button--secondary button--outline button--sm"
            onClick={() => setPosition(0)}>
            重置
          </button>
        </div>
        <input
          id={sliderId}
          className={styles.slider}
          type="range"
          min={0}
          max={MAX_POSITION}
          step={1}
          value={position}
          onChange={(event) => setPosition(Number(event.target.value))}
        />
        <div className={`${styles.controls} margin-bottom--md`} aria-hidden="true">
          <small>0</small>
          <small>{MAX_POSITION}</small>
        </div>
        <div className={styles.grid}>
          {Array.from({length: DIMENSION / 2}, (_, index) => {
            const frequency = BASE ** (-2 * index / DIMENSION);
            const angle = position * frequency;
            const phase = angle % TAU;
            const turns = angle / TAU;
            return (
              <div key={index} className="text--center">
                <strong>i = {index}</strong>
                <svg
                  className={styles.clock}
                  viewBox="0 0 120 120"
                  role="img"
                  aria-label={`第 ${index} 个时钟：旋转 ${turns.toFixed(3)} 圈，圈内角度 ${(phase / Math.PI).toFixed(3)} π`}>
                  <circle className={styles.face} cx="60" cy="60" r="48" />
                  <path className={styles.axis} d="M12 60H108 M60 12V108" />
                  <text className={styles.zero} x="112" y="64">0</text>
                  <line
                    className={styles.hand}
                    x1="60"
                    y1="60"
                    x2={60 + 42 * Math.cos(phase)}
                    y2={60 - 42 * Math.sin(phase)}
                  />
                  <circle className={styles.center} cx="60" cy="60" r="3" />
                </svg>
                <small className={styles.readout}>
                  θ<sub>{index}</sub> = {Number(frequency.toPrecision(4))}
                  <br />
                  {(phase / Math.PI).toFixed(3)}π · {turns.toFixed(3)} 圈
                </small>
              </div>
            );
          })}
        </div>
      </div>
      <div className="card__footer">
        <small>
          圈内角度为 mθᵢ mod 2π，圈数为 mθᵢ / 2π。i 越大，频率越低，指针转得越慢。
        </small>
      </div>
    </div>
  );
}
