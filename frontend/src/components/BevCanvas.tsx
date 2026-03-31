import { useMemo } from "react";

import type { AgentState, Point2D, SceneGeometry } from "../types";

type Props = {
  agents: AgentState[];
  sceneGeometry?: SceneGeometry;
};

type Bounds = {
  minX: number;
  maxX: number;
  minY: number;
  maxY: number;
};

const WIDTH = 880;
const HEIGHT = 560;

function collectPoints(agents: AgentState[]): Point2D[] {
  const pts: Point2D[] = [{ x: 0, y: 0 }];
  for (const agent of agents) {
    pts.push(...agent.history);
    for (const mode of agent.predictions) {
      pts.push(...mode);
    }
  }

  return pts;
}

function computeBounds(points: Point2D[]): Bounds {
  let minX = Number.POSITIVE_INFINITY;
  let maxX = Number.NEGATIVE_INFINITY;
  let minY = Number.POSITIVE_INFINITY;
  let maxY = Number.NEGATIVE_INFINITY;

  for (const p of points) {
    if (!Number.isFinite(p.x) || !Number.isFinite(p.y)) {
      continue;
    }
    minX = Math.min(minX, p.x);
    maxX = Math.max(maxX, p.x);
    minY = Math.min(minY, p.y);
    maxY = Math.max(maxY, p.y);
  }

  if (!Number.isFinite(minX) || !Number.isFinite(maxX)) {
    return { minX: -30, maxX: 30, minY: -10, maxY: 55 };
  }

  const dx = Math.max(10, maxX - minX);
  const dy = Math.max(10, maxY - minY);
  const px = dx * 0.2;
  const py = dy * 0.2;

  return {
    minX: minX - px,
    maxX: maxX + px,
    minY: minY - py,
    maxY: maxY + py
  };
}

function toSvgPoint(p: Point2D, b: Bounds): Point2D {
  const x = ((p.x - b.minX) / (b.maxX - b.minX || 1)) * WIDTH;
  const yWorld = ((p.y - b.minY) / (b.maxY - b.minY || 1)) * HEIGHT;
  return { x, y: HEIGHT - yWorld };
}

function polyline(points: Point2D[]): string {
  return points.map((p) => `${p.x.toFixed(2)},${p.y.toFixed(2)}`).join(" ");
}

function buildTicks(min: number, max: number): number[] {
  const span = Math.max(1, max - min);
  const step = span > 160 ? 40 : span > 100 ? 20 : 10;
  const start = Math.ceil(min / step) * step;
  const out: number[] = [];
  for (let t = start; t <= max; t += step) {
    out.push(Number(t.toFixed(2)));
  }
  return out;
}

function palette(agent: AgentState): { marker: string; line: string; ghost: string } {
  if (agent.is_target) {
    return { marker: "#f97316", line: "#ea580c", ghost: "#fdba74" };
  }
  if (agent.type === "pedestrian") {
    return { marker: "#10b981", line: "#059669", ghost: "#6ee7b7" };
  }
  return { marker: "#38bdf8", line: "#0284c7", ghost: "#7dd3fc" };
}

function shortAgentClass(agent: AgentState): string {
  const raw = (agent.raw_label ?? "").toLowerCase();
  if (raw === "person" || raw === "pedestrian") {
    return "PED";
  }
  if (raw === "bicycle" || raw === "cyclist") {
    return "CYC";
  }
  if (agent.type === "pedestrian") {
    return "VRU";
  }
  return "VEH";
}

function isPedestrianOrCyclist(agent: AgentState): boolean {
  const raw = (agent.raw_label ?? "").toLowerCase();
  if (raw === "person" || raw === "pedestrian" || raw === "bicycle" || raw === "cyclist") {
    return true;
  }
  return agent.type === "pedestrian";
}

export default function BevCanvas({ agents }: Props) {
  const bounds = useMemo(() => computeBounds(collectPoints(agents)), [agents]);
  const vruDisplayIdByTrackId = useMemo(() => {
    const map = new Map<number, number>();
    agents
      .filter((a) => isPedestrianOrCyclist(a))
      .slice()
      .sort((a, b) => a.id - b.id)
      .forEach((a, idx) => {
        map.set(a.id, idx + 1);
      });
    return map;
  }, [agents]);
  const xTicks = useMemo(() => buildTicks(bounds.minX, bounds.maxX), [bounds.maxX, bounds.minX]);
  const yTicks = useMemo(() => buildTicks(bounds.minY, bounds.maxY), [bounds.maxY, bounds.minY]);

  return (
    <div className="bev-shell">
      <svg viewBox={`0 0 ${WIDTH} ${HEIGHT}`} className="bev-svg" role="img" aria-label="BEV forecast map">
        <defs>
          <linearGradient id="bevBg" x1="0%" y1="0%" x2="0%" y2="100%">
            <stop offset="0%" stopColor="#071224" />
            <stop offset="100%" stopColor="#030a16" />
          </linearGradient>
          <pattern id="gridPattern" width="44" height="44" patternUnits="userSpaceOnUse">
            <path d="M 44 0 L 0 0 0 44" fill="none" stroke="rgba(145, 158, 171, 0.16)" strokeWidth="1" />
          </pattern>
          <radialGradient id="egoGlow" cx="50%" cy="50%" r="50%">
            <stop offset="0%" stopColor="rgba(56, 189, 248, 0.55)" />
            <stop offset="100%" stopColor="rgba(56, 189, 248, 0.05)" />
          </radialGradient>
        </defs>

        <rect x={0} y={0} width={WIDTH} height={HEIGHT} fill="url(#bevBg)" />

        <rect x={0} y={0} width={WIDTH} height={HEIGHT} fill="url(#gridPattern)" opacity={0.9} />

        {xTicks.map((tick) => {
          const p = toSvgPoint({ x: tick, y: bounds.minY }, bounds);
          return (
            <g key={`xt-${tick}`}>
              <line x1={p.x} y1={0} x2={p.x} y2={HEIGHT} stroke="rgba(148, 163, 184, 0.13)" strokeWidth={1} />
              <text x={p.x + 2} y={HEIGHT - 6} fill="rgba(203, 213, 225, 0.72)" fontSize={10}>
                {tick.toFixed(0)}
              </text>
            </g>
          );
        })}

        {yTicks.map((tick) => {
          const p = toSvgPoint({ x: bounds.minX, y: tick }, bounds);
          return (
            <g key={`yt-${tick}`}>
              <line x1={0} y1={p.y} x2={WIDTH} y2={p.y} stroke="rgba(148, 163, 184, 0.13)" strokeWidth={1} />
              <text x={4} y={p.y - 4} fill="rgba(203, 213, 225, 0.72)" fontSize={10}>
                {tick.toFixed(0)}
              </text>
            </g>
          );
        })}

        {agents.map((agent) => {
          const colors = palette(agent);
          const history = agent.history.map((p) => toSvgPoint(p, bounds));
          const bestModeIdx = agent.probabilities.length
            ? agent.probabilities.indexOf(Math.max(...agent.probabilities))
            : 0;
          const bestMode = agent.predictions[bestModeIdx] ?? [];
          const bestPath = [agent.history[agent.history.length - 1], ...bestMode]
            .filter(Boolean)
            .map((p) => toSvgPoint(p, bounds));

          const altModes = agent.is_target
            ? agent.predictions
                .map((mode, idx) => ({ mode, idx }))
                .filter((x) => x.idx !== bestModeIdx)
                .slice(0, 2)
            : [];

          const current = history[history.length - 1];
          const previous = history.length >= 2 ? history[history.length - 2] : current;
          const kindTag = shortAgentClass(agent);
          const vruDisplayId = vruDisplayIdByTrackId.get(agent.id);
          const idText = vruDisplayId !== undefined ? `ID ${vruDisplayId}` : `TRK ${agent.id}`;
          const label = agent.is_target ? `${idText} ${kindTag} TARGET` : `${idText} ${kindTag}`;
          const labelWidth = 10 + label.length * 6.4;
          const endPoint = bestPath[bestPath.length - 1];
          const vx = current && previous ? current.x - previous.x : 0;
          const vy = current && previous ? current.y - previous.y : 0;
          const vn = Math.hypot(vx, vy);

          return (
            <g key={agent.id}>
              {altModes.map(({ mode, idx }) => {
                const points = [agent.history[agent.history.length - 1], ...mode]
                  .filter(Boolean)
                  .map((p) => toSvgPoint(p, bounds));
                return (
                  <polyline
                    key={`${agent.id}-alt-${idx}`}
                    points={polyline(points)}
                    fill="none"
                    stroke={colors.ghost}
                    strokeWidth={2}
                    opacity={0.46}
                    strokeDasharray="8 8"
                  />
                );
              })}

              <polyline
                points={polyline(history)}
                fill="none"
                stroke="rgba(226, 232, 240, 0.72)"
                strokeWidth={1.8}
                strokeDasharray="4 6"
              />

              <polyline points={polyline(bestPath)} fill="none" stroke={colors.line} strokeWidth={7} strokeLinecap="round" opacity={0.2} />
              <polyline points={polyline(bestPath)} fill="none" stroke={colors.line} strokeWidth={3.2} strokeLinecap="round" />

              {current && vn > 1e-6 && (
                <line
                  x1={current.x}
                  y1={current.y}
                  x2={current.x + (vx / vn) * 16}
                  y2={current.y + (vy / vn) * 16}
                  stroke={colors.line}
                  strokeWidth={2}
                  strokeLinecap="round"
                  opacity={0.85}
                />
              )}

              {current && (
                <g>
                  <circle cx={current.x} cy={current.y} r={10} fill={colors.marker} fillOpacity={0.25} />
                  <circle cx={current.x} cy={current.y} r={5} fill={colors.marker} />
                  <rect x={current.x + 8} y={current.y - 20} width={labelWidth} height={15} rx={4} fill="rgba(2, 6, 23, 0.72)" />
                  <text x={current.x + 12} y={current.y - 9} fill="#f8fafc" fontSize={10} fontWeight={700}>
                    {label}
                  </text>
                </g>
              )}

              {endPoint && (
                <circle cx={endPoint.x} cy={endPoint.y} r={3.5} fill={colors.line} opacity={0.95} />
              )}
            </g>
          );
        })}

        <g transform="translate(12, 12)">
          <rect x={0} y={0} width={250} height={82} rx={8} fill="rgba(2, 6, 23, 0.65)" stroke="rgba(148, 163, 184, 0.32)" />
          <text x={10} y={16} fill="#f8fafc" fontSize={11} fontWeight={700}>
            BEV Legend
          </text>
          <line x1={10} y1={30} x2={34} y2={30} stroke="#ea580c" strokeWidth={3} />
          <text x={40} y={33} fill="rgba(226, 232, 240, 0.92)" fontSize={10}>
            Target trajectory
          </text>
          <line x1={10} y1={44} x2={34} y2={44} stroke="rgba(226, 232, 240, 0.8)" strokeWidth={2} strokeDasharray="4 6" />
          <text x={40} y={47} fill="rgba(226, 232, 240, 0.92)" fontSize={10}>
            Observed history
          </text>
          <text x={10} y={61} fill="rgba(226, 232, 240, 0.85)" fontSize={10}>
            Map layer hidden for clarity.
          </text>
          <text x={10} y={74} fill="rgba(203, 213, 225, 0.72)" fontSize={9}>
            VRU IDs are counted over pedestrian/cyclist only.
          </text>
        </g>

        <text x={WIDTH - 58} y={22} fill="rgba(226, 232, 240, 0.9)" fontSize={11} fontWeight={700}>
          N
        </text>
        <line x1={WIDTH - 52} y1={38} x2={WIDTH - 52} y2={14} stroke="rgba(226, 232, 240, 0.9)" strokeWidth={2} />

        <circle cx={WIDTH * 0.5} cy={HEIGHT * 0.82} r={30} fill="url(#egoGlow)" />
        <rect x={WIDTH * 0.5 - 10} y={HEIGHT * 0.82 - 16} width={20} height={32} rx={4} fill="rgba(56, 189, 248, 0.85)" />
        <polygon
          points={`${WIDTH * 0.5},${HEIGHT * 0.82 - 26} ${WIDTH * 0.5 - 7},${HEIGHT * 0.82 - 12} ${WIDTH * 0.5 + 7},${HEIGHT * 0.82 - 12}`}
          fill="rgba(125, 211, 252, 0.95)"
        />

        <text x={WIDTH - 110} y={HEIGHT - 8} fill="rgba(203, 213, 225, 0.75)" fontSize={10}>
          X right (m), Y forward (m)
        </text>
      </svg>
    </div>
  );
}
