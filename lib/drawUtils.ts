import { type GesturePoint, calculatePointDistance } from '@/lib/gestureUtils';

export type LaserColor = 'cyan' | 'pink';

export interface LaserStyle {
  color?: LaserColor;
  lineWidth?: number;
  shadowBlur?: number;
  alpha?: number;
}

const LASER_COLORS: Record<LaserColor, string> = {
  cyan: '#22d3ee',
  pink: '#f472b6',
};

export function clearCanvas(canvas: HTMLCanvasElement | null) {
  if (!canvas) return;

  const ctx = canvas.getContext('2d');
  if (!ctx) return;

  ctx.clearRect(0, 0, canvas.width, canvas.height);
}

export function prepareLaserStroke(ctx: CanvasRenderingContext2D, style: LaserStyle = {}) {
  const color = LASER_COLORS[style.color ?? 'cyan'];

  ctx.save();
  ctx.globalAlpha = style.alpha ?? 1;
  ctx.strokeStyle = color;
  ctx.lineWidth = style.lineWidth ?? 4;
  ctx.lineCap = 'round';
  ctx.lineJoin = 'round';
  ctx.shadowBlur = style.shadowBlur ?? 22;
  ctx.shadowColor = color;
}

export function finishLaserStroke(ctx: CanvasRenderingContext2D) {
  ctx.restore();
}

export function drawLaserSegment(
  ctx: CanvasRenderingContext2D,
  from: GesturePoint,
  to: GesturePoint,
  style?: LaserStyle
) {
  prepareLaserStroke(ctx, style);
  ctx.beginPath();
  ctx.moveTo(from.x, from.y);
  ctx.lineTo(to.x, to.y);
  ctx.stroke();
  finishLaserStroke(ctx);
}

export function drawLaserPath(
  ctx: CanvasRenderingContext2D,
  points: GesturePoint[],
  style?: LaserStyle
) {
  if (points.length < 2) return;

  prepareLaserStroke(ctx, style);
  ctx.beginPath();
  ctx.moveTo(points[0].x, points[0].y);

  for (let i = 1; i < points.length - 1; i += 1) {
    const midpoint = {
      x: (points[i].x + points[i + 1].x) / 2,
      y: (points[i].y + points[i + 1].y) / 2,
    };

    ctx.quadraticCurveTo(points[i].x, points[i].y, midpoint.x, midpoint.y);
  }

  const last = points[points.length - 1];
  ctx.lineTo(last.x, last.y);
  ctx.stroke();
  finishLaserStroke(ctx);
}

export function drawLaserLine(
  ctx: CanvasRenderingContext2D,
  start: GesturePoint,
  end: GesturePoint,
  style?: LaserStyle
) {
  drawLaserSegment(ctx, start, end, style);
}

export function drawLaserRectangle(
  ctx: CanvasRenderingContext2D,
  start: GesturePoint,
  end: GesturePoint,
  style?: LaserStyle
) {
  prepareLaserStroke(ctx, style);
  ctx.strokeRect(start.x, start.y, end.x - start.x, end.y - start.y);
  finishLaserStroke(ctx);
}

export function drawLaserCircle(
  ctx: CanvasRenderingContext2D,
  start: GesturePoint,
  end: GesturePoint,
  style?: LaserStyle
) {
  const center = {
    x: (start.x + end.x) / 2,
    y: (start.y + end.y) / 2,
  };
  const radius = calculatePointDistance(start, end) / 2;

  prepareLaserStroke(ctx, style);
  ctx.beginPath();
  ctx.arc(center.x, center.y, radius, 0, Math.PI * 2);
  ctx.stroke();
  finishLaserStroke(ctx);
}

export function resizeCanvasToDisplaySize(canvas: HTMLCanvasElement, width: number, height: number) {
  const nextWidth = Math.max(1, Math.round(width));
  const nextHeight = Math.max(1, Math.round(height));

  if (canvas.width !== nextWidth || canvas.height !== nextHeight) {
    canvas.width = nextWidth;
    canvas.height = nextHeight;
  }
}
