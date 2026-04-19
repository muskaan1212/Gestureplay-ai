'use client';

import { useCallback, useEffect, useRef, useState } from 'react';
import {
  isPinchDistance,
  landmarkToCanvasPoint,
  smoothPoint,
  type GesturePoint,
} from '@/lib/gestureUtils';
import {
  clearCanvas,
  drawLaserCircle,
  drawLaserLine,
  drawLaserPath,
  drawLaserRectangle,
  drawLaserSegment,
} from '@/lib/drawUtils';

type SketchMode = 'freehand' | 'line' | 'rectangle' | 'circle';

interface HandLandmark {
  x: number;
  y: number;
  z?: number;
}

interface Handedness {
  score?: number;
}

interface HandsResults {
  multiHandLandmarks?: HandLandmark[][];
  multiHandedness?: Handedness[];
}

interface MediaPipeHands {
  setOptions: (options: {
    maxNumHands: number;
    modelComplexity: number;
    minDetectionConfidence: number;
    minTrackingConfidence: number;
  }) => void;
  onResults: (callback: (results: HandsResults) => void) => void;
  send: (input: { image: HTMLVideoElement }) => Promise<void>;
  close?: () => void;
}

type HandsConstructor = new (config: {
  locateFile: (file: string) => string;
}) => MediaPipeHands;

interface TrackedHand {
  index: GesturePoint;
  thumb: GesturePoint;
  pinching: boolean;
  score: number;
}

const CANVAS_WIDTH = 1280;
const CANVAS_HEIGHT = 720;
const PINCH_THRESHOLD = 0.055;
const CONFIDENCE_THRESHOLD = 0.65;
const RELEASE_COOLDOWN_MS = 350;

function useWebcamStream(
  cameraOn: boolean,
  videoRef: React.RefObject<HTMLVideoElement | null>,
  onError: (message: string | null) => void
) {
  const streamRef = useRef<MediaStream | null>(null);

  useEffect(() => {
    let cancelled = false;
    const videoElement = videoRef.current;

    async function startStream() {
      if (!cameraOn) return;

      try {
        onError(null);
        const stream = await navigator.mediaDevices.getUserMedia({
          video: {
            width: { ideal: CANVAS_WIDTH },
            height: { ideal: CANVAS_HEIGHT },
            facingMode: 'user',
          },
          audio: false,
        });

        if (cancelled) {
          stream.getTracks().forEach((track) => track.stop());
          return;
        }

        streamRef.current = stream;

        if (videoElement) {
          videoElement.srcObject = stream;
          await videoElement.play();
        }
      } catch {
        onError('Camera permission is needed to sketch with gestures.');
      }
    }

    startStream();

    return () => {
      cancelled = true;
      streamRef.current?.getTracks().forEach((track) => track.stop());
      streamRef.current = null;

      if (videoElement) {
        videoElement.srcObject = null;
      }
    };
  }, [cameraOn, onError, videoRef]);
}

function useHandTracking(
  cameraOn: boolean,
  videoRef: React.RefObject<HTMLVideoElement | null>,
  onHands: (hands: TrackedHand[]) => void,
  onError: (message: string | null) => void
) {
  const handsRef = useRef<MediaPipeHands | null>(null);
  const rafRef = useRef<number | null>(null);
  const runningRef = useRef(false);
  const sendingRef = useRef(false);
  const smoothedRef = useRef<Array<{ index: GesturePoint | null; thumb: GesturePoint | null }>>([]);

  useEffect(() => {
    runningRef.current = cameraOn;

    if (!cameraOn) {
      if (rafRef.current) cancelAnimationFrame(rafRef.current);
      rafRef.current = null;
      handsRef.current?.close?.();
      handsRef.current = null;
      smoothedRef.current = [];
      onHands([]);
      return;
    }

    async function startTracking() {
      try {
        const { Hands } = (await import('@mediapipe/hands')) as unknown as {
          Hands: HandsConstructor;
        };

        const hands = new Hands({
          locateFile: (file: string) =>
            `https://cdn.jsdelivr.net/npm/@mediapipe/hands@0.4.1675469240/${file}`,
        });

        hands.setOptions({
          maxNumHands: 2,
          modelComplexity: 1,
          minDetectionConfidence: 0.7,
          minTrackingConfidence: 0.65,
        });

        hands.onResults((results) => {
          if (!runningRef.current) return;

          const trackedHands =
            results.multiHandLandmarks?.flatMap((landmarks, handIndex) => {
              const score = results.multiHandedness?.[handIndex]?.score ?? 1;

              if (score < CONFIDENCE_THRESHOLD || landmarks.length < 9) {
                return [];
              }

              const indexTip = landmarks[8];
              const thumbTip = landmarks[4];
              const previous = smoothedRef.current[handIndex] ?? {
                index: null,
                thumb: null,
              };
              const index = smoothPoint(
                previous.index,
                landmarkToCanvasPoint(indexTip, CANVAS_WIDTH, CANVAS_HEIGHT)
              );
              const thumb = smoothPoint(
                previous.thumb,
                landmarkToCanvasPoint(thumbTip, CANVAS_WIDTH, CANVAS_HEIGHT)
              );

              smoothedRef.current[handIndex] = { index, thumb };

              return [
                {
                  index,
                  thumb,
                  pinching: isPinchDistance(thumbTip, indexTip, PINCH_THRESHOLD),
                  score,
                },
              ];
            }) ?? [];

          onHands(trackedHands.slice(0, 2));
        });

        handsRef.current = hands;

        const tick = async () => {
          if (!runningRef.current) return;

          const video = videoRef.current;

          if (
            video &&
            video.readyState >= HTMLMediaElement.HAVE_CURRENT_DATA &&
            !sendingRef.current
          ) {
            sendingRef.current = true;

            try {
              await hands.send({ image: video });
            } catch {
              onError('Hand tracking paused. Restart the camera if the feed stopped.');
            } finally {
              sendingRef.current = false;
            }
          }

          rafRef.current = requestAnimationFrame(tick);
        };

        rafRef.current = requestAnimationFrame(tick);
      } catch {
        onError('MediaPipe Hands could not be loaded in this browser.');
      }
    }

    startTracking();

    return () => {
      runningRef.current = false;

      if (rafRef.current) cancelAnimationFrame(rafRef.current);
      rafRef.current = null;
      handsRef.current?.close?.();
      handsRef.current = null;
      sendingRef.current = false;
    };
  }, [cameraOn, onError, onHands, videoRef]);
}

export function LaserSketch() {
  const [cameraOn, setCameraOn] = useState(false);
  const [mode, setMode] = useState<SketchMode>('freehand');
  const [status, setStatus] = useState('Camera off');
  const [error, setError] = useState<string | null>(null);

  const videoRef = useRef<HTMLVideoElement>(null);
  const permanentCanvasRef = useRef<HTMLCanvasElement>(null);
  const previewCanvasRef = useRef<HTMLCanvasElement>(null);
  const modeRef = useRef<SketchMode>('freehand');
  const activePathRef = useRef<GesturePoint[]>([]);
  const wasFreehandPinchingRef = useRef(false);
  const wasShapePinchingRef = useRef(false);
  const lastShapePointsRef = useRef<[GesturePoint, GesturePoint] | null>(null);
  const lastReleaseRef = useRef(0);
  const statusRef = useRef('Camera off');

  useEffect(() => {
    modeRef.current = mode;
  }, [mode]);

  const updateStatus = useCallback((nextStatus: string) => {
    if (statusRef.current === nextStatus) return;

    statusRef.current = nextStatus;
    setStatus(nextStatus);
  }, []);

  const drawPermanentShape = useCallback((shapeMode: SketchMode, start: GesturePoint, end: GesturePoint) => {
    const ctx = permanentCanvasRef.current?.getContext('2d');
    if (!ctx) return;

    if (shapeMode === 'line') drawLaserLine(ctx, start, end, { color: 'pink' });
    if (shapeMode === 'rectangle') drawLaserRectangle(ctx, start, end, { color: 'pink' });
    if (shapeMode === 'circle') drawLaserCircle(ctx, start, end, { color: 'pink' });
  }, []);

  const finalizeShape = useCallback(() => {
    const now = Date.now();
    const points = lastShapePointsRef.current;

    if (!points || now - lastReleaseRef.current < RELEASE_COOLDOWN_MS) return;

    drawPermanentShape(modeRef.current, points[0], points[1]);
    lastReleaseRef.current = now;
    lastShapePointsRef.current = null;
  }, [drawPermanentShape]);

  const handleHands = useCallback(
    (hands: TrackedHand[]) => {
      const previewCanvas = previewCanvasRef.current;
      const previewCtx = previewCanvas?.getContext('2d');
      const permanentCtx = permanentCanvasRef.current?.getContext('2d');
      const activeMode = modeRef.current;

      if (!previewCanvas || !previewCtx || !permanentCtx) return;

      previewCtx.clearRect(0, 0, previewCanvas.width, previewCanvas.height);

      for (const hand of hands) {
        drawLaserLine(previewCtx, hand.thumb, hand.index, {
          color: hand.pinching ? 'pink' : 'cyan',
          lineWidth: hand.pinching ? 3 : 2,
          shadowBlur: hand.pinching ? 24 : 12,
          alpha: hand.pinching ? 0.95 : 0.45,
        });
      }

      const pinchingHands = hands.filter((hand) => hand.pinching);
      const bothPinching = pinchingHands.length >= 2;

      if (!cameraOn) {
        updateStatus('Camera off');
        return;
      }

      if (activeMode === 'freehand') {
        const drawingHand = pinchingHands[0];

        if (drawingHand && !bothPinching) {
          const path = activePathRef.current;
          const previousPoint = path[path.length - 1];

          if (previousPoint) {
            drawLaserSegment(permanentCtx, previousPoint, drawingHand.index, { color: 'cyan' });
          }

          path.push(drawingHand.index);
          wasFreehandPinchingRef.current = true;
          updateStatus('Drawing');
          return;
        }

        if (wasFreehandPinchingRef.current) {
          drawLaserPath(permanentCtx, activePathRef.current, { color: 'cyan' });
          activePathRef.current = [];
          wasFreehandPinchingRef.current = false;
          updateStatus('Path saved');
          return;
        }

        activePathRef.current = [];
        updateStatus(hands.length ? 'Pinch to draw' : 'Show one hand');
        return;
      }

      if (bothPinching) {
        const start = pinchingHands[0].index;
        const end = pinchingHands[1].index;

        if (activeMode === 'line') drawLaserLine(previewCtx, start, end, { color: 'pink' });
        if (activeMode === 'rectangle') drawLaserRectangle(previewCtx, start, end, { color: 'pink' });
        if (activeMode === 'circle') drawLaserCircle(previewCtx, start, end, { color: 'pink' });

        lastShapePointsRef.current = [start, end];
        wasShapePinchingRef.current = true;
        updateStatus(`${activeMode[0].toUpperCase()}${activeMode.slice(1)} preview`);
        return;
      }

      if (wasShapePinchingRef.current) {
        finalizeShape();
        wasShapePinchingRef.current = false;
        updateStatus('Shape saved');
        return;
      }

      updateStatus(hands.length ? 'Pinch with both hands' : 'Show both hands');
    },
    [cameraOn, finalizeShape, updateStatus]
  );

  useWebcamStream(cameraOn, videoRef, setError);
  useHandTracking(cameraOn, videoRef, handleHands, setError);

  const clearDrawing = () => {
    clearCanvas(permanentCanvasRef.current);
    clearCanvas(previewCanvasRef.current);
    activePathRef.current = [];
    lastShapePointsRef.current = null;
    updateStatus(cameraOn ? 'Canvas cleared' : 'Camera off');
  };

  const saveDrawing = () => {
    const permanentCanvas = permanentCanvasRef.current;
    const previewCanvas = previewCanvasRef.current;

    if (!permanentCanvas || !previewCanvas) return;

    const exportCanvas = document.createElement('canvas');
    exportCanvas.width = permanentCanvas.width;
    exportCanvas.height = permanentCanvas.height;

    const ctx = exportCanvas.getContext('2d');
    if (!ctx) return;

    ctx.drawImage(permanentCanvas, 0, 0);
    ctx.drawImage(previewCanvas, 0, 0);

    const link = document.createElement('a');
    link.href = exportCanvas.toDataURL('image/png');
    link.download = `laser-sketch-${Date.now()}.png`;
    link.click();
    updateStatus('PNG saved');
  };

  return (
    <main className="min-h-screen bg-neutral-950 text-white">
      <section className="mx-auto flex min-h-screen w-full max-w-7xl flex-col gap-5 px-4 py-4 sm:px-6 lg:px-8">
        <div className="flex flex-col justify-between gap-4 md:flex-row md:items-end">
          <div>
            <p className="text-sm font-medium uppercase tracking-[0.22em] text-cyan-300">
              Laser Sketch
            </p>
            <h1 className="mt-2 text-3xl font-bold sm:text-5xl">Draw light in the air</h1>
            <p className="mt-2 max-w-2xl text-sm text-zinc-300 sm:text-base">
              Pinch to sketch, use two hands for glowing geometry, and export the result as a PNG.
            </p>
          </div>

          <div className="rounded-md border border-cyan-300/30 bg-cyan-300/10 px-4 py-3 text-sm text-cyan-100 shadow-[0_0_24px_rgba(34,211,238,0.18)]">
            {error ?? status}
          </div>
        </div>

        <div className="flex flex-wrap items-center gap-2 rounded-md border border-white/10 bg-zinc-900/80 p-3">
          <button
            type="button"
            onClick={() => setCameraOn((value) => !value)}
            className={`rounded-md px-4 py-2 text-sm font-semibold transition ${
              cameraOn
                ? 'bg-rose-500 text-white hover:bg-rose-400'
                : 'bg-cyan-300 text-zinc-950 hover:bg-cyan-200'
            }`}
          >
            {cameraOn ? 'Stop camera' : 'Start camera'}
          </button>

          {(['freehand', 'line', 'rectangle', 'circle'] as const).map((nextMode) => (
            <button
              key={nextMode}
              type="button"
              onClick={() => setMode(nextMode)}
              className={`rounded-md px-3 py-2 text-sm font-medium capitalize transition ${
                mode === nextMode
                  ? 'bg-pink-400 text-zinc-950 shadow-[0_0_18px_rgba(244,114,182,0.45)]'
                  : 'bg-zinc-800 text-zinc-200 hover:bg-zinc-700'
              }`}
            >
              {nextMode}
            </button>
          ))}

          <button
            type="button"
            onClick={clearDrawing}
            className="rounded-md bg-zinc-800 px-3 py-2 text-sm font-medium text-zinc-100 transition hover:bg-zinc-700"
          >
            Clear canvas
          </button>

          <button
            type="button"
            onClick={saveDrawing}
            className="rounded-md bg-white px-3 py-2 text-sm font-semibold text-zinc-950 transition hover:bg-zinc-200"
          >
            Save drawing
          </button>
        </div>

        <div className="relative min-h-[360px] flex-1 overflow-hidden rounded-md border border-white/10 bg-black shadow-[0_0_60px_rgba(34,211,238,0.18)]">
          <video
            ref={videoRef}
            muted
            playsInline
            className="absolute inset-0 h-full w-full -scale-x-100 object-cover opacity-70"
            width={CANVAS_WIDTH}
            height={CANVAS_HEIGHT}
          />
          <canvas
            ref={permanentCanvasRef}
            width={CANVAS_WIDTH}
            height={CANVAS_HEIGHT}
            className="absolute inset-0 h-full w-full"
          />
          <canvas
            ref={previewCanvasRef}
            width={CANVAS_WIDTH}
            height={CANVAS_HEIGHT}
            className="absolute inset-0 h-full w-full"
          />

          {!cameraOn && (
            <div className="absolute inset-0 flex items-center justify-center bg-black/70 px-6 text-center">
              <div>
                <p className="text-xl font-semibold text-cyan-100">Camera ready</p>
                <p className="mt-2 max-w-md text-sm text-zinc-300">
                  Start the camera, hold your hand in view, then pinch thumb and index finger to draw.
                </p>
              </div>
            </div>
          )}

          <div className="pointer-events-none absolute bottom-3 left-3 right-3 flex flex-wrap gap-2 text-xs text-zinc-200">
            <span className="rounded bg-black/60 px-2 py-1">Freehand: one-hand pinch</span>
            <span className="rounded bg-black/60 px-2 py-1">Shapes: two-hand pinch and release</span>
            <span className="rounded bg-black/60 px-2 py-1">Neon layers export as PNG</span>
          </div>
        </div>
      </section>
    </main>
  );
}
