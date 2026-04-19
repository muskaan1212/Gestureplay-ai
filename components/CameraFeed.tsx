'use client';

import { useEffect, useRef, useState } from 'react';
import { 
  detectGesture, 
  getCursorPosition, 
  detectScrollGesture,
  type GestureType 
} from '@/lib/gestureUtils';

interface HandLandmark {
  x: number;
  y: number;
  z?: number;
}

interface HandsResults {
  multiHandLandmarks?: HandLandmark[][];
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

interface MediaPipeCamera {
  start: () => void;
  stop: () => void;
}

type CameraConstructor = new (
  video: HTMLVideoElement,
  config: {
    onFrame: () => Promise<void>;
    width: number;
    height: number;
  }
) => MediaPipeCamera;

interface CameraFeedProps {
  onGestureChange: (gesture: GestureType, x?: number, y?: number) => void;
  isActive: boolean;
}

export function CameraFeed({ onGestureChange, isActive }: CameraFeedProps) {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const handsRef = useRef<MediaPipeHands | null>(null);
  const cameraRef = useRef<MediaPipeCamera | null>(null);
  const previousHandRef = useRef<HandLandmark[] | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const lastScrollTimeRef = useRef<number>(0);
  const isRunningRef = useRef(false);

  useEffect(() => {
    isRunningRef.current = isActive;

    if (!isActive) {
      isRunningRef.current = false;
      if (cameraRef.current) {
        cameraRef.current.stop();
        cameraRef.current = null;
      }
      handsRef.current?.close?.();
      handsRef.current = null;
      setIsLoading(true);
      return;
    }

    const initializeMediaPipe = async () => {
      try {
        setError(null);
        
        // Dynamically import MediaPipe modules
        const { Hands } = (await import('@mediapipe/hands')) as unknown as {
          Hands: HandsConstructor;
        };
        
        const hands = new Hands({
          locateFile: (file: string) => {
            return `https://cdn.jsdelivr.net/npm/@mediapipe/hands@0.4.1675469240/${file}`;
          },
        });

        hands.setOptions({
          maxNumHands: 1,
          modelComplexity: 1,
          minDetectionConfidence: 0.5,
          minTrackingConfidence: 0.5,
        });

        hands.onResults((results) => {
          const canvas = canvasRef.current;
          const video = videoRef.current;
          
          if (!canvas || !video) return;

          const ctx = canvas.getContext('2d');
          if (!ctx) return;

          // Draw video frame
          ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

          const hand = results.multiHandLandmarks?.[0];
          // Pass landmarks array directly to gesture detection functions
          const gesture = detectGesture(hand);

          // Detect scroll gestures with debouncing
          const now = Date.now();
          if (now - lastScrollTimeRef.current > 300) {
            const scrollGesture = detectScrollGesture(
              hand,
              previousHandRef.current
            );
            if (scrollGesture !== 'none') {
              onGestureChange(scrollGesture);
              lastScrollTimeRef.current = now;
            }
          }

          if (hand) {
            // Draw hand landmarks
            const canvasWidth = canvas.width;
            const canvasHeight = canvas.height;

            // Draw connections
            const connections = [
              [0, 1], [1, 2], [2, 3], [3, 4], // Thumb
              [5, 6], [6, 7], [7, 8], // Index
              [9, 10], [10, 11], [11, 12], // Middle
              [13, 14], [14, 15], [15, 16], // Ring
              [17, 18], [18, 19], [19, 20], // Pinky
              [0, 5], [5, 9], [9, 13], [13, 17], // Palm connections
            ];

            ctx.strokeStyle = 'rgba(0, 255, 0, 0.5)';
            ctx.lineWidth = 2;
            for (const [start, end] of connections) {
              const p1 = hand[start];
              const p2 = hand[end];
              ctx.beginPath();
              ctx.moveTo(p1.x * canvasWidth, p1.y * canvasHeight);
              ctx.lineTo(p2.x * canvasWidth, p2.y * canvasHeight);
              ctx.stroke();
            }

            // Draw landmarks as circles
            ctx.fillStyle = 'rgba(255, 0, 0, 0.8)';
            for (const landmark of hand) {
              const x = landmark.x * canvasWidth;
              const y = landmark.y * canvasHeight;
              ctx.beginPath();
              ctx.arc(x, y, 5, 0, 2 * Math.PI);
              ctx.fill();
            }

            const { x, y } = getCursorPosition(hand, canvasWidth, canvasHeight);

            if (gesture === 'pinch') {
              ctx.strokeStyle = 'rgba(250, 204, 21, 0.95)';
              ctx.lineWidth = 5;
              ctx.beginPath();
              ctx.arc(x, y, 24, 0, 2 * Math.PI);
              ctx.stroke();

              onGestureChange(gesture, x, y);
            } else {
              ctx.strokeStyle = 'rgba(0, 150, 255, 0.8)';
              ctx.lineWidth = 3;
              ctx.beginPath();
              ctx.arc(x, y, 20, 0, 2 * Math.PI);
              ctx.stroke();

              onGestureChange(gesture, x, y);
            }

            previousHandRef.current = hand;
          } else {
            onGestureChange('none');
            previousHandRef.current = null;
          }
        });

        handsRef.current = hands;

        const videoElement = videoRef.current;

        if (videoElement && isRunningRef.current) {
          const { Camera: CameraClass } = (await import('@mediapipe/camera_utils')) as unknown as {
            Camera: CameraConstructor;
          };
          const camera = new CameraClass(videoElement, {
            onFrame: async () => {
              if (!isRunningRef.current || videoElement.readyState < HTMLMediaElement.HAVE_CURRENT_DATA) {
                return;
              }

              await hands.send({ image: videoElement });
            },
            width: 1280,
            height: 720,
          });

          cameraRef.current = camera;
          camera.start();

          if (isRunningRef.current) {
            setIsLoading(false);
          }
        }
      } catch (err) {
        console.error('Failed to initialize MediaPipe:', err);

        if (isRunningRef.current) {
          setError('Failed to initialize camera. Please check permissions.');
          setIsLoading(false);
        }
      }
    };

    initializeMediaPipe();

    return () => {
      isRunningRef.current = false;

      if (cameraRef.current) {
        cameraRef.current.stop();
        cameraRef.current = null;
      }
      handsRef.current?.close?.();
      handsRef.current = null;
    };
  }, [isActive, onGestureChange]);

  return (
    <div className="relative w-full h-full bg-black rounded-lg overflow-hidden">
      <video
        ref={videoRef}
        className="hidden"
        width={1280}
        height={720}
      />
      <canvas
        ref={canvasRef}
        width={1280}
        height={720}
        className="w-full h-full"
      />
      
      {isLoading && (
        <div className="absolute inset-0 flex items-center justify-center bg-black bg-opacity-75">
          <div className="text-white text-center">
            <div className="animate-spin mb-2">⚙️</div>
            <p>Loading camera...</p>
          </div>
        </div>
      )}

      {error && (
        <div className="absolute inset-0 flex items-center justify-center bg-black bg-opacity-75">
          <div className="text-red-400 text-center px-4">
            <p className="font-semibold mb-2">Error</p>
            <p>{error}</p>
          </div>
        </div>
      )}
    </div>
  );
}
