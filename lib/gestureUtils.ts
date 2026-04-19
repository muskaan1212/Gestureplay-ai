// Types for hand landmarks
interface Landmark {
  x: number;
  y: number;
  z?: number;
}

// MediaPipe returns landmarks as a flat array of Landmark objects
type HandData = Landmark[] | { landmarks: Landmark[] } | null | undefined;

export type GestureType = 'none' | 'pinch' | 'palm' | 'fist' | 'scroll_up' | 'scroll_down' | 'cursor';
export type GesturePoint = { x: number; y: number };

// Helper to normalize hand data - MediaPipe can return landmarks as a flat array
function normalizeHand(hand: HandData): Landmark[] | null {
  if (!hand) return null;
  if (Array.isArray(hand)) return hand;
  if (hand.landmarks && Array.isArray(hand.landmarks)) return hand.landmarks;
  return null;
}

// Calculate distance between two points
export function calculateDistance(p1: Landmark, p2: Landmark): number {
  const z1 = p1.z ?? 0;
  const z2 = p2.z ?? 0;

  return Math.sqrt(
    Math.pow(p1.x - p2.x, 2) +
    Math.pow(p1.y - p2.y, 2) +
    Math.pow(z1 - z2, 2)
  );
}

export function calculatePointDistance(p1: GesturePoint, p2: GesturePoint): number {
  return Math.hypot(p1.x - p2.x, p1.y - p2.y);
}

export function isPinchDistance(
  thumbTip: Landmark,
  indexTip: Landmark,
  threshold: number = 0.055
): boolean {
  return calculateDistance(thumbTip, indexTip) < threshold;
}

export function landmarkToCanvasPoint(
  landmark: Landmark,
  canvasWidth: number,
  canvasHeight: number,
  mirrored: boolean = true
): GesturePoint {
  return {
    x: (mirrored ? 1 - landmark.x : landmark.x) * canvasWidth,
    y: landmark.y * canvasHeight,
  };
}

export function smoothPoint(
  previous: GesturePoint | null,
  next: GesturePoint,
  amount: number = 0.35
): GesturePoint {
  if (!previous) return next;

  return {
    x: previous.x + (next.x - previous.x) * amount,
    y: previous.y + (next.y - previous.y) * amount,
  };
}

function isFingerExtended(tip: Landmark, pip: Landmark): boolean {
  return tip.y < pip.y;
}

function isFingerCurled(tip: Landmark, pip: Landmark): boolean {
  return tip.y >= pip.y;
}

// Detect pinch gesture (thumb and index finger close together)
export function isPinch(hand: HandData): boolean {
  const landmarks = normalizeHand(hand);
  if (!landmarks || landmarks.length < 9) return false;
  
  const thumbTip = landmarks[4];
  const indexTip = landmarks[8];
  const distance = calculateDistance(thumbTip, indexTip);
  
  // Pinch when fingers are close (threshold tuned for typical hand size)
  return distance < 0.05;
}

// Detect palm gesture (open hand, fingers spread)
export function isPalm(hand: HandData): boolean {
  const landmarks = normalizeHand(hand);
  if (!landmarks || landmarks.length < 21) return false;
  
  const indexTip = landmarks[8];
  const middleTip = landmarks[12];
  const ringTip = landmarks[16];
  const pinkyTip = landmarks[20];
  const wrist = landmarks[0];
  
  // Check if hand is open - fingertips are above wrist and spread
  const fingersAboveWrist =
    indexTip.y < wrist.y &&
    middleTip.y < wrist.y &&
    ringTip.y < wrist.y &&
    pinkyTip.y < wrist.y;
  
  // Check finger spread
  const indexToRing = calculateDistance(indexTip, ringTip);
  const isSpread = indexToRing > 0.2;
  
  // Check if fingers aren't pinched together
  const notPinched = calculateDistance(indexTip, middleTip) > 0.03;
  
  return fingersAboveWrist && isSpread && notPinched;
}

export function isTwoFingerGesture(hand: HandData): boolean {
  const landmarks = normalizeHand(hand);
  if (!landmarks || landmarks.length < 21) return false;

  const indexExtended = isFingerExtended(landmarks[8], landmarks[6]);
  const middleExtended = isFingerExtended(landmarks[12], landmarks[10]);
  const ringCurled = isFingerCurled(landmarks[16], landmarks[14]);
  const pinkyCurled = isFingerCurled(landmarks[20], landmarks[18]);
  const fingersTogether = calculateDistance(landmarks[8], landmarks[12]) < 0.14;

  return indexExtended && middleExtended && ringCurled && pinkyCurled && fingersTogether;
}

// Detect fist gesture (all fingers curled)
export function isFist(hand: HandData): boolean {
  const landmarks = normalizeHand(hand);
  if (!landmarks || landmarks.length < 21) return false;
  
  const indexTip = landmarks[8];
  const middleTip = landmarks[12];
  const ringTip = landmarks[16];
  const pinkyTip = landmarks[20];
  const palm = landmarks[9];
  
  // Fingers should be below palm level (curled)
  const indexCurled = indexTip.y > palm.y;
  const middleCurled = middleTip.y > palm.y;
  const ringCurled = ringTip.y > palm.y;
  const pinkyCurled = pinkyTip.y > palm.y;
  
  return indexCurled && middleCurled && ringCurled && pinkyCurled;
}

// Detect scroll up gesture (hand moving upward)
export function detectScrollGesture(
  currentHand: HandData,
  previousHand: HandData | null,
  threshold: number = 0.03
): 'scroll_up' | 'scroll_down' | 'none' {
  const currLandmarks = normalizeHand(currentHand);
  const prevLandmarks = normalizeHand(previousHand);
  
  if (!currLandmarks || !prevLandmarks || currLandmarks.length < 10 || prevLandmarks.length < 10) {
    return 'none';
  }

  if (!isTwoFingerGesture(currLandmarks)) {
    return 'none';
  }
  
  const currentY = currLandmarks[9].y; // Palm center
  const previousY = prevLandmarks[9].y;
  
  const movement = previousY - currentY; // Positive = moving up
  
  if (movement > threshold) {
    return 'scroll_up';
  } else if (movement < -threshold) {
    return 'scroll_down';
  }
  
  return 'none';
}

// Get cursor position from index finger
export function getCursorPosition(hand: HandData, canvasWidth: number, canvasHeight: number) {
  const landmarks = normalizeHand(hand);
  if (!landmarks || landmarks.length < 9) return { x: 0, y: 0 };
  
  const indexTip = landmarks[8];
  
  // Convert normalized coordinates to canvas coordinates
  const x = indexTip.x * canvasWidth;
  const y = indexTip.y * canvasHeight;
  
  return { x, y };
}

// Main gesture detection function
export function detectGesture(hand: HandData): GestureType {
  if (!hand) return 'none';
  
  if (isFist(hand)) return 'fist';
  if (isPinch(hand)) return 'pinch';
  if (isPalm(hand)) return 'palm';
  
  return 'cursor';
}
