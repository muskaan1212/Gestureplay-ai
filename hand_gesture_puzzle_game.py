"""
Hand Gesture Controlled Puzzle Game

Run:
    python hand_gesture_puzzle_game.py

Controls:
    - Selection screen: use both index fingertips as opposite crop corners.
    - Pinch thumb + index finger to capture the selected region.
    - Puzzle screen: pinch over a tile to pick it up, move, then release over
      another tile to swap.
    - Hold both hands pinching for a moment to recapture.
    - Keyboard fallback: R recaptures, Q quits.
"""

from __future__ import annotations

import random
import time
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import mediapipe as mp
import numpy as np


Point = Tuple[int, int]
Rect = Tuple[int, int, int, int]


@dataclass
class GameConfig:
    camera_index: int = 0
    frame_width: int = 1280
    frame_height: int = 720
    grid_size: int = 3
    min_selection_size: int = 140
    pinch_threshold_px: int = 44
    pinch_stable_frames: int = 5
    action_cooldown_sec: float = 0.45
    recapture_hold_sec: float = 1.2
    detection_confidence: float = 0.65
    tracking_confidence: float = 0.65


@dataclass
class HandInfo:
    landmarks: Sequence
    index_tip: Point
    thumb_tip: Point
    middle_tip: Point
    wrist: Point
    confidence: float
    label: str


class GameState(Enum):
    SELECT_REGION = "select_region"
    PUZZLE = "puzzle"
    SOLVED = "solved"


class HandTracker:
    """MediaPipe Hands wrapper that returns pixel-space landmark data."""

    def __init__(self, config: GameConfig):
        self.config = config
        self.mp_hands = mp.solutions.hands
        self.mp_draw = mp.solutions.drawing_utils
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            model_complexity=1,
            min_detection_confidence=config.detection_confidence,
            min_tracking_confidence=config.tracking_confidence,
        )

    def process(self, frame_bgr: np.ndarray) -> List[HandInfo]:
        height, width = frame_bgr.shape[:2]
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        results = self.hands.process(frame_rgb)

        if not results.multi_hand_landmarks:
            return []

        hands: List[HandInfo] = []
        handedness = results.multi_handedness or []

        for hand_index, hand_landmarks in enumerate(results.multi_hand_landmarks):
            score = handedness[hand_index].classification[0].score if hand_index < len(handedness) else 1.0
            label = handedness[hand_index].classification[0].label if hand_index < len(handedness) else "Hand"

            if score < self.config.detection_confidence:
                continue

            def pixel(landmark_id: int) -> Point:
                landmark = hand_landmarks.landmark[landmark_id]
                return int(landmark.x * width), int(landmark.y * height)

            hands.append(
                HandInfo(
                    landmarks=hand_landmarks.landmark,
                    index_tip=pixel(8),
                    thumb_tip=pixel(4),
                    middle_tip=pixel(12),
                    wrist=pixel(0),
                    confidence=score,
                    label=label,
                )
            )

        return hands

    def close(self) -> None:
        self.hands.close()


class GestureDetector:
    """Small stateful gesture helper with smoothing and cooldown guards."""

    def __init__(self, config: GameConfig):
        self.config = config
        self.pinch_frames: Dict[str, int] = {}
        self.last_action_time = 0.0
        self.smoothed_points: Dict[str, Point] = {}

    @staticmethod
    def distance(a: Point, b: Point) -> float:
        return float(np.hypot(a[0] - b[0], a[1] - b[1]))

    def smooth_point(self, key: str, point: Point, alpha: float = 0.38) -> Point:
        previous = self.smoothed_points.get(key)

        if previous is None:
            self.smoothed_points[key] = point
            return point

        smoothed = (
            int(previous[0] + (point[0] - previous[0]) * alpha),
            int(previous[1] + (point[1] - previous[1]) * alpha),
        )
        self.smoothed_points[key] = smoothed
        return smoothed

    def is_pinching_raw(self, hand: HandInfo) -> bool:
        return self.distance(hand.thumb_tip, hand.index_tip) <= self.config.pinch_threshold_px

    def is_stable_pinch(self, hand: HandInfo, key: str) -> bool:
        if self.is_pinching_raw(hand):
            self.pinch_frames[key] = self.pinch_frames.get(key, 0) + 1
        else:
            self.pinch_frames[key] = 0

        return self.pinch_frames[key] >= self.config.pinch_stable_frames

    def can_fire_action(self) -> bool:
        return time.time() - self.last_action_time >= self.config.action_cooldown_sec

    def mark_action(self) -> None:
        self.last_action_time = time.time()


class PuzzleBoard:
    """Swap-based tile puzzle. Each board slot stores an original tile index."""

    def __init__(self, image: np.ndarray, grid_size: int):
        self.grid_size = grid_size
        self.source = self._make_square(image)
        self.tiles = self._split_tiles(self.source)
        self.order = list(range(len(self.tiles)))
        self.selected_slot: Optional[int] = None
        self.drag_position: Optional[Point] = None
        self.shuffle()

    @staticmethod
    def _make_square(image: np.ndarray) -> np.ndarray:
        height, width = image.shape[:2]
        size = min(height, width)
        x0 = (width - size) // 2
        y0 = (height - size) // 2
        return image[y0 : y0 + size, x0 : x0 + size].copy()

    def _split_tiles(self, image: np.ndarray) -> List[np.ndarray]:
        tile_size = image.shape[0] // self.grid_size
        tiles: List[np.ndarray] = []

        for row in range(self.grid_size):
            for col in range(self.grid_size):
                y0 = row * tile_size
                x0 = col * tile_size
                tiles.append(image[y0 : y0 + tile_size, x0 : x0 + tile_size].copy())

        return tiles

    def shuffle(self) -> None:
        while True:
            random.shuffle(self.order)
            if not self.is_solved():
                break

    def is_solved(self) -> bool:
        return all(tile_index == slot for slot, tile_index in enumerate(self.order))

    def slot_at_point(self, point: Point, board_rect: Rect) -> Optional[int]:
        x, y, size, _ = board_rect
        px, py = point

        if px < x or py < y or px >= x + size or py >= y + size:
            return None

        tile_size = size // self.grid_size
        col = (px - x) // tile_size
        row = (py - y) // tile_size
        return int(row * self.grid_size + col)

    def pick(self, slot: Optional[int]) -> bool:
        if slot is None:
            return False

        self.selected_slot = slot
        return True

    def update_drag(self, point: Point) -> None:
        self.drag_position = point

    def drop(self, target_slot: Optional[int]) -> None:
        if self.selected_slot is not None and target_slot is not None:
            self.order[self.selected_slot], self.order[target_slot] = (
                self.order[target_slot],
                self.order[self.selected_slot],
            )

        self.selected_slot = None
        self.drag_position = None

    def render(self, frame: np.ndarray, board_rect: Rect) -> None:
        x, y, size, _ = board_rect
        tile_size = size // self.grid_size

        for slot, tile_index in enumerate(self.order):
            if slot == self.selected_slot:
                continue

            row, col = divmod(slot, self.grid_size)
            tx = x + col * tile_size
            ty = y + row * tile_size
            tile = cv2.resize(self.tiles[tile_index], (tile_size, tile_size))
            frame[ty : ty + tile_size, tx : tx + tile_size] = tile
            cv2.rectangle(frame, (tx, ty), (tx + tile_size, ty + tile_size), (30, 240, 255), 1)

        if self.selected_slot is not None and self.drag_position is not None:
            tile_index = self.order[self.selected_slot]
            tile = cv2.resize(self.tiles[tile_index], (tile_size, tile_size))
            half = tile_size // 2
            cx, cy = self.drag_position
            x0 = max(0, min(frame.shape[1] - tile_size, cx - half))
            y0 = max(0, min(frame.shape[0] - tile_size, cy - half))
            roi = frame[y0 : y0 + tile_size, x0 : x0 + tile_size]
            blended = cv2.addWeighted(roi, 0.25, tile, 0.75, 0)
            frame[y0 : y0 + tile_size, x0 : x0 + tile_size] = blended
            cv2.rectangle(frame, (x0, y0), (x0 + tile_size, y0 + tile_size), (255, 80, 220), 4)

        cv2.rectangle(frame, (x, y), (x + size, y + size), (255, 255, 255), 2)


class GameUI:
    """Drawing utilities for overlays and status panels."""

    @staticmethod
    def put_text(
        frame: np.ndarray,
        text: str,
        origin: Point,
        scale: float = 0.7,
        color: Tuple[int, int, int] = (255, 255, 255),
        thickness: int = 2,
    ) -> None:
        cv2.putText(frame, text, origin, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thickness + 3, cv2.LINE_AA)
        cv2.putText(frame, text, origin, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)

    @staticmethod
    def draw_panel(frame: np.ndarray, rect: Rect, alpha: float = 0.62) -> None:
        x, y, w, h = rect
        overlay = frame.copy()
        cv2.rectangle(overlay, (x, y), (x + w, y + h), (15, 18, 25), -1)
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (45, 220, 255), 1)

    def draw_header(self, frame: np.ndarray, state: GameState, status: str, fps: float) -> None:
        self.draw_panel(frame, (18, 16, 560, 116))
        self.put_text(frame, "Hand Gesture Controlled Puzzle Game", (34, 52), 0.82, (40, 255, 255), 2)
        self.put_text(frame, f"Mode: {state.value.replace('_', ' ').title()} | {status}", (34, 84), 0.56)
        self.put_text(frame, f"FPS: {fps:.1f}", (34, 112), 0.56, (180, 255, 180))

    def draw_instructions(self, frame: np.ndarray, state: GameState) -> None:
        height = frame.shape[0]
        self.draw_panel(frame, (18, height - 112, 640, 92), 0.55)

        if state == GameState.SELECT_REGION:
            lines = [
                "Use both index fingertips as crop corners.",
                "Pinch thumb + index to capture. Minimum selection is enforced.",
            ]
        else:
            lines = [
                "Pinch over a tile to pick it up. Release over another tile to swap.",
                "Hold both hands pinching to recapture. Keyboard fallback: R restart, Q quit.",
            ]

        for i, line in enumerate(lines):
            self.put_text(frame, line, (34, height - 76 + i * 30), 0.55, (235, 235, 235), 1)

    def draw_selection(self, frame: np.ndarray, rect: Optional[Rect], valid: bool, countdown: Optional[float]) -> None:
        if rect is None:
            return

        x, y, w, h = rect
        color = (60, 255, 80) if valid else (70, 70, 255)
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 3)

        if countdown is not None:
            self.put_text(frame, f"Capturing in {countdown:.1f}s", (x, max(36, y - 14)), 0.75, (255, 255, 80), 2)

    def draw_solved(self, frame: np.ndarray) -> None:
        height, width = frame.shape[:2]
        pulse = int(80 + 80 * abs(np.sin(time.time() * 6)))
        self.draw_panel(frame, (width // 2 - 290, height // 2 - 68, 580, 136), 0.72)
        cv2.rectangle(
            frame,
            (width // 2 - 290, height // 2 - 68),
            (width // 2 + 290, height // 2 + 68),
            (pulse, 255, 255),
            4,
        )
        self.put_text(frame, "Puzzle Solved!", (width // 2 - 160, height // 2 - 8), 1.2, (60, 255, 255), 3)
        self.put_text(frame, "Hold both hands pinching or press R to play again.", (width // 2 - 238, height // 2 + 36), 0.55)


class HandGesturePuzzleGame:
    def __init__(self, config: GameConfig):
        self.config = config
        self.tracker = HandTracker(config)
        self.gestures = GestureDetector(config)
        self.ui = GameUI()
        self.state = GameState.SELECT_REGION
        self.puzzle: Optional[PuzzleBoard] = None
        self.capture_started_at: Optional[float] = None
        self.recapture_started_at: Optional[float] = None
        self.tile_was_pinching = False
        self.status = "Waiting for two hands"
        self.prev_time = time.time()
        self.fps = 0.0

    @staticmethod
    def normalized_rect(a: Point, b: Point) -> Rect:
        x0, y0 = min(a[0], b[0]), min(a[1], b[1])
        x1, y1 = max(a[0], b[0]), max(a[1], b[1])
        return x0, y0, x1 - x0, y1 - y0

    def board_rect(self, frame: np.ndarray) -> Rect:
        height, width = frame.shape[:2]
        size = int(min(width * 0.42, height * 0.74))
        x = width - size - 26
        y = max(146, (height - size) // 2)
        return x, y, size, size

    def reset_to_capture(self) -> None:
        self.state = GameState.SELECT_REGION
        self.puzzle = None
        self.capture_started_at = None
        self.recapture_started_at = None
        self.tile_was_pinching = False
        self.status = "Waiting for two hands"

    def crop_region(self, frame: np.ndarray, rect: Rect) -> Optional[np.ndarray]:
        x, y, w, h = rect
        height, width = frame.shape[:2]
        x = max(0, min(width - 1, x))
        y = max(0, min(height - 1, y))
        w = max(0, min(width - x, w))
        h = max(0, min(height - y, h))

        if w < self.config.min_selection_size or h < self.config.min_selection_size:
            return None

        return frame[y : y + h, x : x + w].copy()

    def handle_region_selection(self, frame: np.ndarray, hands: List[HandInfo]) -> None:
        rect: Optional[Rect] = None
        valid = False
        countdown = None

        if len(hands) >= 2:
            p1 = self.gestures.smooth_point("selection_0", hands[0].index_tip)
            p2 = self.gestures.smooth_point("selection_1", hands[1].index_tip)
            rect = self.normalized_rect(p1, p2)
            valid = rect[2] >= self.config.min_selection_size and rect[3] >= self.config.min_selection_size
            pinching = any(self.gestures.is_stable_pinch(hand, f"capture_{i}") for i, hand in enumerate(hands))

            self.status = "Pinch to capture" if valid else "Selection too small"

            if valid and pinching and self.gestures.can_fire_action():
                if self.capture_started_at is None:
                    self.capture_started_at = time.time()

                countdown = max(0.0, 0.8 - (time.time() - self.capture_started_at))

                if countdown <= 0:
                    crop = self.crop_region(frame, rect)

                    if crop is not None:
                        self.puzzle = PuzzleBoard(crop, self.config.grid_size)
                        self.state = GameState.PUZZLE
                        self.status = "Puzzle shuffled"
                        self.gestures.mark_action()
                        self.capture_started_at = None
                        return
            else:
                self.capture_started_at = None
        else:
            self.status = "Show both hands"
            self.capture_started_at = None

        self.ui.draw_selection(frame, rect, valid, countdown)

    def handle_puzzle(self, frame: np.ndarray, hands: List[HandInfo]) -> None:
        if self.puzzle is None:
            self.reset_to_capture()
            return

        board_rect = self.board_rect(frame)
        self.puzzle.render(frame, board_rect)

        if len(hands) >= 2 and all(self.gestures.is_pinching_raw(hand) for hand in hands[:2]):
            if self.recapture_started_at is None:
                self.recapture_started_at = time.time()
            elif time.time() - self.recapture_started_at >= self.config.recapture_hold_sec:
                self.reset_to_capture()
                return
        else:
            self.recapture_started_at = None

        if not hands:
            self.status = "Hand lost"
            self.tile_was_pinching = False
            return

        hand = hands[0]
        pointer = self.gestures.smooth_point("puzzle_pointer", hand.index_tip)
        pinching = self.gestures.is_stable_pinch(hand, "tile_drag")
        target_slot = self.puzzle.slot_at_point(pointer, board_rect)

        cv2.circle(frame, pointer, 10, (255, 80, 220) if pinching else (45, 240, 255), -1)

        if pinching and not self.tile_was_pinching and self.gestures.can_fire_action():
            if self.puzzle.pick(target_slot):
                self.gestures.mark_action()
                self.status = "Tile picked"

        if pinching and self.puzzle.selected_slot is not None:
            self.puzzle.update_drag(pointer)
            self.status = "Dragging tile"

        if not pinching and self.tile_was_pinching:
            self.puzzle.drop(target_slot)
            self.gestures.mark_action()
            self.status = "Tile dropped"

            if self.puzzle.is_solved():
                self.state = GameState.SOLVED
                self.status = "Solved"

        self.tile_was_pinching = pinching

    def update_fps(self) -> None:
        now = time.time()
        dt = max(1e-6, now - self.prev_time)
        instant_fps = 1.0 / dt
        self.fps = self.fps * 0.9 + instant_fps * 0.1 if self.fps else instant_fps
        self.prev_time = now

    def run(self) -> None:
        cap = cv2.VideoCapture(self.config.camera_index)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.frame_width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.frame_height)

        if not cap.isOpened():
            raise RuntimeError("Could not open webcam. Try changing camera_index in GameConfig.")

        window_name = "Hand Gesture Controlled Puzzle Game"

        try:
            while True:
                ok, frame = cap.read()

                if not ok:
                    self.status = "Camera frame unavailable"
                    continue

                frame = cv2.flip(frame, 1)
                hands = self.tracker.process(frame)
                self.update_fps()

                if self.state == GameState.SELECT_REGION:
                    self.handle_region_selection(frame, hands)
                elif self.state in (GameState.PUZZLE, GameState.SOLVED):
                    self.handle_puzzle(frame, hands)

                    if self.state == GameState.SOLVED:
                        self.ui.draw_solved(frame)

                self.ui.draw_header(frame, self.state, self.status, self.fps)
                self.ui.draw_instructions(frame, self.state)

                cv2.imshow(window_name, frame)
                key = cv2.waitKey(1) & 0xFF

                if key == ord("q"):
                    break
                if key == ord("r"):
                    self.reset_to_capture()
        finally:
            cap.release()
            self.tracker.close()
            cv2.destroyAllWindows()


def main() -> None:
    config = GameConfig()
    game = HandGesturePuzzleGame(config)
    game.run()


if __name__ == "__main__":
    main()
