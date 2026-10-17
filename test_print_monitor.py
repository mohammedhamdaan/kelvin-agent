"""
End-to-end pipeline test:
    Camera -> Frame -> ROI -> vision pipeline -> layer detection -> width
    -> debounced print status -> query API

Runs against the live camera when one is configured, and always runs a
synthetic-frame test so the detection/debouncing logic can be verified without
hardware.

Usage:
    python test_print_monitor.py              # synthetic + live camera (if available)
    python test_print_monitor.py --synthetic  # synthetic only, no camera needed
    python test_print_monitor.py --frames 60  # how many live frames to process
"""

import argparse
import sys
import time

import cv2
import numpy as np

from agent import config
from agent.camera import CameraError, CameraStream
from agent.print_monitor import (
    PrintMonitor,
    get_camera_status,
    get_current_layer_status,
    get_current_layer_width,
    get_current_printing_status,
)
from agent.state import PrinterState
from agent.vision import VisionPipeline


def _synthetic_frame(n_layers=12, bead_h=20, gap=4, width=160, jitter=0):
    """A stand-in printed wall: stacked bright layer bands on a dark background."""
    img = np.full((480, 640, 3), 20, np.uint8)
    for i in range(n_layers):
        y = 80 + i * (bead_h + gap) + (jitter if i % 2 else 0)
        cv2.rectangle(img, (240, y), (240 + width, y + bead_h), (200, 200, 200), -1)
    return img


def run_synthetic() -> bool:
    print("=== SYNTHETIC PIPELINE TEST (no camera required) ===")
    pipeline = VisionPipeline()
    monitor = PrintMonitor()
    state = PrinterState()
    ok = True

    # 1. A printing machine: layers present, frame changing between reads.
    print("\n[1] Simulating an active print (layers present, frame changing)...")
    for i in range(max(config.LAYER_DETECT_FRAMES, config.PRINT_ACTIVE_FRAMES) + 3):
        frame = _synthetic_frame(width=160 + (i % 3) * 6, jitter=i % 2)
        analysis = pipeline.analyze(frame)
        obs = monitor.update(frame, analysis, camera_connected=True)
        state.update_vision(analysis, 30.0)
        state.update_print_observation(obs)

    printing = get_current_printing_status(state)
    layer = get_current_layer_status(state)
    width = get_current_layer_width(state)
    print(f"    printing        : {printing['printing']}  ({printing['status']})")
    print(f"    layer_detected  : {layer['layer_detected']}  ({layer['status']}, conf {layer['confidence']})")
    print(f"    width px        : {width['width_px']}")
    print(f"    width mm        : {width['width']}  {'<- ' + width['reason'] if width['reason'] else ''}")
    print(f"    camera          : {get_camera_status(state)}")

    if not layer["layer_detected"]:
        print("    FAIL: a layer should have been detected after the debounce window.")
        ok = False
    if width["width_px"] is None:
        print("    FAIL: expected a raw pixel width measurement.")
        ok = False
    if config.PIXELS_PER_MM <= 0 and width["width"] is not None:
        print("    FAIL: mm value reported without calibration.")
        ok = False

    # 2. Layer disappears -> must debounce to LAYER_MISSING, not flip instantly.
    print("\n[2] Layer disappears (blank frames)...")
    blank = np.full((480, 640, 3), 20, np.uint8)
    obs_after_one = monitor.update(blank, pipeline.analyze(blank), camera_connected=True)
    print(f"    after 1 blank frame : layer_detected={obs_after_one.layer_detected} "
          f"(should still be True -- debouncing)")
    if not obs_after_one.layer_detected:
        print("    FAIL: state flipped on a single bad frame; debouncing is not working.")
        ok = False

    for _ in range(config.LAYER_MISSING_FRAMES + 2):
        obs = monitor.update(blank, pipeline.analyze(blank), camera_connected=True)
    state.update_print_observation(obs)
    print(f"    after {config.LAYER_MISSING_FRAMES}+ blank frames: layer_detected={obs.layer_detected} "
          f"status={obs.layer_status.value}")
    if obs.layer_detected:
        print("    FAIL: layer should have been marked missing by now.")
        ok = False

    # 3. Camera failure path.
    print("\n[3] Camera failure...")
    obs = monitor.update(None, None, camera_connected=False)
    state.update_print_observation(obs)
    print(f"    status: {obs.status.value} (expected CAMERA_ERROR)")
    if obs.status.value != "CAMERA_ERROR":
        print("    FAIL: camera failure not reported as CAMERA_ERROR.")
        ok = False

    # 4. Output contract.
    print("\n[4] Pipeline output contract:")
    required = {"printing", "layer_detected", "layer_width", "width_unit", "confidence", "timestamp"}
    payload = obs.to_dict()
    missing = required - set(payload)
    print(f"    keys: {sorted(payload)}")
    if missing:
        print(f"    FAIL: missing keys {missing}")
        ok = False

    return ok


def run_live(max_frames: int) -> bool:
    print("\n=== LIVE CAMERA PIPELINE TEST ===")
    print(f"Camera source: {config.CAMERA_SOURCE.upper()}")

    camera = CameraStream()
    try:
        camera.open()
    except CameraError as exc:
        print(f"Camera unavailable -- skipping live test.\n{str(exc).splitlines()[0]}")
        return True  # not a pipeline failure

    pipeline = VisionPipeline()
    monitor = PrintMonitor()
    state = PrinterState()
    processed = 0
    started = time.time()

    try:
        while processed < max_frames:
            frame = camera.read()
            if frame is None:
                if time.time() - started > 20:
                    break
                continue
            analysis = pipeline.analyze(frame)
            obs = monitor.update(frame, analysis, camera_connected=True)
            state.update_vision(analysis, camera.measured_fps)
            state.update_print_observation(obs)
            processed += 1
    finally:
        camera.release()

    if processed == 0:
        print("No frames processed -- check the camera.")
        return False

    obs = state.latest_print_observation()
    print(f"Frames processed : {processed}")
    print(f"Measured FPS     : {camera.measured_fps:.1f}")
    print(f"ROI              : {analysis.roi or 'full frame'}")
    print(f"Printing         : {obs.printing} ({obs.status.value})")
    print(f"Layer            : {obs.layer_status.value} (confidence {obs.confidence:.2f})")
    print(f"Width (px)       : {obs.layer_width_px}")
    print(f"Width (mm)       : {obs.layer_width}")
    print(f"Motion score     : {obs.motion_score:.2f}")
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description="kelvin6k print-monitor pipeline test")
    parser.add_argument("--synthetic", action="store_true", help="skip the live camera test")
    parser.add_argument("--frames", type=int, default=40, help="live frames to process")
    args = parser.parse_args()

    ok = run_synthetic()
    if not args.synthetic:
        ok = run_live(args.frames) and ok

    print("\n=== RESULT:", "PASS" if ok else "FAIL", "===")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
