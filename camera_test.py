"""
Camera test: verifies the camera opens, delivers frames, and reports FPS.

Usage:
    python camera_test.py            # uses CAMERA_INDEX from .env
    python camera_test.py --index 1  # try a different camera
    python camera_test.py --no-window  # headless check (no GUI needed)

Press Q or ESC in the window to exit.
"""

import argparse
import sys
import time

import cv2

from agent import config
from agent.camera import CameraError, CameraStream

_QUIT_KEYS = {ord("q"), ord("Q"), 27}


def main() -> int:
    parser = argparse.ArgumentParser(description="kelvin6k camera test")
    parser.add_argument("--index", type=int, default=None, help="camera index (default: CAMERA_INDEX from .env)")
    parser.add_argument("--no-window", action="store_true", help="don't open a window; just verify frames")
    parser.add_argument("--seconds", type=float, default=5.0, help="headless test duration")
    args = parser.parse_args()

    index = config.CAMERA_INDEX if args.index is None else args.index
    is_rtsp = config.CAMERA_SOURCE == "rtsp"

    print("=== KELVIN6K CAMERA TEST ===")
    print(f"Camera source        : {'RTSP (EZVIZ / IP camera)' if is_rtsp else 'USB'}")
    if is_rtsp:
        try:
            from agent.camera import build_rtsp_url, mask_rtsp_url

            print(f"RTSP URL             : {mask_rtsp_url(build_rtsp_url())}")
            print(f"RTSP transport       : {config.CAMERA_RTSP_TRANSPORT}")
        except CameraError as exc:
            print(f"\nRESULT: FAILURE\n{exc}", file=sys.stderr)
            return 1
    else:
        print(f"Configured index     : {index}")
        print(f"Backend preference   : {config.CAMERA_BACKEND}")
        if config.CAMERA_NAME_HINT:
            print(f"Camera name hint     : {config.CAMERA_NAME_HINT}")
    print(f"Requested resolution : {config.CAMERA_WIDTH}x{config.CAMERA_HEIGHT} @ {config.CAMERA_FPS} fps")
    print()

    camera = CameraStream(index=index)
    try:
        camera.open()
    except CameraError as exc:
        print(f"\nRESULT: FAILURE\n{exc}", file=sys.stderr)
        print("\nRun `python find_cameras.py` to scan every index and backend.")
        return 1

    width, height = camera.actual_resolution

    frames = 0
    started = time.time()

    try:
        if args.no_window:
            print(f"\nCapturing for {args.seconds:.0f}s (headless)...")
            while time.time() - started < args.seconds:
                if camera.read() is not None:
                    frames += 1
        else:
            print("\nLive window open. Press Q or ESC to exit.")
            while True:
                frame = camera.read()
                if frame is None:
                    if cv2.waitKey(1) & 0xFF in _QUIT_KEYS:
                        break
                    continue
                frames += 1
                fps = camera.measured_fps
                cv2.putText(frame, f"FPS: {fps:.1f}", (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (120, 200, 255), 2)
                cv2.putText(frame, f"frames: {frames}", (12, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (235, 235, 235), 2)
                cv2.putText(frame, "Q / ESC to quit", (12, height - 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 155), 1)
                cv2.imshow("kelvin6k camera test", frame)
                if cv2.waitKey(1) & 0xFF in _QUIT_KEYS:
                    break
    except KeyboardInterrupt:
        pass
    finally:
        camera.release()
        if not args.no_window:
            cv2.destroyAllWindows()

    elapsed = max(1e-6, time.time() - started)
    print("\n--- results ---")
    print(f"Camera index    : {camera.index}" + (f" ({camera.device_name})" if camera.device_name else ""))
    print(f"Backend used    : {camera.backend_name}")
    print(f"Actual resolution: {width}x{height}")
    print(f"Frames received : {frames}")
    print(f"Average FPS     : {frames / elapsed:.1f}")
    print(f"Clean exit      : OK")
    print(f"\nRESULT: {'SUCCESS' if frames > 0 else 'FAILURE -- no frames received'}")
    return 0 if frames > 0 else 1


if __name__ == "__main__":
    sys.exit(main())
