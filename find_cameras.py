"""
Scans every camera index/backend and reports which ones actually deliver frames.

On Windows it tests DirectShow first (the backend that works with external USB
cameras when Media Foundation throws OnReadSample errors), then MSMF, then ANY.

Usage:
    python find_cameras.py                 # scan + report working cameras
    python find_cameras.py --preview       # also show a live preview of each
    python find_cameras.py --preview 1     # preview only index 1
    python find_cameras.py --max-index 8   # scan further
    python find_cameras.py --modes 1       # list resolutions index 1 supports

Press Q or ESC to move past a preview.
"""

import argparse
import sys

import cv2

from agent import config
from agent.camera import CameraStream, list_camera_names, probe_cameras, supported_modes

_QUIT_KEYS = {ord("q"), ord("Q"), 27}


def _preview(index: int, seconds: float = 30.0) -> None:
    """Live preview so you can confirm which physical camera an index is."""
    stream = CameraStream(index=index, allow_scan=False)
    try:
        stream.open()
    except Exception as exc:
        print(f"  Could not preview index {index}: {exc}")
        return

    name = stream.device_name or f"index {index}"
    print(f"  Previewing index {index} ({name}) -- press Q or ESC to continue.")
    window = f"find_cameras -- index {index}"
    try:
        while True:
            frame = stream.read()
            if frame is None:
                if cv2.waitKey(1) & 0xFF in _QUIT_KEYS:
                    break
                continue
            label = f"index {index} | {stream.backend_name} | {stream.actual_resolution[0]}x{stream.actual_resolution[1]} | {stream.measured_fps:.1f} fps"
            cv2.putText(frame, label, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (120, 200, 255), 2)
            cv2.putText(frame, name[:60], (12, 54), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (235, 235, 235), 1)
            cv2.putText(frame, "Q / ESC to continue", (12, frame.shape[0] - 14),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 155), 1)
            cv2.imshow(window, frame)
            if cv2.waitKey(1) & 0xFF in _QUIT_KEYS:
                break
    except KeyboardInterrupt:
        pass
    finally:
        stream.release()
        cv2.destroyAllWindows()


def main() -> int:
    parser = argparse.ArgumentParser(description="Scan for working USB cameras")
    parser.add_argument("--max-index", type=int, default=config.CAMERA_SCAN_MAX_INDEX)
    parser.add_argument("--preview", nargs="?", const=-1, type=int, default=None,
                        help="preview all working cameras, or a specific index")
    parser.add_argument("--modes", type=int, default=None,
                        help="list the resolutions a given index actually supports")
    args = parser.parse_args()

    print("=== kelvin6k camera scanner ===")
    print(f"Platform backend order: {'DirectShow -> MSMF -> ANY (Windows)' if sys.platform.startswith('win') else 'V4L2 -> ANY'}")

    names = list_camera_names()
    if names:
        print("\nDevices reported by Windows (DirectShow order):")
        for i, name in enumerate(names):
            print(f"  index {i}: {name}")
    else:
        print("\nDevice names unavailable (install with: pip install pygrabber)")

    if args.modes is not None:
        print(f"\nProbing supported resolutions on index {args.modes}...")
        modes = supported_modes(args.modes)
        if modes:
            for width, height in modes:
                print(f"  {width}x{height}")
        else:
            print("  none -- the camera did not deliver frames at any tested mode")
        return 0

    print(f"\nScanning indexes 0..{args.max_index} (this can take a few seconds)...\n")
    results = probe_cameras(args.max_index)

    working = [r for r in results if r["working"]]
    for r in results:
        if r["working"]:
            w, h = r["resolution"]
            label = f" ({r['name']})" if r["name"] else ""
            print(f"  [WORKING] index {r['index']}{label}: {w}x{h} @ {r['fps']} fps via {r['backend']}")
        else:
            label = f" ({r['name']})" if r["name"] else ""
            print(f"  [ -- ]    index {r['index']}{label}: no frames")

    if not working:
        print("\nRESULT: no working camera found.")
        print("\nChecks:")
        print("  - Is the external USB camera plugged in? Try a different USB port.")
        print("  - Close anything else using it (Camera app, Teams, Zoom, OBS, browser).")
        print("  - Settings > Privacy & security > Camera: allow desktop apps.")
        print("  - Some USB cameras need a moment after plug-in; re-run this script.")
        return 1

    print(f"\nRESULT: {len(working)} working camera(s).")
    best = working[0]
    print(f"\nTo use index {best['index']}, set this in your .env:")
    print(f"  CAMERA_INDEX={best['index']}")
    if best["resolution"]:
        print(f"  CAMERA_WIDTH={best['resolution'][0]}")
        print(f"  CAMERA_HEIGHT={best['resolution'][1]}")
    if best["name"]:
        hint = best["name"].split()[0]
        print(f"  CAMERA_NAME_HINT={hint}   # sticks to this camera even if indexes shuffle")

    if args.preview is not None:
        targets = [args.preview] if args.preview >= 0 else [r["index"] for r in working]
        print()
        for index in targets:
            _preview(index)

    return 0


if __name__ == "__main__":
    sys.exit(main())
