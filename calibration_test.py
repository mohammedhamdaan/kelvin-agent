"""
ArUco calibration + measurement test using a real camera.

What this proves, live, with no simulated values anywhere:

  1. the camera opens and delivers frames
  2. a DICT_5X5_50 ArUco marker is detected in the frame
  3. the detection is confirmed (marker id + corner geometry)
  4. pixels_per_mm is computed from the marker's known physical size
  5. a calibration baseline y_0 (the marker's bottom edge, i.e. the build
     surface) is established and locked once the measurement is stable
  6. the frame is shown with the detection and calibration drawn on it
  7. the printed part / test object is segmented (marker excluded)
  8. its width and height are converted to millimetres
  9. results are printed in the terminal

Every value printed is measured from the current frame. When a step cannot be
completed, the exact reason is printed and the dependent values are shown as
"unavailable" -- never as a number.

Usage:
    python calibration_test.py                    # built-in camera, 30 mm marker
    python calibration_test.py --marker-mm 40     # different printed marker size
    python calibration_test.py --index 1          # a different camera
    python calibration_test.py --no-window        # headless (no GUI)

Keys in the window:
    Q / ESC  quit
    B        re-lock the calibration baseline on the current frame
    S        save the current annotated frame as a PNG
"""

import argparse
import statistics
import sys
import time
from collections import deque
from typing import Optional

import cv2
import numpy as np

from agent.camera import CameraError, CameraStream
from agent.segmentation import segment_part

_QUIT_KEYS = {ord("q"), ord("Q"), 27}

# The baseline locks once this many consecutive frames agree on pixels_per_mm
# to within LOCK_TOLERANCE. A single frame is not enough -- marker corners
# jitter by a pixel or two, and locking on jitter bakes that error into every
# later measurement.
LOCK_FRAMES = 15
LOCK_TOLERANCE = 0.02  # 2 % spread across the window

# A marker viewed at a steep angle measures short on one axis. Above this
# side-length imbalance the calibration is reported as unreliable rather than
# used silently.
MAX_SIDE_IMBALANCE = 0.15  # 15 %

# Pixels of margin when masking the marker out of the object search.
MARKER_MASK_PAD = 8


class MarkerMeasurement:
    """One marker's geometry in the current frame."""

    def __init__(self, marker_id: int, corners: np.ndarray, marker_mm: float):
        self.id = int(marker_id)
        self.corners = corners.reshape(4, 2).astype(np.float32)

        # Side lengths, in order: TL-TR, TR-BR, BR-BL, BL-TL.
        self.sides = [
            float(np.linalg.norm(self.corners[i] - self.corners[(i + 1) % 4]))
            for i in range(4)
        ]
        self.mean_side_px = float(np.mean(self.sides))
        self.imbalance = (
            (max(self.sides) - min(self.sides)) / max(self.sides)
            if max(self.sides) > 0
            else 1.0
        )
        self.pixels_per_mm = self.mean_side_px / marker_mm if marker_mm > 0 else 0.0

        # The build surface: the marker's lower edge in image coordinates.
        self.baseline_y = float(np.max(self.corners[:, 1]))
        self.centre = self.corners.mean(axis=0)

    @property
    def reliable(self) -> bool:
        return self.imbalance <= MAX_SIDE_IMBALANCE and self.pixels_per_mm > 0


def build_detector(dict_name: str):
    """Builds an ArUco detector across both OpenCV API generations."""
    if not hasattr(cv2, "aruco"):
        raise RuntimeError(
            "This OpenCV build has no cv2.aruco module. Install a build that "
            "includes it:  pip install opencv-contrib-python"
        )
    if not hasattr(cv2.aruco, dict_name):
        raise RuntimeError(f"cv2.aruco has no dictionary named {dict_name}.")

    dictionary = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, dict_name))

    if hasattr(cv2.aruco, "ArucoDetector"):  # OpenCV >= 4.7
        params = cv2.aruco.DetectorParameters()
        detector = cv2.aruco.ArucoDetector(dictionary, params)
        return lambda gray: detector.detectMarkers(gray)

    params = cv2.aruco.DetectorParameters_create()  # OpenCV < 4.7
    return lambda gray: cv2.aruco.detectMarkers(gray, dictionary, parameters=params)


def mask_out_marker(gray: np.ndarray, marker: MarkerMeasurement) -> np.ndarray:
    """
    Removes the marker from the image before the object is segmented.

    The marker is high-contrast black-and-white, so leaving it in guarantees it
    becomes the "detected object" and the reported width is the marker's, not
    the printed part's. It is filled with the frame's median grey so it blends
    into the background instead of becoming a new dark blob.
    """
    out = gray.copy()
    polygon = marker.corners.astype(np.int32)
    filler = np.zeros(gray.shape, dtype=np.uint8)
    cv2.fillPoly(filler, [polygon], 255)
    if MARKER_MASK_PAD > 0:
        kernel = np.ones((MARKER_MASK_PAD * 2 + 1, MARKER_MASK_PAD * 2 + 1), np.uint8)
        filler = cv2.dilate(filler, kernel)
    out[filler > 0] = int(np.median(gray))
    return out


def draw_overlay(
    frame: np.ndarray,
    marker: Optional[MarkerMeasurement],
    locked_ppm: Optional[float],
    locked_y0: Optional[float],
    object_bbox,
    object_mm,
    status: str,
) -> np.ndarray:
    height, width = frame.shape[:2]

    if marker is not None:
        polygon = marker.corners.astype(np.int32)
        colour = (0, 230, 120) if marker.reliable else (0, 170, 255)
        cv2.polylines(frame, [polygon], True, colour, 2)
        cx, cy = marker.centre.astype(int)
        cv2.putText(frame, f"id {marker.id}", (cx - 22, cy),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, colour, 2)
        cv2.putText(frame, f"{marker.mean_side_px:.1f} px side",
                    (polygon[0][0], max(18, polygon[0][1] - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour, 1)

    if locked_y0 is not None:
        y = int(round(locked_y0))
        cv2.line(frame, (0, y), (width, y), (255, 190, 60), 2)
        cv2.putText(frame, "baseline y_0", (10, max(16, y - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 190, 60), 1)

    if object_bbox is not None:
        x, y, w, h = object_bbox
        cv2.rectangle(frame, (x, y), (x + w, y + h), (80, 160, 255), 2)
        if object_mm is not None:
            w_mm, h_mm, _above = object_mm
            cv2.putText(frame, f"{w_mm:.1f} x {h_mm:.1f} mm", (x, max(16, y - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (80, 160, 255), 2)

    lines = [status]
    if locked_ppm is not None:
        lines.append(f"pixels_per_mm {locked_ppm:.4f}  (1 mm = {locked_ppm:.2f} px)")
    else:
        lines.append("pixels_per_mm not established")
    for i, text in enumerate(lines):
        cv2.putText(frame, text, (10, 24 + i * 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (235, 235, 235), 2)

    cv2.putText(frame, "Q quit   B re-lock baseline   S save frame",
                (10, height - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (150, 150, 155), 1)
    return frame


def main() -> int:
    parser = argparse.ArgumentParser(description="kelvin6k ArUco calibration + measurement test")
    parser.add_argument("--index", type=int, default=0,
                        help="camera index (default 0 = the built-in laptop camera)")
    parser.add_argument("--marker-mm", type=float, default=30.0,
                        help="the printed marker's real side length in mm (default 30)")
    parser.add_argument("--dict", default="DICT_5X5_50", help="ArUco dictionary name")
    parser.add_argument("--no-window", action="store_true", help="headless; no GUI window")
    parser.add_argument("--seconds", type=float, default=20.0,
                        help="headless run duration in seconds")
    args = parser.parse_args()

    print("=== KELVIN6K ARUCO CALIBRATION TEST ===")
    print(f"Camera index    : {args.index}")
    print(f"Marker dict     : {args.dict}")
    print(f"Marker size     : {args.marker_mm} mm per side")
    print()

    if args.marker_mm <= 0:
        print("RESULT: FAILURE\n--marker-mm must be a positive length.", file=sys.stderr)
        return 1

    try:
        detect = build_detector(args.dict)
    except RuntimeError as exc:
        print(f"RESULT: FAILURE\nStep 2 (marker detection) cannot run: {exc}", file=sys.stderr)
        return 1

    # Explicit index, no name hint, no scanning: this test must open the camera
    # the user asked for, not whichever device the .env hint prefers.
    camera = CameraStream(index=args.index, name_hint="", allow_scan=False, source="usb")
    try:
        camera.open()
    except CameraError as exc:
        print(f"RESULT: FAILURE\nStep 1 (open the camera) failed:\n{exc}", file=sys.stderr)
        print("\nRun `python find_cameras.py` to see which indexes exist.")
        return 1

    width, height = camera.actual_resolution
    print(f"Camera opened   : index {camera.index}"
          + (f" ({camera.device_name})" if camera.device_name else ""))
    print(f"Backend         : {camera.backend_name}")
    print(f"Resolution      : {width}x{height}")
    print()
    print("Hold the printed marker flat and facing the camera.")
    print("Calibration locks automatically once the measurement is stable.\n")

    ppm_window: deque = deque(maxlen=LOCK_FRAMES)
    locked_ppm: Optional[float] = None
    locked_y0: Optional[float] = None
    lock_spread: Optional[float] = None

    frames = 0
    frames_with_marker = 0
    read_failures = 0
    last_report = 0.0
    last_reason = "waiting for the first frame"
    final_object_mm = None
    final_object_px = None
    object_reason = "the capture loop did not run"
    marker: Optional[MarkerMeasurement] = None
    started = time.time()

    try:
        while True:
            if args.no_window and time.time() - started >= args.seconds:
                break

            frame = camera.read()
            if frame is None:
                read_failures += 1
                last_reason = "the camera returned no frame (dropped read)"
                if not args.no_window and (cv2.waitKey(1) & 0xFF) in _QUIT_KEYS:
                    break
                continue

            frames += 1
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # --- steps 2 & 3: detect and confirm the marker ---
            corners, ids, _rejected = detect(gray)
            marker = None
            if ids is not None and len(ids) > 0:
                frames_with_marker += 1
                measured = [MarkerMeasurement(ids[i][0], corners[i], args.marker_mm)
                            for i in range(len(ids))]
                # Largest marker in view = the closest / most accurate one.
                marker = max(measured, key=lambda m: m.mean_side_px)

            # --- steps 4 & 5: pixels_per_mm and the baseline ---
            if marker is None:
                ppm_window.clear()
                last_reason = "no DICT_5X5_50 marker is visible in the frame"
            elif not marker.reliable:
                ppm_window.clear()
                last_reason = (
                    f"marker {marker.id} is at too steep an angle "
                    f"(side lengths differ by {marker.imbalance * 100:.0f} %, "
                    f"limit {MAX_SIDE_IMBALANCE * 100:.0f} %) -- "
                    "hold it flat and square to the camera"
                )
            else:
                ppm_window.append((marker.pixels_per_mm, marker.baseline_y))
                if locked_ppm is None and len(ppm_window) == LOCK_FRAMES:
                    values = [v for v, _ in ppm_window]
                    spread = (max(values) - min(values)) / statistics.mean(values)
                    if spread <= LOCK_TOLERANCE:
                        locked_ppm = statistics.mean(values)
                        locked_y0 = statistics.mean([y for _, y in ppm_window])
                        lock_spread = spread
                        print("--- CALIBRATION LOCKED ---")
                        print(f"  marker id            : {marker.id}")
                        print(f"  marker side          : {marker.mean_side_px:.2f} px "
                              f"= {args.marker_mm} mm")
                        print(f"  pixels_per_mm        : {locked_ppm:.4f}")
                        print(f"  1 mm                 : {locked_ppm:.2f} px")
                        print(f"  baseline y_0         : {locked_y0:.1f} px "
                              "(marker bottom edge = build surface)")
                        print(f"  stability            : {spread * 100:.2f} % spread "
                              f"over {LOCK_FRAMES} frames")
                        print()
                    else:
                        last_reason = (
                            f"measurement still unstable ({spread * 100:.1f} % spread "
                            f"over {LOCK_FRAMES} frames, needs "
                            f"<= {LOCK_TOLERANCE * 100:.0f} %) -- hold the marker steady"
                        )

            # --- steps 7 & 8: the object, in millimetres ---
            object_bbox = None
            object_mm = None
            object_reason = ""
            search_gray = mask_out_marker(gray, marker) if marker is not None else gray
            seg = segment_part(search_gray)
            if seg.ok:
                object_bbox = seg.bbox
                x, y, w, h = seg.bbox
                final_object_px = (w, h)
                if locked_ppm:
                    above = (locked_y0 - (y + h)) / locked_ppm if locked_y0 is not None else None
                    object_mm = (w / locked_ppm, h / locked_ppm, above)
                    final_object_mm = object_mm
                else:
                    object_reason = ("object found, but pixels_per_mm is not established "
                                     "yet -- no mm value can be given")
            else:
                object_reason = seg.reason

            # --- step 6: show it ---
            if locked_ppm is not None:
                status = "CALIBRATED"
            elif marker is not None:
                status = "MARKER SEEN - calibrating..."
            else:
                status = "NO MARKER"

            if not args.no_window:
                annotated = draw_overlay(frame.copy(), marker, locked_ppm, locked_y0,
                                         object_bbox, object_mm, status)
                cv2.imshow("kelvin6k calibration test", annotated)
                key = cv2.waitKey(1) & 0xFF
                if key in _QUIT_KEYS:
                    break
                if key in (ord("b"), ord("B")):
                    locked_ppm = locked_y0 = lock_spread = None
                    ppm_window.clear()
                    print("Baseline released -- re-calibrating.\n")
                if key in (ord("s"), ord("S")):
                    name = f"calibration_{int(time.time())}.png"
                    cv2.imwrite(name, annotated)
                    print(f"Saved {name}\n")

            # --- step 9: terminal reporting, once a second ---
            now = time.time()
            if now - last_report >= 1.0:
                last_report = now
                if locked_ppm is not None and object_mm is not None:
                    w_mm, h_mm, above = object_mm
                    x, y, w, h = object_bbox
                    above_text = f"{above:.1f} mm" if above is not None else "unavailable"
                    print(f"[{now - started:5.1f}s] OBJECT  "
                          f"{w_mm:7.2f} mm wide x {h_mm:7.2f} mm tall  "
                          f"({w} x {h} px)   height above y_0: {above_text}")
                elif locked_ppm is not None:
                    print(f"[{now - started:5.1f}s] CALIBRATED, no object measured: "
                          f"{object_reason}")
                else:
                    print(f"[{now - started:5.1f}s] NOT CALIBRATED: {last_reason}")

    except KeyboardInterrupt:
        print("\nInterrupted.")
    finally:
        camera.release()
        if not args.no_window:
            cv2.destroyAllWindows()

    # --- final summary: each step, pass or the exact reason it did not ---
    elapsed = max(1e-6, time.time() - started)
    print("\n" + "=" * 58)
    print("RESULTS")
    print("=" * 58)

    ok = True

    print(f"1. Camera opened            : OK  (index {camera.index}, {width}x{height})")
    print(f"   frames received          : {frames} in {elapsed:.1f}s "
          f"({frames / elapsed:.1f} fps), {read_failures} dropped reads")
    if frames == 0:
        print("   -> FAILED: no frames arrived from the camera.")
        ok = False

    if frames_with_marker > 0:
        print(f"2. Marker detected          : OK  (seen in {frames_with_marker} "
              f"of {frames} frames)")
        print(f"3. Detection confirmed      : OK  ({args.dict}, marker id {marker.id})"
              if marker is not None else
              "3. Detection confirmed      : OK")
    else:
        print(f"2. Marker detected          : FAILED")
        print(f"   -> No {args.dict} marker was found in any of {frames} frames.")
        print("      Check that: the marker is printed from the DICT_5X5_50 set,")
        print("      it is fully inside the view, in focus, and evenly lit.")
        print("3. Detection confirmed      : not reached")
        ok = False

    if locked_ppm is not None:
        print(f"4. pixels_per_mm            : {locked_ppm:.4f} px/mm "
              f"(from a {args.marker_mm} mm marker)")
        print(f"5. Baseline y_0             : {locked_y0:.1f} px "
              f"(+/- {lock_spread * 100:.2f} % over {LOCK_FRAMES} frames)")
    else:
        print("4. pixels_per_mm            : unavailable")
        print(f"   -> {last_reason}")
        print("5. Baseline y_0             : unavailable (needs calibration first)")
        ok = False

    print(f"6. Annotated frame shown    : "
          + ("OK" if not args.no_window else "skipped (--no-window)"))

    if final_object_px is not None:
        print(f"7. Object detected          : OK  "
              f"({final_object_px[0]} x {final_object_px[1]} px)")
    else:
        print("7. Object detected          : FAILED")
        print(f"   -> {object_reason or 'no object was segmented'}")
        ok = False

    if final_object_mm is not None:
        w_mm, h_mm, above = final_object_mm
        print(f"8. Measured width           : {w_mm:.2f} mm")
        print(f"   Measured height          : {h_mm:.2f} mm")
        if above is not None:
            print(f"   Height above baseline    : {above:.2f} mm")
    else:
        print("8. Measured width / height  : unavailable")
        print("   -> " + (object_reason or
                          "calibration or object detection did not complete; "
                          "no millimetre value is reported"))
        ok = False

    print("=" * 58)
    print(f"RESULT: {'SUCCESS' if ok else 'INCOMPLETE -- see the reasons above'}")

    if locked_ppm is not None:
        print(f"\nTo use this calibration elsewhere in the project, put it in .env:")
        print(f"    PIXELS_PER_MM={locked_ppm:.4f}")
        print("(valid only for this camera at this distance and zoom -- "
              "re-run the test if either changes.)")

    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
