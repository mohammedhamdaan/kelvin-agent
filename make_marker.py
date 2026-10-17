"""
Generates a printable DICT_5X5_50 ArUco marker at an exact physical size.

The calibration test converts pixels to millimetres using the marker's REAL
printed side length, so the print must come out at exactly the size asked for.
The PNG is written at 300 DPI with the DPI recorded in the file, so printing at
"100% / actual size" (NOT "fit to page") gives the correct dimensions.

Usage:
    python make_marker.py                      # id 0, 30 mm, -> marker_id0_30mm.png
    python make_marker.py --id 7 --mm 40
    python make_marker.py --dpi 600

After printing, MEASURE the black square with a ruler. If it is not the size
you asked for, pass the measured value to the test instead of assuming:
    python calibration_test.py --marker-mm <the measured size>
"""

import argparse
import sys

import cv2
import numpy as np
from PIL import Image

# White margin around the marker, as a fraction of the marker's side. ArUco
# needs a quiet zone to find the outer border; printed edge-to-edge it is
# unreliable.
QUIET_ZONE = 0.25


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate a printable ArUco marker")
    parser.add_argument("--id", type=int, default=0, help="marker id, 0-49 (default 0)")
    parser.add_argument("--mm", type=float, default=30.0,
                        help="printed side length of the black square in mm (default 30)")
    parser.add_argument("--dpi", type=int, default=300, help="print resolution (default 300)")
    parser.add_argument("--dict", default="DICT_5X5_50", help="ArUco dictionary name")
    parser.add_argument("--out", default=None, help="output PNG path")
    args = parser.parse_args()

    if not 0 <= args.id <= 49:
        print("RESULT: FAILURE\nDICT_5X5_50 holds ids 0-49 only.", file=sys.stderr)
        return 1
    if args.mm <= 0 or args.dpi <= 0:
        print("RESULT: FAILURE\n--mm and --dpi must be positive.", file=sys.stderr)
        return 1
    if not hasattr(cv2, "aruco") or not hasattr(cv2.aruco, args.dict):
        print(f"RESULT: FAILURE\nThis OpenCV build has no cv2.aruco.{args.dict}.",
              file=sys.stderr)
        return 1

    dictionary = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, args.dict))
    side_px = int(round(args.mm / 25.4 * args.dpi))

    if hasattr(cv2.aruco, "generateImageMarker"):  # OpenCV >= 4.7
        marker = cv2.aruco.generateImageMarker(dictionary, args.id, side_px)
    else:
        marker = cv2.aruco.drawMarker(dictionary, args.id, side_px)

    pad = int(round(side_px * QUIET_ZONE))
    canvas = np.full((side_px + 2 * pad, side_px + 2 * pad), 255, dtype=np.uint8)
    canvas[pad:pad + side_px, pad:pad + side_px] = marker

    out = args.out or f"marker_id{args.id}_{args.mm:g}mm.png"
    Image.fromarray(canvas).save(out, dpi=(args.dpi, args.dpi))

    print("=== ARUCO MARKER GENERATED ===")
    print(f"File          : {out}")
    print(f"Dictionary    : {args.dict}")
    print(f"Marker id     : {args.id}")
    print(f"Printed size  : {args.mm} mm per side (black square only)")
    print(f"Image         : {canvas.shape[1]}x{canvas.shape[0]} px at {args.dpi} DPI")
    print()
    print("PRINT IT LIKE THIS:")
    print("  1. Open the PNG and print at 100% / 'Actual size'.")
    print("     Do NOT use 'Fit to page' -- that changes the size and every")
    print("     millimetre measurement with it.")
    print("  2. Measure the black square with a ruler after printing.")
    print(f"  3. If it is not {args.mm} mm, run the test with what you measured:")
    print(f"        python calibration_test.py --marker-mm <measured mm>")
    print("  4. Glue or tape it to something flat and rigid. A curled or")
    print("     bent marker measures short and the calibration inherits that error.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
