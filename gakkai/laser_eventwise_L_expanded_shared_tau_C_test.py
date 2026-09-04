"""Run the event-wise prompt+slow test with every listed x=5.10 L scan.

This is an expanded-L configuration of
``laser_eventwise_L_shared_tau_C_test.py``.  It keeps the same event-level
absolute-IQ waveform definition, fit model, quality metrics, and C comparison,
while increasing the L training/validation positions from 6 to 16.
"""

from __future__ import annotations

import sys

import laser_eventwise_L_shared_tau_C_test as analysis


# x = 5.10 mm (L-center), transcribed from the position-scan run table.
analysis.L_RUNS = {
    # 6.90: "135121",
    6.85: "135036",
    6.80: "134954",
    6.75: "134913",
    6.70: "134832",
    6.60: "134748",
    6.50: "134710",
    6.40: "134626",
    6.30: "134447",
    6.20: "134407",
    6.10: "134328",
    6.05: "134248",
    6.00: "134208",
    5.95: "134123",
    5.90: "134041",
    # 5.70: "133955",
}

# Keep the corrected C comparison set from the base analysis.
analysis.C_RUNS = {
    6.80: "130955",
    6.70: "131115",
    6.30: "131249",
    6.00: "131518",
}


def main():
    # Preserve explicit command-line choices.  Only supply a different default
    # prefix so this expanded run cannot overwrite the original six-L result.
    if "--output-prefix" not in sys.argv:
        sys.argv.extend([
            "--output-prefix",
            "laser_eventwise_L_expanded_shared_tau_C_test",
        ])
    analysis.main()


if __name__ == "__main__":
    main()
