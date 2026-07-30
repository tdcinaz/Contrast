from __future__ import annotations

import argparse
import json
import logging
import sys
import traceback
from pathlib import Path

import numpy as np

from deep_denoiser import FFDNetDenoiser
from logging_setup import configure_logging


LOGGER = logging.getLogger("contrast.ngc_worker")


def respond(payload: dict[str, object]) -> None:
    print(json.dumps(payload), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=("ffdnet",), required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--precision", choices=("fp16", "fp32"), default="fp16")
    arguments = parser.parse_args()

    configure_logging("INFO")
    LOGGER.info("Starting NGC worker: model=%s precision=%s", arguments.model, arguments.precision)
    denoiser = FFDNetDenoiser(arguments.weights, arguments.precision)
    input_frames: np.memmap | None = None
    output_frames: np.memmap | None = None
    respond({"status": "ready", "device": denoiser.device_name})

    for line in sys.stdin:
        try:
            request = json.loads(line)
            command = request.get("command")
            if command == "setup":
                shape = tuple(int(value) for value in request["shape"])
                capacity = int(request["capacity"])
                input_frames = np.memmap(request["input"], mode="r+", dtype=np.uint8, shape=(capacity, *shape))
                output_frames = np.memmap(request["output"], mode="r+", dtype=np.uint8, shape=(capacity, *shape))
                LOGGER.info("Configured shared buffers: shape=%s capacity=%s", shape, capacity)
                respond({"status": "ok"})
            elif command == "denoise":
                if input_frames is None or output_frames is None:
                    raise RuntimeError("Shared frame buffers have not been configured.")
                count = int(request["count"])
                results = denoiser.denoise_array(input_frames[:count], float(request["sigma"]))
                output_frames[:count] = results
                LOGGER.debug("Processed NGC batch count=%s", count)
                respond({"status": "ok"})
            elif command == "close":
                LOGGER.info("Closing NGC worker")
                respond({"status": "closed"})
                break
            else:
                raise ValueError(f"Unknown command: {command}")
        except Exception as exc:
            LOGGER.exception("NGC worker request failed")
            respond({"status": "error", "message": str(exc), "traceback": traceback.format_exc()})


if __name__ == "__main__":
    main()