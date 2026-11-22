"""The two MD80 hip drives over CAN: torque commands and reads."""

import sys

import numpy as np
from jaxtyping import Float

MAX_TORQUE = 13.0  # [Nm] hardware torque limit per motor


class Drives:
    """The two MD80 hip drives over CAN in raw torque mode; as a context, left at zero torque and disabled."""

    def __init__(self):
        import pyCandle  # only on the Pi; drives_mock swaps in simulated legs

        self.candle = pyCandle.Candle(pyCandle.CAN_BAUD_1M, True)
        self.ids = self.candle.ping()[:2]
        if len(self.ids) < 2:
            sys.exit(f"expected 2 drives, found {self.ids}: check the CAN cable, the drives' power and the CANdle USB")
        for id in self.ids:
            self.candle.addMd80(id)
        for id, md in zip(self.ids, self.candle.md80s):
            if not (self.candle.controlMd80Mode(id, pyCandle.RAW_TORQUE) and self.candle.controlMd80Enable(id, True)):
                sys.exit(f"drive {id} refused raw torque mode or enabling")
            md.setMaxTorque(MAX_TORQUE)
        self.candle.begin()

    def __enter__(self) -> "Drives":
        return self

    def __exit__(self, *_) -> None:
        self.apply(np.zeros(2))
        self.candle.end()
        for id in self.ids:
            self.candle.controlMd80Enable(id, False)

    def read(self, t: float) -> Float[np.ndarray, "3 2"]:
        """Position [rad], velocity [rad/s] and measured torque [Nm] per leg."""
        return np.array([[getattr(md, get)() for md in self.candle.md80s]
                         for get in ("getPosition", "getVelocity", "getTorque")])

    def print_status(self) -> None:
        """Each drive's quick status, where this pyCandle has it."""
        for id, md in zip(self.ids, self.candle.md80s):
            if (status := getattr(md, "getQuickStatus", None)) is not None:
                print(f"  drive {id}: quick status {status():#06x}")

    def apply(self, torque: Float[np.ndarray, "2"]) -> None:
        for md, leg_torque in zip(self.candle.md80s, torque):
            md.setTargetTorque(float(leg_torque))
