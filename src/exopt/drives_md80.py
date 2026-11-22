"""The two MD80 hip drives over CAN: torque commands, reads, fault recovery, error registers."""

import sys

import numpy as np
from jaxtyping import Float

MAX_TORQUE = 13.0  # [Nm] hardware torque limit per motor
STALL_LIMIT = 0.1  # [s] of identical reads that mean the CAN link dropped (the drive watchdog trips at 0.2 s)
RECOVERY_RETRY = 1.0  # [s] between attempts to re-enable faulted drives

QUICK_STATUS_BITS = {
    0: "main encoder error", 1: "output encoder error", 2: "calibration encoder error", 3: "MOSFET bridge error",
    4: "hardware error", 5: "communication error", 6: "motion error", 7: "misc error", 8: "configuration error",
    14: "drive disabled",
}
DRIVE_DISABLED = 1 << 14
TARGET_REACHED = 1 << 15  # toggles with the command, not a fault
ENCODER_BITS = {0: "communication", 1: "wrong direction", 2: "empty LUT", 3: "faulty LUT", 4: "calibration failed",
                5: "position invalid", 6: "init", 30: "low accuracy (warning)"}
ERROR_REGISTERS = {
    "hardwareErrors": {0: "over current", 1: "over voltage", 2: "under voltage", 3: "motor over temperature",
                       4: "MOSFET over temperature", 5: "ADC current offsets"},
    "bridgeErrors": {0: "bridge communication", 1: "bridge over current", 2: "bridge general fault"},
    "communicationErrors": {30: "CAN watchdog tripped (warning)"},
    "motionErrors": {0: "position outside limits", 1: "velocity outside limits", 24: "acceleration clipped",
                     25: "torque clipped", 26: "velocity clipped", 27: "position clipped"},
    "mainEncoderErrors": ENCODER_BITS,
    "outputEncoderErrors": ENCODER_BITS,
    "calibrationErrors": {0: "offset", 1: "resistance", 2: "inductance", 3: "pole pairs", 4: "setup"},
}


def set_bits(value: int, names: dict[int, str]) -> str:
    return ", ".join(names.get(bit, f"bit {bit}") for bit in range(32) if value >> bit & 1) or "none"


class Drives:
    """The two MD80 hip drives over CAN in raw torque mode; as a context, left at zero torque and disabled."""

    def __init__(self):
        import pyCandle  # only on the Pi; drives_mock swaps in simulated legs

        self.api = pyCandle
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
        self.statuses: list[int | None] = [None, None]
        self.reads: Float[np.ndarray, "3 2"] | None = None
        self.read_at, self.fresh_at, self.retry_at = 0.0, 0.0, 0.0
        self.faulted = False  # torque must stay off, until the drives are re-enabled
        self.print_registers("at start")
        self.candle.begin()

    def __enter__(self) -> "Drives":
        return self

    def __exit__(self, *_) -> None:
        self.apply(np.zeros(2))
        self.candle.end()
        self.print_registers("at stop")
        for id in self.ids:
            self.candle.controlMd80Enable(id, False)

    @property
    def stalled(self) -> bool:
        """The reads froze: the CAN link dropped."""
        return self.read_at - self.fresh_at > STALL_LIMIT

    @property
    def disabled(self) -> bool:
        return any(status is not None and status & DRIVE_DISABLED for status in self.statuses)

    def read(self, t: float) -> Float[np.ndarray, "3 2"]:
        """Position [rad], velocity [rad/s] and measured torque [Nm] per leg; prints a quick status that changed."""
        for leg, (id, md) in enumerate(zip(self.ids, self.candle.md80s)):
            if not hasattr(md, "getQuickStatus"):
                continue
            status = md.getQuickStatus() & ~TARGET_REACHED
            if status != self.statuses[leg]:
                print(f"t={t:6.1f} s drive {id} says: quick status {status:#06x} ({set_bits(status, QUICK_STATUS_BITS)})")
                self.statuses[leg] = status
        reads = np.array([[getattr(md, get)() for md in self.candle.md80s]
                          for get in ("getPosition", "getVelocity", "getTorque")])
        if self.reads is None or not np.array_equal(reads, self.reads):
            self.fresh_at = t
        self.reads, self.read_at = reads, t
        return reads

    def apply(self, torque: Float[np.ndarray, "2"]) -> None:
        for md, leg_torque in zip(self.candle.md80s, torque):
            md.setTargetTorque(float(leg_torque))

    def watch(self) -> None:
        """Fault on frozen reads or a disabled drive; once reads flow again re-enable, every RECOVERY_RETRY."""
        t = self.read_at
        if not self.faulted and (self.stalled or self.disabled):
            self.faulted = True
            print(f"t={t:6.1f} s drive fault ({'reads frozen' if self.stalled else 'drive disabled'}): torque off")
        if self.faulted and not self.stalled and t >= self.retry_at:
            self.faulted = not self.reenable()
            self.statuses, self.retry_at = [None, None], t + RECOVERY_RETRY
            print(f"t={t:6.1f} s drives {'still faulted' if self.faulted else 're-enabled'}")

    def reenable(self) -> bool:
        """Clear each drive's errors and re-enable it at zero torque; True if every drive came back."""
        self.apply(np.zeros(2))
        self.candle.end()
        clear_errors = getattr(getattr(self.api, "Md80Reg_E", None), "runClearErrors", None)
        every_drive_back = True
        for id in self.ids:
            try:
                if clear_errors is not None:
                    self.candle.writeMd80Register(id, clear_errors, 1)
                self.candle.controlMd80Enable(id, False)
                every_drive_back &= bool(self.candle.controlMd80Mode(id, self.api.RAW_TORQUE)
                                         and self.candle.controlMd80Enable(id, True))
            except Exception as error:  # a still-silent drive or a pyCandle without register writes
                print(f"drive {id} re-enable failed: {error}")
                every_drive_back = False
        self.candle.begin()
        return every_drive_back

    def print_registers(self, when: str) -> None:
        """Each drive's error registers and watchdog timeout, read over CAN (the candle must be in config mode)."""
        for id in self.ids:
            try:
                read = lambda name: self.candle.readMd80Register(id, getattr(self.api.Md80Reg_E, name), 0)
                errors = {name: read(name) for name in ERROR_REGISTERS}
                watchdog = read("canWatchdog")
            except Exception as error:  # a silent drive or a pyCandle without register reads
                print(f"drive {id} registers {when}: unreadable ({error})")
                continue
            found = "; ".join(f"{name} {value:#x}: {set_bits(value, ERROR_REGISTERS[name])}"
                              for name, value in errors.items() if value) or "no errors"
            print(f"drive {id} registers {when}: {found} (CAN watchdog {watchdog} ms)")
