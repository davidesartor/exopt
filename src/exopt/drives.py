"""What the MD80 drives report about themselves: quick status on every frame, error registers in config mode."""

QUICK_STATUS_BITS = {
    0: "main encoder error", 1: "output encoder error", 2: "calibration encoder error", 3: "MOSFET bridge error",
    4: "hardware error", 5: "communication error", 6: "motion error", 7: "misc error", 8: "configuration error",
    14: "drive disabled",
}
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


def print_status_changes(md80s: list, ids: list[int], statuses: list[int | None], t: float) -> None:
    """Print each drive's quick status when it changes (updates statuses in place); a no-op without the getter."""
    for leg, (id, md) in enumerate(zip(ids, md80s)):
        if not hasattr(md, "getQuickStatus"):
            continue
        status = md.getQuickStatus() & ~TARGET_REACHED
        if status != statuses[leg]:
            print(f"t={t:6.1f} s drive {id} says: quick status {status:#06x} ({set_bits(status, QUICK_STATUS_BITS)})")
            statuses[leg] = status


def print_registers(pyCandle, candle, ids: list[int], when: str) -> None:
    """Each drive's error registers and watchdog timeout, read over CAN (the candle must be in config mode)."""
    for id in ids:
        try:
            read = lambda name: candle.readMd80Register(id, getattr(pyCandle.Md80Reg_E, name), 0)
            errors = {name: read(name) for name in ERROR_REGISTERS}
            watchdog = read("canWatchdog")
        except Exception as error:  # a silent drive or a pyCandle without register reads
            print(f"drive {id} registers {when}: unreadable ({error})")
            continue
        found = "; ".join(f"{name} {value:#x}: {set_bits(value, ERROR_REGISTERS[name])}"
                          for name, value in errors.items() if value) or "no errors"
        print(f"drive {id} registers {when}: {found} (CAN watchdog {watchdog} ms)")
