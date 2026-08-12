import pyCandle
from time import sleep, time
import math
import glob
import json
import os
import re
import sys
import numpy as np

args = [a for a in sys.argv[1:] if not a.startswith("--")]
EXP_DIR = args[0] if args else os.path.dirname(os.path.abspath(__file__))
RERUN = "--rerun" in sys.argv

config_numbers = [
    int(re.search(r"config_(\d+)\.json$", p).group(1))
    for p in glob.glob(os.path.join(EXP_DIR, "config_*.json"))
]
if not config_numbers:
    sys.exit(f"No config_*.json in {EXP_DIR}")


def run_file(i):
    return os.path.join(EXP_DIR, f"run_{i}.txt")


pending = sorted(i for i in config_numbers if not os.path.exists(run_file(i)))

if RERUN:
    RUN_NUMBER = max(config_numbers)
    answer = input(f"Re-run config {RUN_NUMBER}, appending to its run file? [y/N] ")
    if answer.strip().lower() != "y":
        sys.exit("EXIT: nothing written")
elif not pending:
    sys.exit(
        f"Every config in {EXP_DIR} has been run already. "
        "Propose the next one with bo_step.py, or pass --rerun to repeat the newest."
    )
else:
    RUN_NUMBER = pending[0]
    if len(pending) > 1:
        print(f"{len(pending)} configs still to run: {pending}")

run_path = run_file(RUN_NUMBER)

print(f"Experiment folder: {EXP_DIR}")
print(f"Running config {RUN_NUMBER}")

TORQUE_SCALE = 5.0

candle = pyCandle.Candle(pyCandle.CAN_BAUD_1M, True)

ids = candle.ping()

if len(ids) == 0:
    sys.exit("EXIT FALIURE")

for id in ids:
    candle.addMd80(id)

candle.controlMd80SetEncoderZero(ids[0])
candle.controlMd80Mode(ids[0], pyCandle.RAW_TORQUE)
candle.controlMd80Enable(ids[0], True)

candle.controlMd80SetEncoderZero(ids[1])
candle.controlMd80Mode(ids[1], pyCandle.RAW_TORQUE)
candle.controlMd80Enable(ids[1], True)

candle.md80s[0].setMaxTorque(13)
candle.md80s[1].setMaxTorque(13)


times = []

end_time = 60
dt = 0.01

with open(os.path.join(EXP_DIR, f'config_{RUN_NUMBER}.json')) as f:
    cfg = json.load(f)

MODE = cfg.get("mode")

if MODE == "vector":
    AMPLITUDE = cfg["amplitude"] * TORQUE_SCALE
    PHASE = cfg["phase"] * 2 * math.pi
    print(f"sine profile: amplitude {AMPLITUDE:.4f} Nm, phase {PHASE:.4f} rad")

    def torque(u):
        return AMPLITUDE * math.sin(u - PHASE)

elif MODE == "functional":
    rho = cfg["rho"][0]
    basis = [p[0] for p in cfg["x"]]
    coeffs = cfg["a"]
    print(f"rkhs profile: rho {rho}, basis {basis}, coeffs {coeffs}")

    def profile(v):
        return sum(a * math.exp(-0.5 * ((v - x) / rho) ** 2)
                   for x, a in zip(basis, coeffs))

    PEAK = max(abs(profile(j / 20000.0)) for j in range(20001))
    if PEAK < 1e-9:
        sys.exit(f"config {RUN_NUMBER} is flat, nothing to apply")
    print(f"peak |profile| {PEAK:.4f} -> scaled to +-{TORQUE_SCALE} Nm")

    def torque(u):
        return TORQUE_SCALE * profile((u / (2 * math.pi)) % 1.0) / PEAK

else:
    sys.exit(f"config {RUN_NUMBER} has mode {MODE!r}; expected 'vector' or 'functional'")


VELOCITY_SCALE = dt

POSITION_OFFSET = [0.0, 0.0]


def gait_phase(position, velocity, leg):
    return math.atan2(velocity * VELOCITY_SCALE, position - POSITION_OFFSET[leg])


alpha = 0.9
gain_ramp = 1
pos_prv = [0.0, 0.0]
vel_prv = [0.0, 0.0]

num_steps = int(end_time/dt)
positions = np.zeros((num_steps, 2))
velocities = np.zeros((num_steps, 2))
gait_phases = np.zeros((num_steps, 2))
torques  = np.zeros((num_steps, 2))
actual_torques  = np.zeros((num_steps, 2))
mechanicalPower = np.zeros((num_steps, 2))
off_status = 0
candle.begin()
input("Press enter/any key to start")
file1=open(run_path, "a+")
file1.write("time position_0 position_1 velocity_0 velocity_1 gait_phase_0 gait_phase_1 torque_0 torque_1 actual_torque_0 actual_torque_1 mechanicalPower_0 mechanicalPower_1\n")


start_time = time()
print('Test started')
for i in range(num_steps):
    try:
        curr_time = time() - start_time
        loop_start = time()

        if curr_time > (end_time - 10):
            if off_status == 0:
                print('Exo will turn off in 10 seconds')
                off_status = 1
        if curr_time > end_time:
            print('Stop the treadmill')
            candle.md80s[0].setTargetTorque(0)
            candle.md80s[1].setTargetTorque(0)
            sleep(15)
            file1.flush()
            file1.close()
            print('File saved')
            break


        times.append(curr_time)


        gain = min(curr_time/5, 1.0)
        if gain >= 1.0 and gain_ramp == 1:
            print('Gain ramped up')
            gain_ramp = 0


        positions[i] = [candle.md80s[0].getPosition(), candle.md80s[1].getPosition()]
        velocities[i] = [candle.md80s[0].getVelocity(), candle.md80s[1].getVelocity()]

        pos_prv = [alpha*p + (1-alpha)*positions[i,j] for j, p in enumerate(pos_prv)]
        vel_prv = [alpha*v + (1-alpha)*velocities[i,j] for j, v in enumerate(vel_prv)]

        gait_phases[i] = [gait_phase(pos_prv[j], vel_prv[j], j) for j in (0, 1)]
        torque_0 = gain*torque(gait_phases[i, 0])
        torque_1 = gain*torque(gait_phases[i, 1])


        torques[i] = [torque_0, torque_1]


        candle.md80s[0].setTargetTorque(float(torque_0))
        candle.md80s[1].setTargetTorque(float(torque_1))

        actual_torques[i] = [candle.md80s[0].getTorque(), candle.md80s[1].getTorque()]

        mechanicalPower[i,0] = actual_torques[i,0]*velocities[i,0]
        mechanicalPower[i,1] = actual_torques[i,1]*velocities[i,1]


        file1.write(str(times[i])+" "+str(positions[i, 0])+" "+str(positions[i, 1])+" "+str(velocities[i, 0])+" "+str(velocities[i, 1])+" "+str(gait_phases[i, 0])+" "+str(gait_phases[i, 1])+" "+str(torques[i, 0])+" "+str(torques[i, 1])+ " "+str(actual_torques[i, 0])+" "+str(actual_torques[i, 1])+" "+str(mechanicalPower[i, 0])+" "+str(mechanicalPower[i, 1])+"\n" )

        loop_end = time()

        if dt - (loop_end - loop_start) < 0:
            print((loop_end - loop_start))
        else:
            sleep(dt - (loop_end - loop_start))
    except KeyboardInterrupt:
        candle.md80s[0].setTargetTorque(0.0)
        candle.md80s[1].setTargetTorque(0.0)
        sleep(60)
        candle.end()
        sys.exit("EXIT SUCCESS")

        file1.flush()
        file1.close()
        break
candle.end()
sys.exit("EXIT SUCCESS")
