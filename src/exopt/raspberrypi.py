"""The Pi over ssh, in-process (no ssh client or setup, Windows too): routes, forwarded ports, its copy of exopt."""

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import io
import json
import os
import re
import shlex
import signal
import socket
import socketserver
import subprocess
import sys
import tarfile
import threading

import paramiko

USER = "pi"
CHECKOUT = "exo_files/exopt"  # the Pi's copy of exopt, from its home folder
# ways to the Pi, all tried at once: the first here that answers is used (scripts/pi-setup.sh sets them up)
ROUTES = {"wifi": "exo-pi.local", "hotspot": "10.42.0.1", "usb": "10.55.0.1"}
HOTSPOT = "exo-hotspot"  # the Pi's NetworkManager profile of its own wifi network
SETTINGS_FILE = Path.home() / ".config" / "exopt" / "pi.json"
KEPT = 20000  # characters of a command's output kept
REPO = Path(__file__).resolve().parents[2]
COPIED = ("src", "pyproject.toml", "uv.lock")  # what the Pi runs: this computer's files as they are, committed or not


def load_settings() -> dict:
    """This computer's {wifi: the Pi's address on the lab wifi, password: its ssh password}."""
    try:
        saved = json.loads(SETTINGS_FILE.read_text())
    except (OSError, ValueError):
        saved = {}
    return dict(wifi=ROUTES["wifi"], password="") | (saved if isinstance(saved, dict) else {})


def save_settings(wifi: str, password: str) -> None:
    SETTINGS_FILE.parent.mkdir(parents=True, exist_ok=True)
    SETTINGS_FILE.write_text(json.dumps(dict(wifi=wifi, password=password), indent=1))
    SETTINGS_FILE.chmod(0o600)


def answering(routes: dict[str, str], port: int = 22) -> str | None:
    """The first route whose ssh port answers."""

    def answers(address: str) -> bool:
        try:
            socket.create_connection((address, port), timeout=3).close()
            return True
        except OSError:
            return False

    with ThreadPoolExecutor(len(routes)) as pool:
        answered = list(pool.map(answers, routes.values()))
    return next((route for route, up in zip(routes, answered) if up), None)


class Tunnel:
    """One ssh connection to the Pi over the route that answers, forwarding localhost ports to the same ports there."""

    def __init__(self, routes: dict[str, str], password: str, ports: tuple[int, ...]):
        self.target = "the Pi"
        self.route: str | None = None
        self.error: str | None = None
        self.connected = False
        self.closed = False
        self.command: RemoteCommand | None = None
        self.client = paramiko.SSHClient()
        self.servers: list[socketserver.ThreadingTCPServer] = []
        threading.Thread(target=self.run, args=(routes, password, ports), daemon=True).start()

    def run(self, routes: dict[str, str], password: str, ports: tuple[int, ...]) -> None:
        try:
            self.route = answering(routes)
            if self.route is None:
                tried = ", ".join(f"{route} ({address})" for route, address in routes.items())
                raise OSError(f"No answer over {tried}. Join the lab wifi or the Pi's exo-pi wifi, or plug in the "
                              "USB-C cable.")
            self.target = f"{self.route} · {USER}@{routes[self.route]}"
            self.client.load_system_host_keys()
            self.client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            # a blank password falls back to the ssh agent and the default keys
            self.client.connect(routes[self.route], 22, USER, password=password or None, timeout=5, banner_timeout=10,
                                auth_timeout=10, look_for_keys=not password, allow_agent=not password)
            transport = self.client.get_transport()
            assert transport is not None
            transport.set_keepalive(5)
            for port in ports:
                self.servers.append(forward_server(transport, port))
            self.connected = True
            if self.closed:  # disconnected while connecting
                self.close()
        except paramiko.AuthenticationException:
            self.error = f"{USER}'s password was refused: set it under Settings."
            self.close()
        except Exception as error:
            self.error = str(error) or type(error).__name__
            self.close()

    def alive(self) -> bool:
        """Still connecting, or connected with the ssh session up."""
        if self.closed:
            return False
        transport = self.client.get_transport()
        return not self.connected or (transport is not None and transport.is_active())

    def start(self, script: str) -> "RemoteCommand":
        """Copy this computer's exopt to the Pi, then run one of its scripts there (no git, no internet needed)."""
        try:
            self.send_code()
            warning = ""
        except Exception as error:
            warning = f"echo {shlex.quote(f'warning: copying exopt failed ({error}): running the copy on the Pi')}; "
        transport = self.client.get_transport()
        assert transport is not None
        # no project build: it needs hatchling from PyPI, and the Pi may be offline
        self.command = RemoteCommand(transport, (
            f'{warning}export PATH="$HOME/.local/bin:$PATH"; cd {CHECKOUT} && '
            "uv sync -q --frozen --inexact --no-install-project --group controller && "
            f"PYTHONPATH=src exec uv run --no-sync python {shlex.quote(script)}"))
        return self.command

    def send_code(self) -> None:
        """Replace the Pi's COPIED files with this computer's, as one tar stream over ssh."""
        archive = io.BytesIO()
        with tarfile.open(fileobj=archive, mode="w:gz") as tar:
            for name in COPIED:
                tar.add(REPO / name, name, filter=lambda file: None if "__pycache__" in file.name else file)
        stdin, stdout, stderr = self.client.exec_command(
            f"mkdir -p {CHECKOUT} && cd {CHECKOUT} && rm -rf src && tar -xzf -", timeout=60)
        stdin.write(archive.getvalue())
        stdin.channel.shutdown_write()
        if stdout.channel.recv_exit_status():
            raise OSError(stderr.read().decode().strip() or "tar failed")

    def execute(self, command: str, timeout: float = 30) -> str:
        """Run a short shell command on the Pi and wait for its stdout."""
        _, stdout, _ = self.client.exec_command(command, timeout=timeout)
        return stdout.read().decode()

    def hotspot_on(self) -> bool:
        return HOTSPOT in self.execute("nmcli -t -f NAME con show --active").splitlines()

    def switch_wifi(self, hotspot: bool) -> None:
        """Turn the Pi's hotspot on, or off for the lab wifi (back on if that isn't joined within 30 s).

        Detached from this session, which the switch drops unless it runs over usb."""
        switch = f"sudo nmcli con up {HOTSPOT}" if hotspot else (
            f"sudo nmcli con down {HOTSPOT}; sleep 30; "
            f"nmcli -t -f TYPE,STATE dev | grep -qx wifi:connected || sudo nmcli con up {HOTSPOT}")
        self.execute(f"nohup sh -c {shlex.quote(switch)} >/dev/null 2>&1 &")

    def close(self) -> None:
        # ctrl-c first, so the controller releases the drives before the session drops
        if self.command is not None and self.command.running():
            self.command.interrupt()
            self.command.wait(5)
        self.closed, self.connected = True, False
        for server in self.servers:
            server.shutdown()
            server.server_close()
        self.servers = []
        self.client.close()


class RemoteCommand:
    """A command in a pty on the Pi, so ctrl-c interrupts it as in a terminal."""

    def __init__(self, transport: paramiko.Transport, command: str):
        self.output = ""
        self.channel = transport.open_session()
        self.channel.get_pty(width=160)
        self.channel.exec_command(command)
        threading.Thread(target=self.read, daemon=True).start()

    def read(self) -> None:
        try:
            while data := self.channel.recv(4096):
                text = re.sub(r"\x1b\[[0-9;?]*[A-Za-z]|\r", "", data.decode(errors="replace"))
                self.output = (self.output + text)[-KEPT:]
        except (OSError, EOFError, paramiko.SSHException):
            pass

    def running(self) -> bool:
        return not self.channel.closed and not self.channel.exit_status_ready()

    def exit_status(self) -> int | None:
        return self.channel.recv_exit_status() if self.channel.exit_status_ready() else None

    def interrupt(self) -> None:
        self.channel.send(b"\x03")

    def wait(self, timeout: float) -> None:
        self.channel.status_event.wait(timeout)


class Local:
    """This computer standing in for the Pi's Tunnel: a controller script (simulated legs) runs here, nothing forwarded."""

    target, route, error, connected = "this computer", "local", None, True

    def __init__(self):
        self.closed = False
        self.command: LocalCommand | None = None

    def alive(self) -> bool:
        return not self.closed

    def start(self, script: str) -> "LocalCommand":
        self.command = LocalCommand(str(REPO / script))
        return self.command

    def close(self) -> None:
        if self.command is not None and self.command.running():
            self.command.interrupt()
            self.command.wait(5)
            self.command.process.terminate()
        self.closed = True


class LocalCommand:
    """A python script as a process on this computer, with RemoteCommand's interface."""

    def __init__(self, script: str):
        self.output = ""
        # its own process group on Windows, so a ctrl-break reaches it alone
        group = dict(creationflags=subprocess.CREATE_NEW_PROCESS_GROUP) if os.name == "nt" else {}
        self.process = subprocess.Popen([sys.executable, "-u", script], stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                        text=True, errors="replace", **group)
        threading.Thread(target=self.read, daemon=True).start()

    def read(self) -> None:
        assert self.process.stdout is not None
        for line in self.process.stdout:
            self.output = (self.output + line)[-KEPT:]

    def running(self) -> bool:
        return self.process.poll() is None

    def exit_status(self) -> int | None:
        return self.process.poll()

    def interrupt(self) -> None:
        self.process.send_signal(signal.CTRL_BREAK_EVENT if os.name == "nt" else signal.SIGINT)

    def wait(self, timeout: float) -> None:
        try:
            self.process.wait(timeout)
        except subprocess.TimeoutExpired:
            pass


def forward_server(transport: paramiko.Transport, port: int) -> socketserver.ThreadingTCPServer:
    """Listen on localhost:port and pipe each connection to localhost:port on the Pi."""

    class Forward(socketserver.BaseRequestHandler):
        def handle(self):
            try:
                channel = transport.open_channel("direct-tcpip", ("localhost", port), self.request.getpeername())
            except (OSError, paramiko.SSHException):
                return  # nothing listening on the Pi yet: the client retries

            # one thread per direction, plain blocking reads, so it works on every OS
            def upstream():
                pipe(self.request.recv, channel.sendall)
                channel.close()

            sender = threading.Thread(target=upstream, daemon=True)
            sender.start()
            pipe(channel.recv, self.request.sendall)
            try:
                self.request.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            sender.join(timeout=1)
            channel.close()

    # rebind right after a disconnect despite TIME_WAIT; on Windows the same flag would allow double binds
    server = socketserver.ThreadingTCPServer(("127.0.0.1", port), Forward, bind_and_activate=False)
    server.allow_reuse_address, server.daemon_threads = os.name != "nt", True
    try:
        server.server_bind()
        server.server_activate()
    except OSError:
        server.server_close()
        raise
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


def pipe(receive, send) -> None:
    try:
        while data := receive(65536):
            send(data)
    except (OSError, EOFError, paramiko.SSHException):
        pass

