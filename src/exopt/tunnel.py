"""Local port forwarding to the Pi over ssh, in-process, so it needs no ssh client or setup (Windows too)."""

from pathlib import Path
import json
import os
import re
import socket
import socketserver
import stat
import threading

import paramiko


class Tunnel:
    """Forward localhost ports to the same ports on the Pi over one ssh connection."""

    def __init__(self, host: str, user: str, port: int, password: str, ports: tuple[int, ...], keys: tuple[str, ...] = ()):
        self.target = f"{user}@{host}:{port}"
        self.error: str | None = None
        self.connected = False
        self.closed = False
        self.command: RemoteCommand | None = None
        self.sftp: paramiko.SFTPClient | None = None
        self.sftp_lock = threading.Lock()
        self.client = paramiko.SSHClient()
        self.servers: list[socketserver.ThreadingTCPServer] = []
        threading.Thread(target=self.run, args=(host, user, port, password, ports, keys), daemon=True).start()

    def run(self, host: str, user: str, port: int, password: str, ports: tuple[int, ...], keys: tuple[str, ...]) -> None:
        try:
            self.client.load_system_host_keys()
            self.client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            # a blank password falls back to the host's key files, the ssh agent and the default keys
            self.client.connect(host, port, user, password=password or None, key_filename=list(keys) or None, timeout=5,
                                banner_timeout=10, auth_timeout=10, look_for_keys=not password, allow_agent=not password)
            transport = self.client.get_transport()
            assert transport is not None
            transport.set_keepalive(5)
            bound: list[socketserver.ThreadingTCPServer] = []
            try:
                for remote_port in ports:
                    bound.append(forward_server(transport, remote_port))
            except OSError:
                for server in bound:
                    server.server_close()
                raise
            if self.closed:  # disconnected while connecting
                for server in bound:
                    server.server_close()
                return
            for server in bound:
                threading.Thread(target=server.serve_forever, daemon=True).start()
            self.servers, self.connected = bound, True
        except Exception as error:
            self.error = str(error) or type(error).__name__
            self.close()

    def alive(self) -> bool:
        """Still connecting, or connected with the ssh session up."""
        if self.closed:
            return False
        transport = self.client.get_transport()
        return not self.connected or (transport is not None and transport.is_active())

    def start(self, command: str) -> "RemoteCommand":
        """Run a shell command on the host over the tunnel's ssh session."""
        transport = self.client.get_transport()
        assert transport is not None
        self.command = RemoteCommand(transport, command)
        return self.command

    def browse(self, folder: str) -> tuple[str, list[tuple[str, bool]]]:
        """A host folder's absolute path and its visible entries as (name, is a folder), folders first."""
        # one sftp request at a time: concurrent callbacks would interleave on its channel
        with self.sftp_lock:
            try:
                if self.sftp is None:
                    self.sftp = self.client.open_sftp()
                return list_folder(self.sftp, folder)
            except (paramiko.SSHException, EOFError) as error:
                self.sftp = None
                raise OSError(f"sftp: {error or type(error).__name__}") from error

    def close(self) -> None:
        # ctrl-c first, so the controller releases the drives before the session drops
        if self.command is not None and self.command.running():
            self.command.interrupt()
            self.command.channel.status_event.wait(5)
        self.closed, self.connected = True, False
        for server in self.servers:
            server.shutdown()
            server.server_close()
        self.client.close()


def list_folder(sftp: paramiko.SFTPClient, folder: str) -> tuple[str, list[tuple[str, bool]]]:
    folder = sftp.normalize(folder)
    entries = []
    for entry in sftp.listdir_attr(folder):
        if entry.filename.startswith(".") or entry.filename == "__pycache__":
            continue
        mode = entry.st_mode or 0
        if stat.S_ISLNK(mode):  # a link counts as what it points to
            try:
                mode = sftp.stat(f"{folder.rstrip('/')}/{entry.filename}").st_mode or 0
            except OSError:
                continue
        entries.append((entry.filename, stat.S_ISDIR(mode)))
    return folder, sorted(entries, key=lambda e: (not e[1], e[0].lower()))


class RemoteCommand:
    """A command in a pty on the host, so its prompts take enter and ctrl-c interrupts it, as in a terminal."""

    KEPT = 20000  # characters of output kept

    def __init__(self, transport: paramiko.Transport, command: str):
        self.output, self.answered = "", 0  # answered: output length at the last enter
        self.lock = threading.Lock()
        self.channel = transport.open_session()
        self.channel.get_pty(width=160)
        self.channel.exec_command(command)
        threading.Thread(target=self.read, daemon=True).start()

    def read(self) -> None:
        try:
            while data := self.channel.recv(4096):
                text = re.sub(r"\x1b\[[0-9;?]*[A-Za-z]|\r", "", data.decode(errors="replace"))
                with self.lock:
                    text = self.output + text
                    trimmed = max(len(text) - self.KEPT, 0)
                    self.output, self.answered = text[trimmed:], max(self.answered - trimmed, 0)
        except (OSError, EOFError, paramiko.SSHException):
            pass

    def running(self) -> bool:
        return not self.channel.closed and not self.channel.exit_status_ready()

    def exit_status(self) -> int | None:
        return self.channel.recv_exit_status() if self.channel.exit_status_ready() else None

    def prompt(self) -> str | None:
        """The 'press enter' prompt the command waits on, if any."""
        with self.lock:
            output, answered = self.output, self.answered
        at = output.lower().rfind("press enter")
        if at < 0 or at < answered or not self.running():
            return None
        return output[output.rfind("\n", 0, at) + 1:].split("\n")[0].strip()

    def enter(self) -> None:
        with self.lock:
            self.answered = len(self.output)
        self.channel.send(b"\r")

    def interrupt(self) -> None:
        self.channel.send(b"\x03")


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
    return server


def pipe(receive, send) -> None:
    try:
        while data := receive(65536):
            send(data)
    except (OSError, EOFError, paramiko.SSHException):
        pass


SSH_CONFIG = Path.home() / ".ssh" / "config"
CONTROLLERS_FILE = Path.home() / ".config" / "exopt" / "controllers.json"  # the dashboard's controllers, last used first
SSH_KEYS = {"hostname": "HostName", "user": "User", "port": "Port", "password": "Password"}


def known_hosts() -> dict[str, dict]:
    """The concrete Host aliases of ~/.ssh/config: hostname, user, port, key files, password.

    OpenSSH has no password option; one is read from `Password` behind `IgnoreUnknown Password`."""
    if not SSH_CONFIG.exists():
        return {}
    config = paramiko.SSHConfig.from_path(str(SSH_CONFIG))
    hosts = {}
    for alias in sorted(config.get_hostnames()):
        if any(c in alias for c in "*?!"):
            continue
        entry = config.lookup(alias)
        keys = [k for k in entry.get("identityfile", []) if os.path.exists(k)]
        hosts[alias] = dict(hostname=entry.get("hostname", alias), user=entry.get("user", ""),
                            port=int(entry.get("port", 22)), keys=keys, password=entry.get("password", ""))
    return hosts


def save_host(alias: str, hostname: str | None = None, user: str | None = None, port: int | None = None,
              password: str | None = None) -> None:
    """Write an alias's given HostName, User, Port and Password to ~/.ssh/config, adding the alias if new."""
    values = {k: str(v) for k, v in dict(hostname=hostname, user=user, port=port, password=password).items() if v}
    lines = SSH_CONFIG.read_text().splitlines() if SSH_CONFIG.exists() else []
    # OpenSSH refuses the whole file over an unknown option unless told to skip it, before any Host block
    if password and not any(re.match(r"\s*ignoreunknown\s.*\bpassword\b", line, re.I) for line in lines):
        lines[0:0] = ["IgnoreUnknown Password", ""]
    block = lambda line: re.match(r"\s*(host|match)\s", line, re.I)
    start = next((i for i, line in enumerate(lines) if re.fullmatch(rf"\s*host\s+{re.escape(alias)}\s*", line, re.I)), None)
    if start is None:
        # before any catch-all block, as ssh keeps the first value it finds
        at = next((i for i, line in enumerate(lines) if re.fullmatch(r"\s*(host\s+\*|match\s+all)\s*", line, re.I)),
                  len(lines))
        added = [f"Host {alias}", *(f"    {SSH_KEYS[k]} {v}" for k, v in values.items())]
        before = [""] if at and lines[at - 1].strip() else []
        lines[at:at] = before + added + ([""] if at < len(lines) else [])
    else:
        end = next((i for i in range(start + 1, len(lines)) if block(lines[i])), len(lines))
        # replace the keys the alias already sets, add the others right below its Host line
        for i in range(start + 1, end):
            key = lines[i].split(maxsplit=1)[0].lower() if lines[i].strip() else ""
            if key in values:
                indent = lines[i][: len(lines[i]) - len(lines[i].lstrip())]
                lines[i] = f"{indent}{SSH_KEYS[key]} {values.pop(key)}"
        lines[start + 1:start + 1] = [f"    {SSH_KEYS[k]} {v}" for k, v in values.items()]
    SSH_CONFIG.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    SSH_CONFIG.write_text("\n".join(lines) + "\n")
    SSH_CONFIG.chmod(0o600)


def saved_controllers() -> list[dict]:
    """The configured controllers, {name, host (an ~/.ssh/config alias), script}, last used first."""
    try:
        entries = json.loads(CONTROLLERS_FILE.read_text())
        return [c for c in entries if isinstance(c, dict) and {"name", "host", "script"} <= c.keys()]
    except (OSError, ValueError, TypeError):
        return []


def save_controller(name: str, host: str, script: str) -> None:
    """Add a controller first, replacing any of the same name (saving an existing one marks it last used)."""
    others = [c for c in saved_controllers() if c["name"] != name]
    CONTROLLERS_FILE.parent.mkdir(parents=True, exist_ok=True)
    CONTROLLERS_FILE.write_text(json.dumps([dict(name=name, host=host, script=script), *others], indent=1))
