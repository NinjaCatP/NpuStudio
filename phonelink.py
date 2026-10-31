"""Desktop <-> phone background link: queue commands for the Companion app's background service (find my phone)."""
import os, json, urllib.request
APP = os.path.dirname(os.path.abspath(__file__))
CMDS = {"find_phone", "stop_sound"}          # the only things the PC may push to the phone

def _desk():
    srv = os.path.join(APP, "models", "remote_server.json"); cli = os.path.join(APP, "models", "remote.json")
    if os.path.exists(srv): c = json.load(open(srv)); return "127.0.0.1", c.get("port", 47860), c["key"]
    if os.path.exists(cli): c = json.load(open(cli)); return c["host"], c.get("port", 47860), c["key"]
    raise RuntimeError("This PC isn't the desktop server and isn't linked to one.")

def _api(path, body=None, timeout=6):
    host, port, key = _desk()
    r = urllib.request.Request(f"http://{host}:{port}/api/v1/{path}", data=None if body is None else json.dumps(body).encode(),
                               headers={"X-Key": key, "Content-Type": "application/json"}, method="GET" if body is None else "POST")
    return json.load(urllib.request.urlopen(r, timeout=timeout))

def status():
    """{'connected': bool, 'last_seen_s': seconds or None, 'via': 'home'|'remote'|None}"""
    return _api("phone/status")

def send(name):
    if name not in CMDS: raise ValueError("not a phone command")
    return _api("phone/command", {"name": name, "args": {}})

def find_phone():
    try:
        r = send("find_phone"); s = r.get("status", {})
        if s.get("connected"): return "Ringing your phone now (it plays a loud alarm for up to 60 s; tap 'Found it' to stop)."
        ago = s.get("last_seen_s")
        return ("Queued: your phone isn't connected right now" + (f" (last seen {int(ago // 60)} min ago)" if ago is not None else " (never connected - open the Companion app v2.1 once)")
                + "; it will ring as soon as it reconnects.")
    except Exception as e: return f"Couldn't reach the desktop server: {e}"

def devices():
    """Mesh devices from the desktop server: [{'id','name','kind','online',...}]."""
    return _api("devices").get("devices", [])

def remote_chat(message, device, confirm, on_text=None, timeout=180):
    """Chat aimed at another device in the mesh. The desktop server understands it and runs it on the target; any
    risky action asks `confirm` HERE (on the PC you're chatting from) first. Returns the job result dict."""
    import time
    jid = _api("jobs", {"type": "chat", "message": message, "device": device})["id"]
    end = time.time() + timeout; shown = ""
    while time.time() < end:
        s = _api(f"jobs/{jid}")
        t = s.get("text") or ""
        if on_text and t != shown and t.startswith(shown): on_text(t[len(shown):]); shown = t
        if s["state"] == "confirm": _api(f"jobs/{jid}/confirm", {"yes": bool(confirm(s.get("confirm") or "Are you sure?"))})
        elif s["state"] not in ("running", "confirm"):
            r = _api(f"jobs/{jid}/result", timeout=10); r["_shown"] = shown; return r
        time.sleep(0.4)
    raise RuntimeError("the desktop server didn't answer in time")
