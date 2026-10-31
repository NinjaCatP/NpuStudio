"""'Start with Windows': a per-user HKCU\\...\\Run entry that opens NpuStudio with pythonw (no console) at sign-in,
plus a 'start the server with the app' setting (models/startup.json). No admin rights needed; off = entry removed."""
import os, sys, json, logging
APP = os.path.dirname(os.path.abspath(__file__)); CFG = os.path.join(APP, "models", "startup.json")
KEY = r"Software\Microsoft\Windows\CurrentVersion\Run"; NAME = "NpuStudio"

def command():
    pyw = os.path.join(APP, "venv", "Scripts", "pythonw.exe")
    if not os.path.exists(pyw): pyw = os.path.join(os.path.dirname(sys.executable), "pythonw.exe")   # whatever runs NpuStudio now
    return f'"{pyw}" "{os.path.join(APP, "npustudio.py")}" --autostart'

def enabled():
    try:
        import winreg
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, KEY) as k: v, _ = winreg.QueryValueEx(k, NAME)
        return bool(v)
    except Exception: return False

def set_enabled(on):
    import winreg
    with winreg.OpenKey(winreg.HKEY_CURRENT_USER, KEY, 0, winreg.KEY_SET_VALUE) as k:
        if on: winreg.SetValueEx(k, NAME, 0, winreg.REG_SZ, command())
        else:
            try: winreg.DeleteValue(k, NAME)
            except FileNotFoundError: pass
    logging.info("start with Windows: %s", on)

def _cfg():
    try: return json.load(open(CFG))
    except Exception: return {}

def server_on_start():
    return bool(_cfg().get("server", True))

def set_server_on_start(on):
    c = _cfg(); c["server"] = bool(on); os.makedirs(os.path.dirname(CFG), exist_ok=True); json.dump(c, open(CFG, "w"))

if __name__ == "__main__":   # python startup.py on|off|status
    a = sys.argv[1] if len(sys.argv) > 1 else "status"
    if a in ("on", "off"): set_enabled(a == "on")
    print("Start with Windows:", "ON" if enabled() else "OFF", "| server on start:", server_on_start(), "|", command())
