"""Allowlisted assistant actions for NpuStudio. No arbitrary commands, no deletion, no admin."""
import os, re, json, glob, time, difflib, subprocess, ctypes, datetime, logging, webbrowser, urllib.parse

CONFIRM = {"close_app", "sleep", "restart", "shutdown"}
NOWIN = 0x08000000  # CREATE_NO_WINDOW

SETTINGS = {"wifi": "network-wifi", "wi-fi": "network-wifi", "network": "network", "bluetooth": "bluetooth",
            "display": "display", "screen": "display", "sound": "sound", "audio": "sound", "battery": "batterysaver",
            "power": "powersleep", "update": "windowsupdate", "updates": "windowsupdate", "windows update": "windowsupdate",
            "apps": "appsfeatures", "notifications": "notifications", "personalization": "personalization",
            "background": "personalization-background", "colors": "colors", "theme": "themes", "mouse": "mousetouchpad",
            "touchpad": "devices-touchpad", "keyboard": "typing", "privacy": "privacy", "storage": "storagesense",
            "date": "dateandtime", "time": "dateandtime", "language": "regionlanguage", "accounts": "yourinfo",
            "printers": "printers", "night light": "nightlight", "about": "about", "vpn": "network-vpn",
            "windows updates": "windowsupdate", "check for updates": "windowsupdate", "default apps": "defaultapps",
            "wireless": "network-wifi", "internet": "network", "ethernet": "network-ethernet", "airplane mode": "network-airplanemode",
            "hotspot": "network-mobilehotspot", "mobile hotspot": "network-mobilehotspot", "proxy": "network-proxy",
            "monitor": "display", "resolution": "display", "speakers": "sound", "microphone": "privacy-microphone", "mic": "privacy-microphone",
            "camera": "privacy-webcam", "webcam": "privacy-webcam", "location": "privacy-location", "apps and features": "appsfeatures",
            "installed apps": "appsfeatures", "startup apps": "startupapps", "startup": "startupapps", "disk": "storagesense",
            "power and battery": "powersleep", "power & battery": "powersleep", "sleep": "powersleep", "battery saver": "batterysaver",
            "focus": "quiethours", "do not disturb": "quiethours", "lock screen": "lockscreen", "wallpaper": "personalization-background",
            "taskbar": "taskbar", "start menu": "personalization-start", "fonts": "fonts", "region": "regionlanguage",
            "devices": "devices", "printer": "printers", "gaming": "gaming-gamebar", "game mode": "gaming-gamemode",
            "accessibility": "easeofaccess", "narrator": "easeofaccess-narrator", "sign-in options": "signinoptions",
            "password": "signinoptions", "backup": "backup", "recovery": "recovery", "activation": "activation",
            "troubleshoot": "troubleshoot", "security": "windowsdefender", "windows security": "windowsdefender",
            "multitasking": "multitasking", "clipboard": "clipboard", "graphics": "display-advancedgraphics"}
# Words that mean a Settings page even without the word "settings" ("open windows update", "open bluetooth").
SETTINGS_BARE = {"windows update", "windows updates", "updates", "check for updates", "bluetooth", "wifi", "wi-fi", "wireless",
                 "display settings", "default apps", "apps and features", "installed apps", "startup apps", "notifications",
                 "privacy", "personalization", "storage", "power and battery", "power & battery", "battery saver", "night light",
                 "airplane mode", "mobile hotspot", "hotspot", "proxy", "vpn", "lock screen", "taskbar", "sign-in options",
                 "windows security", "accessibility", "multitasking", "about", "date and time", "network"}
FOLDERS = {"downloads": "Downloads", "documents": "Documents", "pictures": "Pictures", "desktop": "Desktop",
           "music": "Music", "videos": "Videos"}

TOOLS_PROMPT = """You are a helpful, concise assistant running on the user's Windows PC.
You can also control THIS PC (the one the user is chatting from). When the user asks you to DO something, reply with exactly one line:
ACTION: {"name": "<name>", "args": {...}}
Available actions:
open_app {"app": name} | close_app {"app": name} | open_url {"url": url} | web_search {"query": text}
open_folder {"folder": downloads|documents|pictures|desktop|music|videos} | settings {"page": wifi|bluetooth|display|sound|battery|updates|...}
volume {"level": 0-100} | brightness_change {"delta": -100..100} | volume_change {"delta": -100..100} | mute {} | unmute {}
brightness {"level": 0-100} | dark_mode {"on": true|false} | screenshot {} | timer {"minutes": n, "message": text}
battery {} | time {} | lock {} | sleep {} | restart {} | shutdown {}
media {"key": play_pause|next|previous|stop} | search_files {"query": name} | clipboard_get {} | clipboard_set {"text": text}
system_info {} | running_apps {}
Examples:
User: pull up the calculator -> ACTION: {"name": "open_app", "args": {"app": "calculator"}}
User: make the screen dimmer -> ACTION: {"name": "brightness_change", "args": {"delta": -15}}
User: I want to change my bluetooth settings -> ACTION: {"name": "settings", "args": {"page": "bluetooth"}}
For normal questions, just answer in plain text without ACTION."""

# ---------------- app index ----------------
_apps = None
def app_index():
    global _apps
    if _apps is not None: return _apps
    idx = {}
    for root in [os.path.join(os.environ.get("PROGRAMDATA", r"C:\ProgramData"), r"Microsoft\Windows\Start Menu\Programs"),
                 os.path.join(os.environ.get("APPDATA", ""), r"Microsoft\Windows\Start Menu\Programs")]:
        for p in glob.glob(os.path.join(root, "**", "*.lnk"), recursive=True):
            n = os.path.splitext(os.path.basename(p))[0]
            if re.search(r"uninstall|readme|help|documentation|website", n, re.I): continue
            idx.setdefault(n.lower(), ("lnk", p, n))
    try:
        out = subprocess.run(["powershell", "-NoProfile", "-Command",
                              "Get-StartApps | ForEach-Object { $_.Name + '|' + $_.AppID }"],
                             capture_output=True, text=True, timeout=30, creationflags=NOWIN).stdout
        for line in out.splitlines():
            if "|" in line:
                n, a = line.rsplit("|", 1)
                idx.setdefault(n.strip().lower(), ("uwp", a.strip(), n.strip()))
    except Exception: logging.exception("Get-StartApps")
    _apps = idx; return idx

def find_app(name):
    idx = app_index(); q = name.lower().strip()
    if q in idx: return idx[q]
    starts = [k for k in idx if k.startswith(q)] or [k for k in idx if re.search(r"\b" + re.escape(q), k)]
    if starts: return idx[min(starts, key=len)]
    m = difflib.get_close_matches(q, list(idx), n=1, cutoff=0.6)
    return idx[m[0]] if m else None

# ---------------- audio / screen ----------------
def _endpoint():
    from pycaw.pycaw import AudioUtilities
    sp = AudioUtilities.GetSpeakers()
    if hasattr(sp, "EndpointVolume"): return sp.EndpointVolume
    from ctypes import cast, POINTER
    from comtypes import CLSCTX_ALL
    from pycaw.pycaw import IAudioEndpointVolume
    return cast(sp.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None), POINTER(IAudioEndpointVolume))

def get_volume():
    import comtypes; comtypes.CoInitialize()
    e = _endpoint(); return round(e.GetMasterVolumeLevelScalar() * 100), bool(e.GetMute())

def set_volume(level=None, mute=None):
    import comtypes; comtypes.CoInitialize()
    e = _endpoint()
    if level is not None: e.SetMasterVolumeLevelScalar(max(0, min(100, level)) / 100.0, None)
    if mute is not None: e.SetMute(1 if mute else 0, None)

def _ps(cmd):
    r = subprocess.run(["powershell", "-NoProfile", "-Command", cmd], capture_output=True, text=True, timeout=30, creationflags=NOWIN)
    if r.returncode: raise RuntimeError(r.stderr.strip()[:200] or "failed")
    return r.stdout.strip()

def get_brightness():
    return int(_ps("(Get-CimInstance -Namespace root/WMI -ClassName WmiMonitorBrightness).CurrentBrightness | Select-Object -First 1"))

def set_brightness(level):
    level = max(0, min(100, int(level)))
    _ps(f"Get-CimInstance -Namespace root/WMI -ClassName WmiMonitorBrightnessMethods | Invoke-CimMethod -MethodName WmiSetBrightness -Arguments @{{Timeout=1;Brightness={level}}} | Out-Null")

def get_dark():
    import winreg
    k = winreg.OpenKey(winreg.HKEY_CURRENT_USER, r"Software\Microsoft\Windows\CurrentVersion\Themes\Personalize")
    return winreg.QueryValueEx(k, "AppsUseLightTheme")[0] == 0

def set_dark(on):
    import winreg
    k = winreg.OpenKey(winreg.HKEY_CURRENT_USER, r"Software\Microsoft\Windows\CurrentVersion\Themes\Personalize", 0, winreg.KEY_SET_VALUE)
    for n in ("AppsUseLightTheme", "SystemUsesLightTheme"): winreg.SetValueEx(k, n, 0, winreg.REG_DWORD, 0 if on else 1)
    try:  # tell running apps the theme changed
        ctypes.windll.user32.SendMessageTimeoutW(0xFFFF, 0x001A, 0, "ImmersiveColorSet", 2, 500, None)
    except Exception: pass

def battery():
    class S(ctypes.Structure):
        _fields_ = [("ac", ctypes.c_byte), ("flag", ctypes.c_byte), ("pct", ctypes.c_byte), ("r", ctypes.c_byte),
                    ("life", ctypes.c_ulong), ("full", ctypes.c_ulong)]
    s = S(); ctypes.windll.kernel32.GetSystemPowerStatus(ctypes.byref(s))
    pct = s.pct & 0xFF
    if pct == 255: return "I couldn't read the battery level."
    return f"Battery is at {pct}%" + (" and charging." if s.ac == 1 else " (on battery).")

def folder_path(name):
    if name.lower() in ("downloads",):
        try:
            import winreg
            k = winreg.OpenKey(winreg.HKEY_CURRENT_USER, r"Software\Microsoft\Windows\CurrentVersion\Explorer\Shell Folders")
            return winreg.QueryValueEx(k, "{374DE290-123F-4565-9164-39C4925E467B}")[0]
        except Exception: pass
    if name.lower() in ("documents", "pictures", "music", "videos", "desktop"):
        try:
            import winreg
            reg = {"documents": "Personal", "pictures": "My Pictures", "music": "My Music", "videos": "My Video", "desktop": "Desktop"}[name.lower()]
            k = winreg.OpenKey(winreg.HKEY_CURRENT_USER, r"Software\Microsoft\Windows\CurrentVersion\Explorer\Shell Folders")
            return winreg.QueryValueEx(k, reg)[0]
        except Exception: pass
    return os.path.join(os.path.expanduser("~"), FOLDERS.get(name.lower(), name))

def running_images():
    out = subprocess.run(["tasklist", "/fo", "csv", "/nh"], capture_output=True, text=True, creationflags=NOWIN).stdout
    return sorted({l.split('","')[0].strip('"') for l in out.splitlines() if l.strip()})

def find_process(name):
    q = re.sub(r"\.exe$", "", name.lower().strip()).replace(" ", "")
    imgs = running_images(); low = {i.lower()[:-4] if i.lower().endswith(".exe") else i.lower(): i for i in imgs}
    if q in low: return low[q]
    hit = [k for k in low if k.startswith(q)] or difflib.get_close_matches(q, list(low), n=1, cutoff=0.75)
    if hit: return low[hit[0]]
    a = find_app(name)  # e.g. "word" -> WINWORD.EXE via shortcut target
    if a and a[0] == "lnk":
        try:
            import win32com.client
            t = os.path.basename(win32com.client.Dispatch("WScript.Shell").CreateShortcut(a[1]).TargetPath).lower()
            for i in imgs:
                if i.lower() == t: return i
        except Exception: pass
    return None

# ---------------- router ----------------
NUM = r"(\d{1,3})\s*%?"
def route(text):
    t = text.lower().strip().rstrip(".!?")
    t = re.sub(r"^(please |can you |could you |hey |would you )+", "", t)
    t = re.sub(r"\s+(please|for me|now)$", "", t)
    r = _route_more(t)
    if r: return r
    if re.fullmatch(r"(check|look|scan) for (windows )?updates", t): return {"name": "settings", "args": {"page": "windows update"}}
    m = re.match(r"(?:open|show|go to|bring up|pull up|launch|start) (?:the |my )?(.+?)$", t)
    if m:
        x = m.group(1).strip()
        if x in ("windows settings", "settings", "the settings", "settings app", "pc settings", "control settings"): return {"name": "settings", "args": {"page": ""}}
        y = re.sub(r"^(windows |pc )", "", x) if x not in SETTINGS else x
        if x in SETTINGS_BARE or y in SETTINGS_BARE:
            k = x if x in SETTINGS else y if y in SETTINGS else "date" if "date" in x else "display"
            return {"name": "settings", "args": {"page": k}}
    m = re.search(r"(?:volume|sound)\s*(?:to|at|=)?\s*" + NUM, t) or re.search(r"set (?:the )?(?:volume|sound)(?: level)? to " + NUM, t)
    if m: return {"name": "volume", "args": {"level": int(m.group(1))}}
    if re.search(r"\b(turn|volume|sound)\b.*\b(up|louder)\b|\blouder\b", t): return {"name": "volume_change", "args": {"delta": 10}}
    if re.search(r"\b(turn|volume|sound)\b.*\b(down|quieter)\b|\bquieter\b|\bsofter\b", t): return {"name": "volume_change", "args": {"delta": -10}}
    if re.fullmatch(r"(un-?mute)( (the )?(sound|volume|audio|computer|laptop|pc))?", t): return {"name": "unmute", "args": {}}
    if re.fullmatch(r"mute( (the )?(sound|volume|audio|computer|laptop|pc))?", t): return {"name": "mute", "args": {}}
    m = re.search(r"brightness\s*(?:to|at|=)?\s*" + NUM, t)
    if m: return {"name": "brightness", "args": {"level": int(m.group(1))}}
    if re.search(r"\b(brighter|brightness up|increase (the )?brightness)\b", t): return {"name": "brightness_change", "args": {"delta": 15}}
    if re.search(r"\b(screen|display)\b.*\b(brighter|lighter)\b", t): return {"name": "brightness_change", "args": {"delta": 15}}
    if re.search(r"\b(screen|display)\b.*\b(darker|dimmer)\b", t): return {"name": "brightness_change", "args": {"delta": -15}}
    if re.search(r"\b(dimmer|dim (the )?screen|brightness down|decrease (the )?brightness|lower (the )?brightness)\b", t): return {"name": "brightness_change", "args": {"delta": -15}}
    if re.search(r"dark (mode|theme)", t):
        return {"name": "dark_mode", "args": {"on": not re.search(r"\b(off|disable|turn off|stop)\b", t)}}
    if re.search(r"light (mode|theme)", t):
        return {"name": "dark_mode", "args": {"on": bool(re.search(r"\b(off|disable)\b", t))}}
    if re.search(r"\b(take|grab|capture) (a )?(screenshot|screen ?shot)|^screenshot$", t): return {"name": "screenshot", "args": {}}
    m = re.search(r"(?:set )?(?:a )?(?:timer|reminder|remind me)\b.*?(\d+)\s*(min|minute|minutes|sec|second|seconds|hour|hours|hr)s?\b(?:\s*(?:to|that|about)\s+(.*))?", t)
    if m:
        n = int(m.group(1)); u = m.group(2)
        mins = n / 60 if u.startswith("sec") else n * 60 if u.startswith("h") else n
        msg = (m.group(3) or "").strip() or "Time's up!"
        if not m.group(3):
            m2 = re.search(r"remind me to (.+?) in \d+", t)
            if m2: msg = m2.group(1)
        return {"name": "timer", "args": {"minutes": mins, "message": msg}}
    if re.search(r"\bbattery\b", t) and not re.search(r"setting", t): return {"name": "battery", "args": {}}
    if re.search(r"^(what('s| is) the )?(time|date|day)( is it)?( today)?$|what time|what('s| is) (the )?(date|day)|today's date", t): return {"name": "time", "args": {}}
    if re.fullmatch(r"lock( (the|my) )?( ?(pc|computer|laptop|screen))?", t): return {"name": "lock", "args": {}}
    if re.fullmatch(r"(put (the |my )?(pc|computer|laptop) to sleep|sleep|go to sleep)", t): return {"name": "sleep", "args": {}}
    if re.fullmatch(r"(restart|reboot)( (the|my) (pc|computer|laptop))?", t): return {"name": "restart", "args": {}}
    if re.fullmatch(r"(shut ?down|turn off|power off)( (the|my) (pc|computer|laptop))?", t): return {"name": "shutdown", "args": {}}
    m = re.match(r"(?:open|show|go to) (?:the |my )?(?:windows )?(.+?) settings?$", t) or re.match(r"(?:open )?settings? (?:for )?(.+)$", t)
    if m and m.group(1).strip() in SETTINGS: return {"name": "settings", "args": {"page": m.group(1).strip()}}
    m = re.search(r"\b(" + "|".join(sorted(map(re.escape, SETTINGS), key=len, reverse=True)) + r") settings?\b", t)
    if m and re.search(r"\b(open|change|show|go|need|want|adjust|get)\b", t): return {"name": "settings", "args": {"page": m.group(1)}}
    if re.fullmatch(r"open (windows )?settings", t): return {"name": "settings", "args": {"page": ""}}
    m = re.match(r"(?:search (?:the web |google |online )?for|google|look up) (.+)$", t)
    if m: return {"name": "web_search", "args": {"query": m.group(1)}}
    m = re.match(r"(?:open|go to|visit|launch) (\S+\.(?:com|org|net|io|gov|edu|tv|co|ai|app|dev)\S*|https?://\S+)$", t)
    if m: return {"name": "open_url", "args": {"url": m.group(1)}}
    m = re.match(r"(?:open|show) (?:my |the )?(downloads|documents|pictures|desktop|music|videos)(?: folder)?$", t)
    if m: return {"name": "open_folder", "args": {"folder": m.group(1)}}
    m = re.match(r"(?:close|quit|exit|kill) (?:the )?(.+?)(?: app| window| program)?$", t)
    if m: return {"name": "close_app", "args": {"app": m.group(1)}}
    m = re.match(r"(?:open|launch|start|run|pull up|bring up|fire up|load up) (?:the |my )?(?:app )?(.+?)(?: app| program)?$", t)
    if m: return {"name": "open_app", "args": {"app": m.group(1)}}
    return None

def _route_more(t):
    if re.fullmatch(r"(play|pause|resume|play/pause|play pause|toggle play(back)?|pause (the )?(music|song|video|media)|play (the )?(music|song|media)|resume (the )?(music|song|media))", t):
        return {"name": "media", "args": {"key": "play_pause"}}
    if re.fullmatch(r"(next|skip)( (the )?(song|track|video))?|next song|skip (this )?(song|track)", t): return {"name": "media", "args": {"key": "next"}}
    if re.fullmatch(r"(previous|prev|last|go back)( (the )?(song|track|video))?|previous song", t): return {"name": "media", "args": {"key": "previous"}}
    if re.fullmatch(r"stop (the )?(music|song|playback|media)", t): return {"name": "media", "args": {"key": "stop"}}
    m = re.match(r"(?:find|search(?: for)?|locate|where is|where's) (?:the |my )?(?:file|files|document|folder)s?(?: named| called)? (.+?)(?: on (?:my|the) (?:pc|computer|laptop))?$", t)
    if m: return {"name": "search_files", "args": {"query": m.group(1)}}
    if re.search(r"\b(what('s| is) (on|in) (my |the )?clipboard|read (my |the )?clipboard|clipboard contents?|paste (my |the )?clipboard)\b", t): return {"name": "clipboard_get", "args": {}}
    m = re.match(r"copy (.+?) to (?:my |the )?clipboard$", t)
    if m: return {"name": "clipboard_set", "args": {"text": m.group(1).strip("\"' ")}}
    if re.search(r"\b(system info|system information|pc specs|computer specs|specs of (this|my) (pc|computer|laptop)|how much (ram|memory)|(cpu|gpu|ram) (usage|info)|uptime|disk space|free space)\b", t):
        return {"name": "system_info", "args": {}}
    if re.fullmatch(r"(what|which) (apps|programs|windows|applications)( do i have| have i got)? (are |is )?(open|running)( right| (right )?now| on (my|the|this) (pc|computer|laptop))?"
                    r"|what('s| is) (open|running)( right| (right )?now| on (my|the|this) (pc|computer|laptop))?"
                    r"|(list|show( me)?|tell me)( (all|the|my))? (open|running) (apps|programs|windows|applications)"
                    r"|(list|show( me)?)( (all|the|my))? (apps|programs|windows|applications) (that are )?(open|running)"
                    r"|(open|running) (apps|programs|applications)\??", t):
        return {"name": "running_apps", "args": {}}
    return None

def parse_action(text, names=None):
    if names is not None:     # phone actions: same parser, other name list
        global ALL
        saved = ALL; ALL = {n: None for n in names}
        try: return parse_action(text)
        finally: ALL = saved
    m = re.search(r"ACTION:\s*(\{.*\})", text, re.S)
    if not m:
        t = text.strip()
        m2 = re.match(r"(?:ACTION:\s*)?(\w+)\s*(\{.*\})?\s*$", t, re.S)  # e.g. brightness {"level": 50}
        if m2 and m2.group(1) in ALL:
            try: return {"name": m2.group(1), "args": json.loads(m2.group(2) or "{}")}
            except Exception: return {"name": m2.group(1), "args": {}}
        m3 = re.match(r"(?:ACTION:\s*)?(\{.*\})\s*$", t, re.S)  # e.g. {"settings": {"page": "bluetooth"}}
        if m3:
            try:
                d = json.loads(m3.group(1))
                if isinstance(d, dict) and d.get("name") in ALL:
                    return {"name": d["name"], "args": d.get("args") if isinstance(d.get("args"), dict) else {}}
                if isinstance(d, dict) and len(d) == 1:
                    k, v = next(iter(d.items()))
                    if k in ALL: return {"name": k, "args": v if isinstance(v, dict) else {}}
            except Exception: pass
        return None
    s = m.group(1)
    for cand in (s, s[:s.rfind("}") + 1], s.split("\n")[0]):
        try:
            d = json.loads(cand)
            if isinstance(d, dict) and "name" in d:
                d.setdefault("args", {}); return d if d["name"] in ALL else None
        except Exception: pass
    m = re.search(r'"name"\s*:\s*"(\w+)"', s)
    if m and m.group(1) in ALL:
        args = dict(re.findall(r'"(\w+)"\s*:\s*"([^"]*)"', s)); args.pop("name", None)
        for k, v in re.findall(r'"(\w+)"\s*:\s*(-?\d+|true|false)', s): args[k] = json.loads(v)
        return {"name": m.group(1), "args": args}
    return None

# ---------------- execution ----------------
def describe(a):
    n, g = a["name"], a.get("args", {})
    return {"close_app": f"close {g.get('app', '?')}", "sleep": "put the PC to sleep", "restart": "restart the PC",
            "shutdown": "shut down the PC", "open_url": f"open {g.get('url', '?')}"}.get(n, n)

def execute(a, confirm=None, timer_cb=None):
    """Run an allowlisted action. confirm(msg)->bool is required for CONFIRM actions. Returns a short result string."""
    n, g = a["name"], a.get("args", {}) or {}
    if n not in ALL: return f"I can't do '{n}'."
    if n in CONFIRM:
        if not confirm or not confirm(f"Do you want me to {describe(a)}?"): return f"Cancelled: {describe(a)}."
    return ALL[n](g, timer_cb)

def _open_app(g, _):
    name = str(g.get("app", "")).strip()
    a = find_app(name)
    if not a: return f"I couldn't find an app called '{name}'."
    if a[0] == "lnk": os.startfile(a[1])
    else: subprocess.Popen(["explorer.exe", "shell:AppsFolder\\" + a[1]], creationflags=NOWIN)
    return f"Opened {a[2]}."

def _close_app(g, _):
    name = str(g.get("app", "")).strip(); img = find_process(name)
    if not img: return f"'{name}' doesn't seem to be running."
    if img.lower() in ("explorer.exe", "csrss.exe", "winlogon.exe", "svchost.exe", "lsass.exe", "dwm.exe", "pythonw.exe", "python.exe"):
        return f"I won't close {img}, it's needed by Windows or by me."
    subprocess.run(["taskkill", "/IM", img, "/F"], capture_output=True, creationflags=NOWIN)
    return f"Closed {img}."

def _open_url(g, _):
    u = str(g.get("url", "")).strip()
    if not re.match(r"https?://", u): u = "https://" + u
    webbrowser.open(u); return f"Opened {u}."

def _search(g, _):
    q = str(g.get("query", "")).strip()
    webbrowser.open("https://www.bing.com/search?q=" + urllib.parse.quote_plus(q)); return f"Searched the web for '{q}'."

def _folder(g, _):
    f = str(g.get("folder", "downloads")).strip().strip('"')
    if os.path.isabs(f) and os.path.isdir(f): os.startfile(f); return f"Opened {f}."
    p = folder_path(f)
    os.startfile(p); return f"Opened {p}."

def _settings(g, _):
    p = str(g.get("page", "")).lower().strip()
    page = SETTINGS.get(p, p.replace(" ", "") if re.fullmatch(r"[a-z\- ]*", p) else "")
    os.startfile("ms-settings:" + page); return f"Opened {p or 'Windows'} settings."

def _volume(g, _):
    lv = int(g.get("level", 50)); set_volume(lv, mute=False if lv > 0 else None); return f"Volume set to {max(0, min(100, lv))}%."
def _vchange(g, _):
    v, _m = get_volume(); lv = max(0, min(100, v + int(g.get("delta", 10)))); set_volume(lv, mute=False); return f"Volume set to {lv}%."
def _mute(g, _): set_volume(mute=True); return "Sound muted."
def _unmute(g, _): set_volume(mute=False); return "Sound unmuted."
def _bright(g, _):
    try: set_brightness(g.get("level", 50)); return f"Brightness set to {max(0, min(100, int(g.get('level', 50))))}%."
    except Exception as e: return f"I couldn't change the brightness ({e})."
def _bchange(g, _):
    try: lv = max(0, min(100, get_brightness() + int(g.get("delta", 15)))); set_brightness(lv); return f"Brightness set to {lv}%."
    except Exception as e: return f"I couldn't change the brightness ({e})."
def _dark(g, _):
    on = g.get("on", True); on = on if isinstance(on, bool) else str(on).lower() in ("true", "on", "1", "yes")
    set_dark(on); return "Dark mode is on." if on else "Light mode is on."
def _shot(g, _):
    from PIL import ImageGrab
    d = os.path.join(folder_path("pictures"), "Screenshots"); os.makedirs(d, exist_ok=True)
    p = os.path.join(d, datetime.datetime.now().strftime("NpuStudio %Y-%m-%d %H-%M-%S.png"))
    ImageGrab.grab(all_screens=True).save(p); return f"Screenshot saved to {p}."
def _timer(g, cb):
    try: mins = float(g.get("minutes", 5))
    except Exception: mins = 5
    msg = str(g.get("message", "")) or "Time's up!"
    if cb: cb(mins * 60, msg)
    nice = f"{int(mins*60)} seconds" if mins < 1 else f"{mins:g} minute" + ("s" if mins != 1 else "")
    return f"Timer set for {nice}" + (f": {msg}" if msg != "Time's up!" else ".")
def _battery(g, _): return battery()
def _time(g, _):
    n = datetime.datetime.now(); return n.strftime("It's %I:%M %p on %A, %B %d, %Y.").replace(" 0", " ")
def _lock(g, _): ctypes.windll.user32.LockWorkStation(); return "Locked the PC."
def _sleep(g, _): ctypes.windll.powrprof.SetSuspendState(0, 0, 0); return "Going to sleep."
def _restart(g, _): subprocess.Popen(["shutdown", "/r", "/t", "5"], creationflags=NOWIN); return "Restarting in 5 seconds."
def _shutdown(g, _): subprocess.Popen(["shutdown", "/s", "/t", "5"], creationflags=NOWIN); return "Shutting down in 5 seconds."

VK = {"play_pause": 0xB3, "next": 0xB0, "previous": 0xB1, "stop": 0xB2}
def _media(g, _):
    k = str(g.get("key", "play_pause")).lower().replace("/", "_").replace(" ", "_")
    k = {"play": "play_pause", "pause": "play_pause", "prev": "previous", "skip": "next"}.get(k, k)
    if k not in VK: return "I can play/pause, skip to the next or previous track, or stop."
    u = ctypes.windll.user32; u.keybd_event(VK[k], 0, 0, 0); u.keybd_event(VK[k], 0, 2, 0)
    return {"play_pause": "Toggled play/pause.", "next": "Skipped to the next track.", "previous": "Went to the previous track.", "stop": "Stopped playback."}[k]

def _search_files(g, _):
    q = str(g.get("query", "")).strip().lower()
    if len(q) < 2: return "Tell me part of the file name to look for."
    home = os.path.expanduser("~"); roots = [folder_path(f) for f in FOLDERS] + [home]
    hits, seen, t0 = [], set(), time.time()
    for root in roots:
        for dp, dns, fns in os.walk(root):
            dns[:] = [d for d in dns if not d.startswith(".") and d.lower() not in ("appdata", "node_modules", "venv", "__pycache__")]
            if dp in seen: dns[:] = []; continue
            seen.add(dp)
            for f in fns + dns:
                if q in f.lower(): hits.append(os.path.join(dp, f))
            if len(hits) >= 15 or time.time() - t0 > 8: break
        if len(hits) >= 15 or time.time() - t0 > 8: break
    if not hits: return f"No files named like '{q}' in your user folders."
    return f"Found {len(hits)}{'+' if len(hits) >= 15 else ''}:\n" + "\n".join(hits[:15])

def _clip_get(g, _):
    r = subprocess.run(["powershell", "-NoProfile", "-Command", "Get-Clipboard -Raw"], capture_output=True, text=True, creationflags=NOWIN, timeout=10)
    t = (r.stdout or "").strip()
    return ("Clipboard: " + (t[:1500] + ("..." if len(t) > 1500 else ""))) if t else "The clipboard is empty (or holds a picture)."
def _clip_set(g, _):
    t = str(g.get("text", ""))[:20000]
    subprocess.run(["powershell", "-NoProfile", "-Command", "$input | Set-Clipboard"], input=t, text=True, creationflags=NOWIN, timeout=10)
    return "Copied to the clipboard."

def _sysinfo(g, _):
    import platform, shutil
    class MS(ctypes.Structure):
        _fields_ = [("l", ctypes.c_ulong), ("load", ctypes.c_ulong)] + [(n, ctypes.c_ulonglong) for n in ("tp", "ap", "tf", "af", "tv", "av", "ae")]
    m = MS(); m.l = ctypes.sizeof(MS); ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
    up = ctypes.windll.kernel32.GetTickCount64() / 1000
    cpu = _ps("(Get-CimInstance Win32_Processor).Name").strip()
    gpus = _ps("(Get-CimInstance Win32_VideoController).Name -join ', '").strip()
    drive = os.environ.get("SystemDrive", "C:") + "\\"; du = shutil.disk_usage(drive)
    return (f"{platform.node()}: Windows {platform.version()}\nCPU: {cpu}\nGPU: {gpus}\n"
            f"RAM: {(m.tp-m.ap)/2**30:.1f} of {m.tp/2**30:.1f} GB used ({m.load}%)\n"
            f"Disk {drive} {du.free/2**30:.0f} GB free of {du.total/2**30:.0f} GB\nUp for {int(up//86400)}d {int(up%86400//3600)}h {int(up%3600//60)}m")

def _running(g, _):
    out = _ps("Get-Process | ? { $_.MainWindowTitle } | Sort-Object ProcessName -Unique | % { $_.ProcessName + ' - ' + $_.MainWindowTitle }")
    lines = [l.strip() for l in out.splitlines() if l.strip()]
    return ("Open apps:\n" + "\n".join(lines[:40])) if lines else "No app windows are open."

ALL = {"open_app": _open_app, "close_app": _close_app, "open_url": _open_url, "web_search": _search, "open_folder": _folder,
       "settings": _settings, "volume": _volume, "volume_change": _vchange, "mute": _mute, "unmute": _unmute,
       "brightness": _bright, "brightness_change": _bchange, "dark_mode": _dark, "screenshot": _shot, "timer": _timer,
       "battery": _battery, "time": _time, "lock": _lock, "sleep": _sleep, "restart": _restart, "shutdown": _shutdown,
       "media": _media, "search_files": _search_files, "clipboard_get": _clip_get, "clipboard_set": _clip_set,
       "system_info": _sysinfo, "running_apps": _running}

# ---------------- phone actions (run by the Android app, not here) ----------------
PHONE = ["open_app", "flashlight", "volume", "volume_change", "mute", "unmute", "alarm", "timer", "open_url", "web_search",
         "battery", "clipboard_set", "notify", "time", "save_image", "share_image"]
PHONE_CONFIRM = {"open_url"}
PHONE_PROMPT = """You are a helpful, concise assistant. The user is chatting from their ANDROID PHONE, and you control the PHONE (not a PC).
When the user asks you to DO something on the phone, reply with exactly one line:
ACTION: {"name": "<name>", "args": {...}}
Available phone actions:
open_app {"app": name} | flashlight {"on": true|false} | volume {"level": 0-100} | volume_change {"delta": -100..100} | mute {} | unmute {}
alarm {"hour": 0-23, "minute": 0-59, "message": text} | timer {"minutes": n, "message": text} | open_url {"url": url} | web_search {"query": text}
battery {} | clipboard_set {"text": text} | notify {"title": text, "text": text} | time {} | save_image {} | share_image {}
Examples:
User: turn on the flashlight -> ACTION: {"name": "flashlight", "args": {"on": true}}
User: wake me up at 6:30 -> ACTION: {"name": "alarm", "args": {"hour": 6, "minute": 30, "message": "Wake up"}}
User: open spotify -> ACTION: {"name": "open_app", "args": {"app": "spotify"}}
For normal questions, just answer in plain text without ACTION."""

def route_phone(text):
    t = text.lower().strip().rstrip(".!?")
    t = re.sub(r"^(please |can you |could you |hey |would you )+", "", t); t = re.sub(r"\s+(please|for me|now)$", "", t)
    if re.search(r"\b(flash ?light|torch)\b", t): return {"name": "flashlight", "args": {"on": not re.search(r"\b(off|disable|stop)\b", t)}}
    m = re.search(r"(?:alarm|wake me(?: up)?)\b.*?\b(\d{1,2})(?::(\d{2}))?\s*(am|pm|a\.m\.|p\.m\.)?", t)
    if m:
        h = int(m.group(1)); mi = int(m.group(2) or 0); ap = (m.group(3) or "").replace(".", "")
        if ap == "pm" and h < 12: h += 12
        if ap == "am" and h == 12: h = 0
        return {"name": "alarm", "args": {"hour": h % 24, "minute": mi % 60, "message": "NpuStudio alarm"}}
    if re.search(r"\b(save|download) (the |this |that )?(last )?(picture|image|photo|result)\b", t): return {"name": "save_image", "args": {}}
    if re.search(r"\bshare (the |this |that )?(last )?(picture|image|photo|result)\b", t): return {"name": "share_image", "args": {}}
    m = re.match(r"copy (.+?) to (?:my |the )?clipboard$", t)
    if m: return {"name": "clipboard_set", "args": {"text": m.group(1).strip("\"' ")}}
    m = re.match(r"(?:notify me|send (?:me )?a notification|remind me now)(?: that| saying| to)? (.+)$", t)
    if m: return {"name": "notify", "args": {"title": "NpuStudio", "text": m.group(1)}}
    a = route(text)        # shared phrasing: volume, mute, timer, battery, time, open url, web search, open app
    if a and a["name"] in PHONE: return a
    return None

def describe_phone(a):
    g = a.get("args", {}); return {"open_url": f"open {g.get('url', '?')} on the phone"}.get(a["name"], a["name"])
