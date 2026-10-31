import sys; import os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import actions
P = [("Set brightness to a 100%", "brightness", {"level": 100}), ("set brightness to 40", "brightness", {"level": 40}),
 ("brightness 70%", "brightness", {"level": 70}), ("max brightness", "brightness", {"level": 100}), ("turn brightness all the way up", "brightness", {"level": 100}),
 ("minimum brightness", "brightness", {"level": 0}), ("turn the brightness up", "brightness_change", None), ("turn down the brightness", "brightness_change", None),
 ("dim the screen", "brightness_change", {"delta": -20}), ("make the screen brighter", "brightness_change", {"delta": 20}),
 ("make the screen darker", "brightness_change", {"delta": -20}), ("turn on auto brightness", "brightness_auto", {"on": True}),
 ("turn off adaptive brightness", "brightness_auto", {"on": False}), ("brightness up by 10", "brightness_change", {"delta": 10}),
 ("set volume to 30", "volume", {"level": 30}), ("volume 100%", "volume", {"level": 100}), ("turn it up", "volume_change", None),
 ("make it quieter", "volume_change", None), ("mute", "mute", {}), ("set ring volume to 50", "volume", {"level": 50, "stream": "ring"}),
 ("alarm volume max", "volume", {"level": 100, "stream": "alarm"}), ("turn the notification volume down", "volume_change", None),
 ("turn on do not disturb", "dnd", {"on": True}), ("turn off dnd", "dnd", {"on": False}), ("put my phone on vibrate", "ring_mode", {"mode": "vibrate"}),
 ("silent mode", "ring_mode", {"mode": "silent"}), ("turn the ringer on", "ring_mode", {"mode": "normal"}), ("lock rotation", "rotation_lock", {"on": True}),
 ("turn on auto rotate", "rotation_lock", {"on": False}), ("keep the screen on", "keep_screen_on", {"on": True}),
 ("turn on wifi", "settings_panel", {"panel": "wifi"}), ("open bluetooth settings", "settings_panel", {"panel": "bluetooth"}),
 ("turn off mobile data", "settings_panel", {"panel": "internet"}), ("turn on nfc", "settings_panel", {"panel": "nfc"}), ("turn on location", "settings_panel", {"panel": "location"}),
 ("open the camera", "camera", {"mode": "photo"}), ("record a video", "camera", {"mode": "video"}), ("navigate to the airport", "navigate", {"place": "the airport"}),
 ("find a coffee shop near me", "maps_search", {"query": "coffee shop"}), ("call mom", "call", {"who": "mom"}), ("call 614-555-0100", "call", {"who": "614-555-0100"}),
 ("text dave saying running late", "sms", {"who": "dave", "text": "running late"}), ("add an event called dentist", "calendar_event", {"title": "dentist"}),
 ("pause the music", "media", {"cmd": "pause"}), ("next song", "media", {"cmd": "next"}), ("play", "media", {"cmd": "play"}),
 ("how much storage do i have", "storage", {}), ("what network am i on", "network", {}), ("take a note buy milk", "note", {"text": "buy milk"}),
 ("show my notes", "notes", {}), ("read this aloud: hello there", "speak", {"text": "hello there"}), ("open settings for spotify", "app_settings", {"app": "spotify"}),
 ("find my phone", "find_phone", {}), ("where is my phone", "find_phone", {}), ("stop the sound", "stop_sound", {}),
 ("turn on the flashlight", "flashlight", {"on": True}), ("wake me up at 6:30 am", "alarm", None), ("set a timer for 10 minutes", "timer", None),
 ("open spotify", "open_app", {"app": "spotify"}), ("what's my battery", "battery", {}),
 ("take a screenshot", "_cant", None), ("lock the computer", "_cant", None), ("turn on dark mode", "_cant", None)]
ok = 0
for q, n, g in P:
    a = actions.route_phone(q); good = a and a["name"] == n and (g is None or all(a["args"].get(k) == v for k, v in g.items()))
    ok += bool(good); print("OK " if good else "BAD", q, "->", a)
# guard: model answers
G = [("Set brightness to a 100%", {"name": "volume", "args": {"level": 100}}, False), ("turn the volume up", {"name": "volume_change", "args": {}}, True),
     ("make it brighter", {"name": "volume", "args": {}}, False), ("brightness 50", {"name": "brightness", "args": {}}, True),
     ("switch on the torch", {"name": "volume", "args": {}}, False), ("hello", {"name": "screenshot", "args": {}}, False)]
for q, a, exp in G:
    r = bool(actions.phone_guard(q, a)); ok += r == exp; print("OK " if r == exp else "BAD", "guard", q, a["name"], r)
PC = [("find my phone", "find_phone"), ("where is my phone", "find_phone"), ("turn on night light", "night_light"), ("empty the recycle bin", "empty_recycle_bin"),
      ("open recent files", "recent_files"), ("turn on bluetooth", "settings"), ("open wifi settings", "settings"), ("battery saver", "settings"),
      ("set volume to 30", "volume"), ("set brightness to 50", "brightness"), ("open windows update", "settings")]
for q, n in PC:
    a = actions.route(q); good = a and a["name"] == n; ok += bool(good); print("OK " if good else "BAD", "PC", q, "->", a)
PW = [("Restart device", "restart"), ("restart the pc", "restart"), ("reboot", "restart"), ("restart it", "restart"), ("reboot this", "restart"),
      ("restart the laptop", "restart"), ("shut down", "shutdown"), ("shut down the laptop", "shutdown"), ("turn off the computer", "shutdown"),
      ("power off the device", "shutdown"), ("sleep", "sleep"), ("put the pc to sleep", "sleep"), ("lock", "lock"), ("lock the computer", "lock"),
      ("log off", "log_off"), ("sign out", "log_off"), ("restart chrome", None), ("turn off the flashlight", None)]
for q, n in PW:
    a = actions.route(q); good = (a or {}).get("name") == n and (n is None or n in actions.CONFIRM); ok += bool(good); print("OK " if good else "BAD", "power", q, "->", a)
PP = ["Restart device", "reboot the phone", "shut down", "turn off my phone", "lock", "sign out"]
for q in PP:
    a = actions.route_phone(q); good = a and a["name"] == "_cant" and "Android" in a["args"]["text"]; ok += bool(good); print("OK " if good else "BAD", "phone power", q)
RF = [("I'm unable to physically interact with devices. Run SHUTDOWN /S /F", True), ("As an AI, I can't restart computers.", True),
      ("Open cmd and type shutdown /r", True), ("Sure! Here's a short poem about autumn.", False), ("The capital of France is Paris.", False)]
for t_, exp in RF:
    r = actions.refusal(t_); ok += r == exp; print("OK " if r == exp else "BAD", "refusal", t_[:40], r)
LB = [({"name": "settings", "args": {"page": "windows update check"}}, "Settings"), ({"name": "log_off", "args": {}}, "Sign out"), ({"name": "system_info", "args": {}}, "System info")]
for a_, exp in LB:
    r = actions.label(a_); ok += r == exp; print("OK " if r == exp else "BAD", "label", r)
print(f"{ok}/{len(P) + len(G) + len(PC) + len(PW) + len(PP) + len(RF) + len(LB)}")
