# NpuStudio worker API (v1)

`worker_server.py` (start it with **NpuStudio-Server.bat**) lets other devices use this PC's AI: the NpuStudio laptop app today, and a phone app later.

## Network and security
- **Address:** port `47860`. The server listens only on this PC's home-network (LAN) address and its Tailscale address (100.64.0.0/10).
  - It never binds to 0.0.0.0. Callers that aren't on a private, Tailscale or loopback address are refused.
  - No port forwarding is needed. From outside the home, use Tailscale.
- **Auth:** every request needs the pairing key, sent as `X-Key: <key>` or `Authorization: Bearer <key>`. A wrong key returns `403`.
  - The key is created on first run and stored in `models\remote_server.json`, and the server window shows it.
  - To change the key, delete `models\remote_server.json` and restart the server.
- **Firewall:** `AllowLaptop-Firewall.ps1` adds an inbound rule for this app's Python on TCP 47860, limited to the Private profile and the local subnet. For Tailscale access, allow the Tailscale interface too.
- **No code over the wire:** requests carry only JSON and images, never pickled objects.

## Conventions
- Bodies are JSON (`Content-Type: application/json`).
- **Images are base64 PNG or JPEG strings.** A `data:image/png;base64,` prefix is accepted. Masks are images where white means the area to change.
- **Long work runs as a job:**
  1. `POST /api/v1/jobs` returns `{"id": ...}`.
  2. Poll `GET /api/v1/jobs/{id}` about every 0.5 s.
  3. When `state` is `done` or `error`, call `GET /api/v1/jobs/{id}/result`. That deletes the job.

## Endpoints

| Method | Path | Returns |
|---|---|---|
| GET | `/api/v1/health` | `{name, version, gpu, use, vram_used, vram_total, watts, temp, busy}`. NVIDIA fields are `null` or `"N/A"` when they can't be read. |
| GET | `/api/v1/monitor` | Chip Monitor data: `{desktop: <health>, chips: {"NVIDIA": {use, ded, shared, watts, wnote, temp}, "Intel Arc": {...}}}`. Unreadable values are `null`, which the app shows as N/A. On this desktop the "Intel Arc" entry is the UHD 770. |
| POST | `/api/v1/jobs` | `{id}`. The body is one of the job types below. |
| GET | `/api/v1/jobs/{id}` | `{id, type, state, progress, total, message, text, confirm}`. `state` is one of `running`, `confirm`, `done` or `error`. |
| POST | `/api/v1/jobs/{id}/confirm` | Body `{"yes": true\|false}`. Answers a chat action that is waiting for confirmation. |
| GET | `/api/v1/jobs/{id}/result` | The job result (see below). An error returns `500 {"error": ...}`. A job that isn't finished returns `409`. |
| POST | `/api/v1/chat/new` | `{ok: true}`. Clears the chat history. |

### Job types (`POST /api/v1/jobs`)

**gen**
```json
{"type": "gen", "prompt": "...", "width": 1080, "height": 2400, "steps": 4, "seed": 42}
```
The image is drawn at the closest 64-multiple base size, upscaled with Sharp, then center-cropped to exactly width×height.
Result: `{image, device, seconds}`.

**upscale**
```json
{"type": "upscale", "image": "<b64>", "scale": 2|4|8|16, "model": "sharp"|"fast", "format": "PNG"|"JPEG"}
```
Result: `{image, device, seconds}`.

**remove_bg**
```json
{"type": "remove_bg", "image": "<b64>"}
```
Result: a PNG with transparency.

**erase**
```json
{"type": "erase", "image": "<b64>", "mask": "<b64>"}
```

**change** (AI Change)
```json
{"type": "change", "image": "<b64>", "instruction": "change outfit to a red dress", "mask": "<b64, optional>", "method": "Auto"|"Redraw"|"Recolor"}
```
The picture is analyzed first.
- If no mask is sent, the target is auto-selected from the instruction. When no target can be found, the change applies to the whole image.

Result: `{image, device, method, prompt, analysis, seconds}`.

**chat**
```json
{"type": "chat", "message": "open notepad"}
```
- While the reply is being written, `text` in the job status grows.
- Allowlisted PC actions (open or close an app, volume, brightness, folders and so on) run **on this desktop**. Only the actions in `actions.py` are possible.
- Actions marked CONFIRM pause the job with `state: "confirm"` and `confirm: "Do you want me to ...?"`. The client must ask the user and then `POST .../confirm {"yes": true}` or `{"yes": false}`. With no answer in 120 s, the action is cancelled.

Result: `{text, action, action_result, device, seconds}`.

### Example (PowerShell)
```powershell
$h = @{ 'X-Key' = '<key>' }; $u = 'http://192.168.0.230:47860/api/v1'
$img = [Convert]::ToBase64String([IO.File]::ReadAllBytes('in.png'))
$id = (Invoke-RestMethod "$u/jobs" -Method Post -Headers $h -ContentType 'application/json' -Body (@{type='upscale'; image=$img; scale=4} | ConvertTo-Json)).id
do { Start-Sleep 1; $s = Invoke-RestMethod "$u/jobs/$id" -Headers $h } while ($s.state -eq 'running')
$r = Invoke-RestMethod "$u/jobs/$id/result" -Headers $h; [IO.File]::WriteAllBytes('out.png', [Convert]::FromBase64String($r.image))
```

## Laptop link (internal)
The NpuStudio laptop app uses a compact binary form of the same jobs: `POST /job`, `GET /job/{id}`, `GET /job/{id}/result`, and `POST /call` for upscale tiles. These carry a JSON header plus PNG and .npy blobs (`wire.py`) and use the same key and network rules. They're internal and may change. Phone apps should use `/api/v1`.

## Mobile web UI (`/app`), so the phone app updates from the desktop
- **What's served:** the desktop hosts the phone UI as plain HTML/JS/CSS from the `webapp\` folder, at `http://<desktop>:47860/app/`.
- **Versioning:** `webapp/version.json` holds `{ui_version, api, min_shell}`. Updating NpuStudio on the desktop updates the phone UI, so there's no app rebuild. Files are sent with `Cache-Control: no-cache`.
- **Access:** `/app` needs no key, because it holds no secrets. It still has the same LAN/Tailscale-only caller rule. The page keeps the pairing key in `localStorage` and sends it on every `/api/v1` call.
- **Android shell, a thin WebView:**
  1. Pairing screen: enter the address and key (or scan a QR code later).
  2. Load `/app/`.
  3. Check `min_shell` against its own shell version, and ask the user to update the shell only if it's too old.
- **JS bridge:** the shell injects `window.NpuShell`, with every call optional and feature-detected by the page:
  - `pickImage()` → Promise<base64>, the gallery picker
  - `share(base64, filename)`, the Android share sheet
  - `download(base64, filename)`, which saves to Downloads or Pictures
  - `notify(title, text)`, for example "Upscale done" while the app is in the background
  - `shellVersion` (int)
- **Status:** the current `index.html` is a stub that does pairing plus a health/GPU readout. The full UI comes later.
