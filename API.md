# NpuStudio worker API (v1)

`worker_server.py` (start it with **NpuStudio-Server.bat**) lets other devices use this PC's AI: the NpuStudio laptop app today, and a phone app later.

## Network and security
- **Address:** port `47860`. The server listens only on this PC's home-network (LAN) address and its Tailscale address (100.64.0.0/10).
  - It never binds to 0.0.0.0. Callers that aren't on a private, Tailscale or loopback address are refused.
  - No port forwarding is needed. From outside the home, use Tailscale.
- **Auth:** every request needs the pairing key, sent as `X-Key: <key>` or `Authorization: Bearer <key>`. A wrong key returns `403`.
  - The key is created on first run and stored in `models\remote_server.json`, and the server window shows it.
  - To change the key, delete `models\remote_server.json` and restart the server.
- **Firewall:** `AllowLaptop-Firewall.ps1` adds an inbound rule for this app's Python on TCP 47860, limited to the Private profile and the local subnet. For Tailscale, run `AllowTailscale-Firewall.ps1` (TCP 47860 from 100.64.0.0/10 only).
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

## Cancel
`POST /api/v1/jobs/{id}/cancel` stops a job at its next progress step. Its state becomes `cancelled`, and `.../result` returns `410`.

## Other endpoints
- `GET /api/v1/info`: `{name, port, lan, tailscale[], ui, busy, jobs[]}`, used by the Desktop tab.
- `POST /api/v1/report`: the laptop app pushes its Chip Monitor numbers here every 3 s. `/monitor` lists them under `others` for 10 s.
- `GET /pair`: the pairing QR page. It only opens on the desktop itself (`http://127.0.0.1:47860/pair`, or **Show-PairingQR.bat**). It needs the `segno` package.

## Mobile web UI (`/app`) and the Android shell
- **What's served:** `webapp/` (`index.html`, `app.js`, `app.css`, `version.json`) is served at `http://<desktop>:47860/app/` with `Cache-Control: no-cache`. Updating the desktop updates the phone, with no APK rebuild.
- **Tabs:** Chat (with Yes/No confirms), Image Gen (including phone-wallpaper sizes), Edit/AI Change, Upscale, Monitor (desktop plus the laptop when it's online) and Desktop status. Every job shows progress and has a Cancel button.
- **Versioning:** `version.json` holds `{ui_version, api, min_shell}`. If the shell is older than `min_shell`, the page asks the user to update the app.
- **Key:** the shell opens `/app/#k=<key>`. The page saves the key in `localStorage`, removes it from the URL, and sends it as `X-Key` on every API call. `/app` itself needs no key, but it is LAN/Tailscale only.
- **JS bridge (`window.NpuShell`, Android only, feature-detected):**
  - `shellVersion()` → int
  - `pickImage(cbId)` opens the gallery. The page receives `window.npuCallback(cbId, base64 | null)`.
  - `saveImage(base64, name)` saves to Pictures/NpuStudio. `download` is an alias.
  - `share(base64, name)` opens the Android share sheet.
  - `notify(title, text)`: the page calls it when a job finishes while the app is in the background.
  - `unpair()` forgets the pairing and shows the pairing screen.
- **In a normal browser:** the same page works without the bridge. It falls back to a file input, a download link, the Web Share API and web notifications.
- **Android source and build:** see `android/README.md`. For use away from home, see `TAILSCALE.md`.

## v1.2 additions
- `POST /api/v1/upscale/check` `{width, height, scale}` → `{out_width, out_height, megapixels, memory_gb, memory_limit_gb, max_side, max_megapixels, ok, reason, fit_width, fit_height}`. Scale 2/4/8/16 (16 = two 4x passes). Limits: 18 GB budget, 16384 px side, 160 MP.
- `GET /api/v1/options` → scales, models, devices, memory limit, gen sizes (same as the PC app), AI Change methods.
- `upscale` job: `scale` up to 16, `device` (Auto/NVIDIA/Intel Arc/CPU/All), `format` AUTO|PNG|JPEG (AUTO = PNG up to 24 MP). Too-big requests fail with a suggested shrink size.
- `gen` job: `upscaler` sharp|fast. `change` job: `method`, `strength`, `grow` (-30..40 px), `whole`.
- `chat` job: `device` = `pc` (default; allowlisted actions run on the desktop, CONFIRM ones wait for `/confirm`) or `phone` (nothing runs on the PC; the result has `phone_action {name, args}`, `needs_confirm`, `confirm_text` and the web UI runs it through `NpuShell.phoneAction`).
- `/pair` (desktop only) also shows a combined Home + Tailscale QR when Tailscale is up.

## v1.3 additions (reverse link: desktop uses the laptop's chips)
- The laptop app starts `worker_server.py --share --parent <pid>` in the background while NpuStudio is open, if "Share this laptop's chips" is on (`models/share.json`, default on). It uses the same port (47860) and the desktop's pairing key from `models/remote.json`, so no extra pairing is needed. It exits when the app closes.
- Both servers bind 127.0.0.1, the LAN IP and any Tailscale (100.64.0.0/10) IPs, never 0.0.0.0. They re-check bindings every 30 s, so Tailscale installed later is picked up. The status is in `models/server_status.json` (`.share` on the laptop): `{state: running|error|conflict|stopped, ips, port, error}`.
- `POST /api/v1/report` (laptop → desktop) now includes `serve` (the share port), and the server records the sender's address as `_ip`. The desktop app reads it from `/api/v1/monitor` → `others` and caches it in `models/laptop.json`.
- Internal wire jobs: `upscale` takes kw `device` (the chip on the serving machine). `chat_ask(text, target, new=bool)` runs the serving PC's chat LLM and streams text as status messages prefixed with `\x00`; the result is `{text, tokens, seconds, device}`. Actions still run on the caller. `chat_new` resets the chat.
- Laptop firewall: `AllowShare-Firewall.ps1` (run as admin) allows TCP 47860 from LocalSubnet and 100.64.0.0/10 only.
