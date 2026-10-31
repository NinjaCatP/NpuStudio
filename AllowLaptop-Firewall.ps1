# Lets the NpuStudio laptop reach this PC's worker. Private (home) networks only, TCP 47860, this app's Python only.
# Right-click > Run with PowerShell (asks for admin).
if (-not ([Security.Principal.WindowsPrincipal][Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)) {
  Start-Process powershell -Verb RunAs -ArgumentList '-NoProfile','-ExecutionPolicy','Bypass','-File',"`"$PSCommandPath`""; exit }
$py = Join-Path (Split-Path -Parent $PSCommandPath) 'venv\Scripts\python.exe'
Remove-NetFirewallRule -DisplayName 'NpuStudio worker (LAN)' -ErrorAction SilentlyContinue
New-NetFirewallRule -DisplayName 'NpuStudio worker (LAN)' -Direction Inbound -Action Allow -Protocol TCP -LocalPort 47860 -Profile Private -RemoteAddress LocalSubnet -Program $py | Out-Null
'Firewall rule added (Private networks, local subnet only).'
