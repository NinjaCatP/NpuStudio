# Run on the LAPTOP: lets the desktop's NpuStudio use this laptop's chips ("Share this laptop's chips").
# Inbound TCP 47860 from the home network (local subnet) and Tailscale (100.64.0.0/10) only, for NpuStudio's own Python only.
# Right-click > Run with PowerShell (it asks for admin). Every request still needs the pairing key.
if (-not ([Security.Principal.WindowsPrincipal][Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)) {
    Start-Process powershell.exe "-NoProfile -ExecutionPolicy Bypass -File `"$PSCommandPath`"" -Verb RunAs; exit
}
$base = Get-ChildItem (Join-Path $PSScriptRoot 'python') -Directory -Filter 'cpython-*' | Select-Object -First 1
$progs = @()
foreach ($exe in 'pythonw.exe', 'python.exe') {
    if ($base) { $p = Join-Path $base.FullName $exe; if (Test-Path $p) { $progs += $p } }
    $v = Join-Path $PSScriptRoot "venv\Scripts\$exe"; if (Test-Path $v) { $progs += $v }
}
Get-NetFirewallRule -DisplayName 'NpuStudio share (LAN + Tailscale)*' -ErrorAction SilentlyContinue | Remove-NetFirewallRule
$i = 0
foreach ($p in $progs) {
    $i++
    New-NetFirewallRule -DisplayName "NpuStudio share (LAN + Tailscale) $i" -Direction Inbound -Action Allow -Protocol TCP -LocalPort 47860 `
        -RemoteAddress LocalSubnet, 100.64.0.0/10 -Program $p -Profile Any | Out-Null
    Write-Host "Allowed $p"
}
Write-Host "Done: TCP 47860 from the home network and Tailscale only."
Read-Host 'Press Enter to close'
