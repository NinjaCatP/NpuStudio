# Lets the NpuStudio phone app reach this desktop over Tailscale (100.64.0.0/10) on TCP 47860.
# Run on the DESKTOP: right-click > Run with PowerShell (it asks for admin). Internet traffic outside Tailscale is not allowed.
if (-not ([Security.Principal.WindowsPrincipal][Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)) {
    Start-Process powershell.exe "-NoProfile -ExecutionPolicy Bypass -File `"$PSCommandPath`"" -Verb RunAs; exit
}
$py = Join-Path $PSScriptRoot 'venv\Scripts\python.exe'
Remove-NetFirewallRule -DisplayName 'NpuStudio worker (Tailscale)' -ErrorAction SilentlyContinue
New-NetFirewallRule -DisplayName 'NpuStudio worker (Tailscale)' -Direction Inbound -Action Allow -Protocol TCP -LocalPort 47860 `
    -RemoteAddress 100.64.0.0/10 -Program $py -Profile Any | Out-Null
Write-Host "Added firewall rule 'NpuStudio worker (Tailscale)': TCP 47860 from 100.64.0.0/10 only."
Read-Host 'Press Enter to close'
