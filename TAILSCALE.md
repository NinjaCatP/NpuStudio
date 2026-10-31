# Using the NpuStudio phone app away from home (Tailscale)

Tailscale makes a private network between your own devices. Nothing is opened to the internet.

1. Install Tailscale on the **desktop**, the **laptop** and the **phone** (tailscale.com/download, or the Play Store on the phone).
2. Sign in on all three with the **same account**.
3. On the desktop, run **AllowTailscale-Firewall.ps1** once (it asks for admin). It allows port 47860 only from Tailscale addresses (100.64.0.0/10).
4. Restart **NpuStudio-Server.bat** on the desktop. It now also listens on its Tailscale address (100.x.x.x).
5. Run **Show-PairingQR.bat** on the desktop. It shows two QR codes: one for home Wi-Fi and one for Tailscale. In the phone app, choose **Change pairing** and scan the **Tailscale** one.
   - Or type the desktop's 100.x address (shown in the Tailscale app, or on the Desktop tab) plus the key.

The Tailscale address also works at home, so you can pair with it and use it everywhere.
