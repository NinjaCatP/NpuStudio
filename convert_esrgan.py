import torch, torch.nn as nn, torch.nn.functional as F, openvino as ov, sys
class RDB(nn.Module):
    def __init__(s, nf=64, gc=32):
        super().__init__()
        s.conv1=nn.Conv2d(nf,gc,3,1,1); s.conv2=nn.Conv2d(nf+gc,gc,3,1,1); s.conv3=nn.Conv2d(nf+2*gc,gc,3,1,1)
        s.conv4=nn.Conv2d(nf+3*gc,gc,3,1,1); s.conv5=nn.Conv2d(nf+4*gc,nf,3,1,1); s.l=nn.LeakyReLU(0.2,True)
    def forward(s,x):
        x1=s.l(s.conv1(x)); x2=s.l(s.conv2(torch.cat((x,x1),1))); x3=s.l(s.conv3(torch.cat((x,x1,x2),1)))
        x4=s.l(s.conv4(torch.cat((x,x1,x2,x3),1))); x5=s.conv5(torch.cat((x,x1,x2,x3,x4),1)); return x5*0.2+x
class RRDB(nn.Module):
    def __init__(s,nf=64,gc=32):
        super().__init__(); s.rdb1=RDB(nf,gc); s.rdb2=RDB(nf,gc); s.rdb3=RDB(nf,gc)
    def forward(s,x): return s.rdb3(s.rdb2(s.rdb1(x)))*0.2+x
class RRDBNet(nn.Module):
    def __init__(s,nb=23,nf=64):
        super().__init__()
        s.conv_first=nn.Conv2d(3,nf,3,1,1); s.body=nn.Sequential(*[RRDB(nf) for _ in range(nb)])
        s.conv_body=nn.Conv2d(nf,nf,3,1,1); s.conv_up1=nn.Conv2d(nf,nf,3,1,1); s.conv_up2=nn.Conv2d(nf,nf,3,1,1)
        s.conv_hr=nn.Conv2d(nf,nf,3,1,1); s.conv_last=nn.Conv2d(nf,3,3,1,1); s.l=nn.LeakyReLU(0.2,True)
    def forward(s,x):
        f=s.conv_first(x); f=f+s.conv_body(s.body(f))
        f=s.l(s.conv_up1(F.interpolate(f,scale_factor=2,mode='nearest')))
        f=s.l(s.conv_up2(F.interpolate(f,scale_factor=2,mode='nearest')))
        return s.conv_last(s.l(s.conv_hr(f)))
class SRVGG(nn.Module):
    def __init__(s,nf=64,nc=32,up=4):
        super().__init__(); b=[nn.Conv2d(3,nf,3,1,1),nn.PReLU(nf)]
        for _ in range(nc): b+=[nn.Conv2d(nf,nf,3,1,1),nn.PReLU(nf)]
        b+=[nn.Conv2d(nf,3*up*up,3,1,1)]; s.body=nn.ModuleList(b); s.up=nn.PixelShuffle(up)
    def forward(s,x):
        o=x
        for m in s.body: o=m(o)
        return s.up(o)+F.interpolate(x,scale_factor=4,mode='nearest')
torch.set_grad_enabled(False)
for name,net,key in [("realesrgan-x4plus",RRDBNet(),"params_ema"),("realesr-general-x4v3",SRVGG(),"params")]:
    sd=torch.load(name.replace("realesrgan-x4plus","RealESRGAN_x4plus")+".pth",map_location="cpu")
    sd=sd.get(key,sd.get("params_ema",sd)); net.load_state_dict(sd,strict=True); net.eval()
    for t in (256,128):
        m=ov.convert_model(net,example_input=torch.rand(1,3,t,t),input=[1,3,t,t])
        ov.save_model(m,f"out/{name}-{t}.xml",compress_to_fp16=True); print(name,t,"ok")
