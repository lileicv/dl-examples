import visdom
import torch
import time

vis = visdom.Visdom(env='main')

for i in range(2, 100):
    print(i)
    time.sleep(1)
    x = torch.arange(1,i).float()*0.1
    y1 = torch.sin(x)
    y2 = torch.cos(x)
    vis.line(X=x, Y=y1, win='sinx', opts={'title':'y=sin(x)'})
    vis.line(X=x, Y=y2, win='cosx', opts={'title':'y=cos(x)'})


