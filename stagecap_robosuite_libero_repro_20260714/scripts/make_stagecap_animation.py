
import os, sys, glob, math
import imageio
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.patches import FancyBboxPatch
from PIL import Image, ImageDraw

ROOT = "/workdata/stage_cap_privileged_data_generation_repro_20260629"
SUCCESS_RUN = (
    ROOT + "/outputs/reproduce_stagecap/smoke_libero_20260630_143848"
    "/libero_object/00_pick_up_the_alphabet_soup_and_place_it_in_the_basket"
    "/round_01/mimo-v2.5/run"
    "/trial_01_sandboxrc_0_reward_1.000_taskcompleted_1"
)
FAIL_RUN = (
    ROOT + "/outputs/reproduce_stagecap/smoke_libero_20260630_142439"
    "/libero_object/00_pick_up_the_alphabet_soup_and_place_it_in_the_basket"
    "/round_01/mimo-v2.5/run"
    "/trial_01_sandboxrc_1_reward_0.000_taskcompleted_0"
)
OUT_GIF = ROOT + "/outputs/stagecap_pipeline_animation.gif"

C_BG="#0D1117"; C_BLUE="#4A90D9"; C_PURPLE="#8B5CF6"; C_GREEN="#22C55E"
C_RED="#EF4444"; C_ORANGE="#F97316"; C_WHITE="#F0F6FC"; C_GRAY="#484F58"; C_YELLOW="#FCD34D"
W,H=1280,720; FPS=12; DPIN=100

def hex2rgb(h):
    h=h.lstrip("#")
    return tuple(int(h[j:j+2],16) for j in (0,2,4))

def fig_to_pil(fig):
    fig.canvas.draw()
    arr=np.asarray(fig.canvas.buffer_rgba())
    return Image.fromarray(arr[:,:,:3]).resize((W,H),Image.LANCZOS)

def make_fig():
    fig=plt.figure(figsize=(W/DPIN,H/DPIN),dpi=DPIN)
    fig.patch.set_facecolor(C_BG)
    ax=fig.add_axes([0,0,1,1])
    ax.set_facecolor(C_BG); ax.set_xlim(0,W); ax.set_ylim(0,H); ax.axis("off")
    return fig,ax

def eio(t): return max(0.0,min(1.0,t*t*(3-2*t)))

def add_title(ax,title,color,fs=16):
    ax.text(W/2,H-30,title,color=color,fontsize=fs,ha="center",va="center",
            fontweight="bold",fontfamily="monospace",
            path_effects=[pe.withStroke(linewidth=3,foreground=C_BG)])

def rbox(ax,x,y,w,h,color,alpha=1.0,lw=2,fill=None):
    fc=fill if fill else "none"
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle="round,pad=4",
        linewidth=lw,edgecolor=color,facecolor=fc,alpha=alpha))

def marrow(ax,x1,y1,x2,y2,color,alpha=1.0,lw=2):
    ax.annotate("",xy=(x2,y2),xytext=(x1,y1),
        arrowprops=dict(arrowstyle="-|>",color=color,lw=lw,alpha=alpha,
                        connectionstyle="arc3,rad=0"))

def progbar(ax,stages,cur,alpha=1.0):
    n=len(stages); sw=W/n
    for i,(lbl,_) in enumerate(stages):
        cx=sw*i+sw/2
        done=i<cur; active=i==cur
        c=C_GREEN if done else (C_PURPLE if active else C_GRAY)
        rbox(ax,cx-sw/2+4,8,sw-8,40,color=c,fill=c if done else None,alpha=alpha)
        ax.text(cx,28,lbl,color=C_WHITE if (done or active) else C_GRAY,
                fontsize=9,ha="center",va="center",fontfamily="monospace",alpha=alpha,
                path_effects=[pe.withStroke(linewidth=2,foreground=C_BG)])

STAGES=[("S1: Grounding","目标定位\n读取坐标\n场景快照"),
        ("S2: Pick&Lift","抓取验证\n夹爪接触\n物体离地"),
        ("S3: Transport","运输保护\n规避碰撞\n稳定移动"),
        ("S4: Release","放置验证\n进入目标区\nreward=1.0")]
TASK_INST='"pick up the alphabet soup\n and place it in the basket"'

av_frames=sorted(glob.glob(SUCCESS_RUN+"/trajectory_images/agentview/frame_*.png"))
fail_frames=sorted(glob.glob(FAIL_RUN+"/kite_failure_keyframes/annotated_frames/*.png"))
print(f"agentview={len(av_frames)}, fail={len(fail_frames)}")

all_frames=[]

# ACT 1
N1=24; rf0=Image.open(av_frames[0]).convert("RGB").resize((420,280),Image.LANCZOS)
for i in range(N1):
    t=i/(N1-1); fig,ax=make_fig()
    for gx in range(0,W,80): ax.axvline(gx,color=C_GRAY,lw=0.3,alpha=0.25)
    for gy in range(0,H,80): ax.axhline(gy,color=C_GRAY,lw=0.3,alpha=0.25)
    add_title(ax,"Act 1  |  Input & Scene Sampling",C_BLUE)
    a1=eio(min(1,t*2.5)); a2=eio(max(0,(t-0.3)*2.5)); a3=eio(max(0,(t-0.55)*3))
    if a1>0:
        rbox(ax,30,H/2+55,290,135,C_BLUE,alpha=a1,lw=2)
        ax.text(175,H/2+165,"Task Instruction",color=C_BLUE,fontsize=12,ha="center",
                va="center",fontfamily="monospace",alpha=a1,
                path_effects=[pe.withStroke(linewidth=2,foreground=C_BG)])
        ax.text(175,H/2+110,TASK_INST,color=C_WHITE,fontsize=10,ha="center",
                va="center",fontfamily="monospace",style="italic",alpha=a1,
                path_effects=[pe.withStroke(linewidth=2,foreground=C_BG)])
    if a2>0.02:
        rf_show=rf0.copy()
        ImageDraw.Draw(rf_show).rectangle([0,0,419,279],outline=hex2rgb(C_BLUE),width=3)
        ax.imshow(np.array(rf_show),extent=[W/2-215,W/2+215,H/2-185,H/2+95],
                  aspect="auto",alpha=a2,zorder=5)
        ax.text(W/2,H/2+115,"LIBERO Simulation  |  agentview RGB",color=C_BLUE,
                fontsize=10,ha="center",alpha=a2,fontfamily="monospace",
                path_effects=[pe.withStroke(linewidth=2,foreground=C_BG)])
    if a3>0:
        for j,(k,v,c) in enumerate([("Env","LIBERO Object Suite",C_BLUE),
                                     ("Task","alphabet_soup -> basket",C_YELLOW),
                                     ("Model","MIMO-v2.5",C_PURPLE),
                                     ("Mode","Privileged CaP",C_GREEN)]):
            yy=H/2+185-j*50
            rbox(ax,W-372,yy-22,342,42,c,alpha=a3*0.8,lw=1.5)
            ax.text(W-360,yy,f"{k}:",color=c,fontsize=10,ha="left",va="center",
                    fontfamily="monospace",alpha=a3,
                    path_effects=[pe.withStroke(linewidth=1.5,foreground=C_BG)])
            ax.text(W-308,yy,v,color=C_WHITE,fontsize=10,ha="left",va="center",
                    fontfamily="monospace",alpha=a3,
                    path_effects=[pe.withStroke(linewidth=1.5,foreground=C_BG)])
    progbar(ax,STAGES,-1,alpha=0.4)
    all_frames.append(fig_to_pil(fig)); plt.close(fig)
print(f"Act1 done")

# ACT 2
N2=30; node_xs=[180,440,700,960]; NY=H/2+20; NW,NH=200,100
for i in range(N2):
    t=i/(N2-1); fig,ax=make_fig()
    for gx in range(0,W,80): ax.axvline(gx,color=C_GRAY,lw=0.3,alpha=0.2)
    add_title(ax,"Act 2  |  Stage Planning via MIMO-v2.5",C_PURPLE)
    am=eio(min(1,t*4))
    if am>0:
        rbox(ax,30,H-130,235,70,C_BLUE,alpha=am,lw=2)
        ax.text(148,H-95,"Task Description",color=C_BLUE,fontsize=11,ha="center",
                va="center",fontfamily="monospace",alpha=am,
                path_effects=[pe.withStroke(linewidth=2,foreground=C_BG)])
    ac=eio(max(0,(t-0.1)*4))
    if ac>0:
        rbox(ax,W/2-87,H-140,174,88,C_PURPLE,alpha=ac,fill="#1E0D33",lw=2.5)
        ax.text(W/2,H-95,"MIMO-v2.5",color=C_PURPLE,fontsize=14,ha="center",va="center",
                fontfamily="monospace",fontweight="bold",alpha=ac,
                path_effects=[pe.withStroke(linewidth=2,foreground=C_BG)])
        ax.text(W/2,H-116,"Stage Planner",color=C_WHITE,fontsize=10,ha="center",
                va="center",fontfamily="monospace",alpha=ac*0.8,
                path_effects=[pe.withStroke(linewidth=2,foreground=C_BG)])
        if ac>0.3: marrow(ax,268,H-96,W/2-89,H-96,C_PURPLE,alpha=ac)
    nvis=max(0,min(4,int(t*5.5)))
    for si in range(nvis):
        an=eio(min(1,max(0,(t-si*0.18)*5))); nx=node_xs[si]; lbl,desc=STAGES[si]
        rbox(ax,nx-NW/2,NY-NH/2,NW,NH,C_PURPLE,alpha=an,fill="#1A0D2E",lw=2.5)
        ax.text(nx,NY+18,lbl,color=C_PURPLE,fontsize=11,ha="center",va="center",
                fontfamily="monospace",fontweight="bold",alpha=an,
                path_effects=[pe.withStroke(linewidth=2,foreground=C_BG)])
        for li,line in enumerate(desc.split("\n")):
            ax.text(nx,NY-4-li*16,line,color=C_WHITE,fontsize=9,ha="center",va="center",
                    fontfamily="monospace",alpha=an*0.8,
                    path_effects=[pe.withStroke(linewidth=2,foreground=C_BG)])
        ax.text(nx-NW/2+14,NY+NH/2-12,f"S{si+1}",color=C_YELLOW,fontsize=10,
                ha="center",va="center",fontfamily="monospace",fontweight="bold",alpha=an)
        if si<3 and nvis>si+1:
            aa=eio(min(1,max(0,(t-(si+0.5)*0.18)*5)))
            marrow(ax,nx+NW/2+2,NY,node_xs[si+1]-NW/2-2,NY,C_PURPLE,alpha=aa,lw=2.5)
    if nvis>=1:
        aa0=eio(min(1,max(0,(t-0.18)*5)))
        marrow(ax,W/2+87,H-96,node_xs[0]-NW/2-2,NY,C_PURPLE,alpha=aa0,lw=2)
    if t>0.8:
        ag=eio((t-0.8)*5)
        ax.text(W/2,NY-NH/2-45,"Generated 4 CaP Code Blocks",color=C_GREEN,fontsize=12,
                ha="center",va="center",fontfamily="monospace",alpha=ag,
                path_effects=[pe.withStroke(linewidth=2,foreground=C_BG)])
    progbar(ax,STAGES,-1,alpha=0.5)
    all_frames.append(fig_to_pil(fig)); plt.close(fig)
print(f"Act2 done")

# ACT 3
N3=36; fidx=[int(i*(len(av_frames)-1)/(N3-1)) for i in range(N3)]
sbounds=[0,32,64,96,129]
def gsi(fi):
    for si,(s,e) in enumerate(zip(sbounds,sbounds[1:])):
        if s<=fi<e: return si
    return 3
scols=[C_BLUE,C_PURPLE,C_ORANGE,C_GREEN]
for i in range(N3):
    t=i/(N3-1); rfi=fidx[i]; cs=gsi(rfi); sc=scols[cs]
    fig,ax=make_fig()
    ri=Image.open(av_frames[rfi]).convert("RGB").resize((560,380),Image.LANCZOS)
    ImageDraw.Draw(ri).rectangle([0,0,559,379],outline=hex2rgb(sc),width=4)
    ax.imshow(np.array(ri),extent=[W/2-285,W/2+285,H/2-230,H/2+150],aspect="auto",zorder=5)
    add_title(ax,"Act 3  |  Stage Execution in LIBERO Simulation",C_BLUE)
    slbl,_=STAGES[cs]
    rbox(ax,30,H/2+65,228,56,sc,alpha=0.9,fill="#0D1117",lw=2.5)
    ax.text(144,H/2+94,"Current Stage",color=C_GRAY,fontsize=10,ha="center",va="center",
            fontfamily="monospace",path_effects=[pe.withStroke(linewidth=1.5,foreground=C_BG)])
    ax.text(144,H/2+73,slbl,color=sc,fontsize=12,ha="center",va="center",
            fontfamily="monospace",fontweight="bold",
            path_effects=[pe.withStroke(linewidth=2,foreground=C_BG)])
    priv=[("object_pos","[0.48,-0.24,0.04]"),("contact","True" if cs>=1 else "False"),
          ("reward","1.0" if cs==3 and t>0.9 else "0.0"),
          ("completed","True" if cs==3 and t>0.9 else "False")]
    rbox(ax,W-278,H/2+10,248,158,C_GREEN,alpha=0.6,lw=1.5)
    ax.text(W-154,H/2+150,"Privileged Info",color=C_GREEN,fontsize=10,ha="center",
            va="center",fontfamily="monospace",fontweight="bold",
            path_effects=[pe.withStroke(linewidth=2,foreground=C_BG)])
    for pi,(pk,pv) in enumerate(priv):
        vc=C_GREEN if pv in("True","1.0") else C_WHITE
        ax.text(W-265,H/2+122-pi*32,f"  {pk}:",color=C_GRAY,fontsize=9,ha="left",
                va="center",fontfamily="monospace")
        ax.text(W-265+120,H/2+122-pi*32,pv,color=vc,fontsize=9,ha="left",va="center",
                fontfamily="monospace",fontweight="bold" if pv in("True","1.0") else "normal")
    ax.text(W/2+280,H/2+140,f"frame {rfi:03d}/{len(av_frames)-1:03d}",
            color=C_GRAY,fontsize=9,ha="right",va="center",fontfamily="monospace")
    marrow(ax,W/2+287,H/2-40,W-278,H/2+88,C_BLUE,lw=1.5,alpha=0.7)
    ax.text(W/2+308,H/2+20,"obs ->",color=C_BLUE,fontsize=9,ha="left",va="center",
            fontfamily="monospace",alpha=0.7)
    progbar(ax,STAGES,cs,alpha=0.9)
    all_frames.append(fig_to_pil(fig)); plt.close(fig)
print(f"Act3 done")

# ACT 4
N4=24
if fail_frames:
    fi_img=Image.open(fail_frames[0]).convert("RGB").resize((420,280),Image.LANCZOS)
else:
    fi_img=Image.open(av_frames[40]).convert("RGB").resize((420,280),Image.LANCZOS)
    ov=Image.new("RGB",fi_img.size,(80,0,0)); fi_img=Image.blend(fi_img,ov,0.35)
for i in range(N4):
    t=i/(N4-1); fig,ax=make_fig(); phase=t*2
    add_title(ax,"Act 4  |  Failure Detection & KITE Repair",C_RED)
    af=eio(min(1,phase*2))
    ax.imshow(np.array(fi_img),extent=[115,545,H/2-180,H/2+100],aspect="auto",
              zorder=5,alpha=max(0.05,af))
    blink=0.5+0.4*math.sin(t*math.pi*8) if phase<1 else 0.3
    rbox(ax,110,H/2-185,440,290,C_RED,alpha=blink*af,lw=3)
    if phase>0.3:
        av2=eio(min(1,(phase-0.3)*3))
        rbox(ax,615,H/2+20,240,85,C_RED,alpha=av2,fill="#2D0808",lw=2.5)
        ax.text(735,H/2+68,"Stage Verifier",color=C_RED,fontsize=11,ha="center",
                va="center",fontfamily="monospace",fontweight="bold",alpha=av2,
                path_effects=[pe.withStroke(linewidth=2,foreground=C_BG)])
        ax.text(735,H/2+44,"needs_repair",color=C_RED,fontsize=13,ha="center",
                va="center",fontfamily="monospace",alpha=av2,
                path_effects=[pe.withStroke(linewidth=2,foreground=C_BG)])
        ax.text(735,H/2+22,"pick_lift_verify FAILED",color=C_WHITE,fontsize=9,
                ha="center",va="center",fontfamily="monospace",alpha=av2*0.8)
        marrow(ax,547,H/2-40,613,H/2+56,C_RED,lw=2,alpha=av2)
    if phase>0.7:
        ad=eio(min(1,(phase-0.7)*4))
        rbox(ax,615,H/2-84,240,94,C_ORANGE,alpha=ad,fill="#1A0D00",lw=2)
        ax.text(735,H/2-32,"Failure Diagnosis",color=C_ORANGE,fontsize=10,ha="center",
                va="center",fontfamily="monospace",fontweight="bold",alpha=ad,
                path_effects=[pe.withStroke(linewidth=2,foreground=C_BG)])
        for di,dl in enumerate(["gripper: no contact","object: on table","-> retry grasp"]):
            ax.text(625,H/2-55+di*19,dl,color=C_WHITE,fontsize=8,ha="left",va="center",
                    fontfamily="monospace",alpha=ad*0.85)
    if phase>1.1:
        ar2=eio(min(1,(phase-1.1)*3))
        rbox(ax,615,H/2-200,240,108,C_ORANGE,alpha=ar2,fill="#1A0A00",lw=2)
        ax.text(735,H/2-150,"Local Code Repair",color=C_ORANGE,fontsize=10,ha="center",
                va="center",fontfamily="monospace",fontweight="bold",alpha=ar2,
                path_effects=[pe.withStroke(linewidth=2,foreground=C_BG)])
        ax.text(735,H/2-170,"pi_i  ->  pi_i'",color=C_PURPLE,fontsize=14,ha="center",
                va="center",fontfamily="monospace",alpha=ar2,
                path_effects=[pe.withStroke(linewidth=2,foreground=C_BG)])
        ax.text(735,H/2-192,"adjust_grasp_height(+0.02)",color=C_WHITE,fontsize=8,
                ha="center",va="center",fontfamily="monospace",alpha=ar2*0.8)
    progbar(ax,STAGES,1,alpha=0.8)
    all_frames.append(fig_to_pil(fig)); plt.close(fig)
print(f"Act4 done")

# ACT 5
N5=30
tidx=[int(i*(len(av_frames)-1)/5) for i in range(6)]
thumbs=[Image.open(av_frames[fi]).convert("RGB").resize((160,108),Image.LANCZOS) for fi in tidx]
for i in range(N5):
    t=i/(N5-1); fig,ax=make_fig()
    add_title(ax,"Act 5  |  Success Trajectory -> VLA Training Dataset",C_GREEN,fs=15)
    av3=eio(min(1,t*3)); at=eio(max(0,(t-0.25)*3)); ao=eio(max(0,(t-0.6)*3))
    if av3>0:
        rbox(ax,W/2-188,H/2+95,376,108,C_GREEN,alpha=av3,fill="#052010",lw=3)
        ax.text(W/2,H/2+154,"Stage Verifier",color=C_GREEN,fontsize=13,ha="center",
                va="center",fontfamily="monospace",fontweight="bold",alpha=av3,
                path_effects=[pe.withStroke(linewidth=2,foreground=C_BG)])
        ax.text(W/2,H/2+130,"reward = 1.0   task_completed = True",color=C_GREEN,
                fontsize=11,ha="center",va="center",fontfamily="monospace",alpha=av3,
                path_effects=[pe.withStroke(linewidth=2,foreground=C_BG)])
        ax.text(W/2,H/2+110,"sandbox_rc = 0  (0 retries needed)",color=C_WHITE,
                fontsize=9,ha="center",va="center",fontfamily="monospace",alpha=av3*0.8)
    if at>0:
        ns=max(1,int(at*6)); tw,th=160,108
        total_w=ns*tw+(ns-1)*12; sx=W/2-total_w/2; ty=H/2-10
        for ti in range(ns):
            tx=sx+ti*(tw+12)
            ax.imshow(np.array(thumbs[ti]),extent=[tx,tx+tw,ty-th,ty],
                      aspect="auto",alpha=at,zorder=5)
            rbox(ax,tx-2,ty-th-2,tw+4,th+4,C_GREEN,alpha=at*0.8,lw=2)
            ax.text(tx+tw/2,ty-th-14,f"t={tidx[ti]}",color=C_GRAY,fontsize=8,
                    ha="center",va="center",fontfamily="monospace")
            if ti<ns-1:
                ax.text(tx+tw+4,ty-th/2,"->",color=C_GREEN,fontsize=12,
                        ha="center",va="center",alpha=at)
    if ao>0:
        rbox(ax,W/2-228,62,456,70,C_GREEN,alpha=ao,fill="#052010",lw=3)
        ax.text(W/2,100,"VLA Training Dataset",color=C_GREEN,fontsize=15,ha="center",
                va="center",fontfamily="monospace",fontweight="bold",alpha=ao,
                path_effects=[pe.withStroke(linewidth=2,foreground=C_BG)])
        ax.text(W/2,78,"-> convert_stagecap_steps_to_lerobot.py -> LeRobot Format",
                color=C_WHITE,fontsize=9,ha="center",va="center",
                fontfamily="monospace",alpha=ao*0.8)
        marrow(ax,W/2,H/2-18,W/2,135,C_GREEN,lw=2.5,alpha=ao)
    if ao>0.3:
        ast=eio(max(0,(t-0.75)*4))
        stats=[("Stages Completed","4 / 4"),("Attempts","1 (no repair)"),
               ("Traj Steps",str(len(av_frames))),("Final Reward","1.000")]
        rbox(ax,28,H/2-20,228,138,C_GREEN,alpha=ast*0.6,lw=1.5)
        ax.text(142,H/2+100,"Run Summary",color=C_GREEN,fontsize=10,ha="center",
                va="center",fontfamily="monospace",fontweight="bold",alpha=ast,
                path_effects=[pe.withStroke(linewidth=2,foreground=C_BG)])
        for si,(sk,sv) in enumerate(stats):
            ax.text(42,H/2+78-si*28,f"  {sk}:",color=C_GRAY,fontsize=9,ha="left",
                    va="center",fontfamily="monospace",alpha=ast)
            ax.text(42+150,H/2+78-si*28,sv,color=C_GREEN,fontsize=9,ha="left",
                    va="center",fontfamily="monospace",fontweight="bold",alpha=ast)
    progbar(ax,STAGES,4,alpha=0.9)
    all_frames.append(fig_to_pil(fig)); plt.close(fig)
print(f"Act5 done")

# Output
total=len(all_frames)
print(f"Total: {total} frames, {total/FPS:.1f}s")
print(f"Saving {OUT_GIF} ...")
imageio.mimsave(OUT_GIF,[np.array(f) for f in all_frames],
                format="GIF",duration=int(1000/FPS),loop=0)
print(f"DONE: {OUT_GIF}")
ki=[N1//2,N1+N2//2,N1+N2+N3//2,N1+N2+N3+N4//2,N1+N2+N3+N4+N5//2]
tw2,th2=W//5,H//5
mosaic=Image.new("RGB",(W,th2+32),C_BG)
draw=ImageDraw.Draw(mosaic)
for ci,k in enumerate(ki):
    thumb=all_frames[k].resize((tw2-4,th2-4),Image.LANCZOS)
    mosaic.paste(thumb,(ci*tw2+2,2))
    draw.rectangle([ci*tw2,0,(ci+1)*tw2-1,th2+30],outline=(80,80,80),width=1)
    draw.text((ci*tw2+tw2//2-25,th2+8),f"Act {ci+1}",fill=(200,200,200))
mp=OUT_GIF.replace(".gif","_keyframes_mosaic.png")
mosaic.save(mp)
print(f"DONE mosaic: {mp}")
