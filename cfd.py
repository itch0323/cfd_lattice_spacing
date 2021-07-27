import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import argparse
from sksparse.cholmod import cholesky
from calc import *

def main():
    #init boundary
    parser = argparse.ArgumentParser(description='cavity')
    parser.add_argument('mesh', type=int, default=64)

    args = parser.parse_args()
    #u, v, p
    #define
    num_gridx = int(args.mesh)
    num_gridy = int(args.mesh)
    num_vx = num_gridx
    num_vy = num_gridy
    num_px = num_vx+1
    num_py = num_vy+1
    u_value = 0.98
    v_value = 0.02
    object_sizex = int(num_gridx/16)
    object_sizey = int(num_gridy/8)
    object_posix = int(args.mesh/8)
    object_posiy = int(args.mesh/2)
    object_startx = object_posix - object_sizex // 2 + 1
    object_endx =  object_startx + object_sizex
    object_starty = object_posiy - object_sizey // 2 + 1
    object_endy =  object_starty + object_sizey
    time_step = 20000
    plot_interbal = 2000

    u = np.zeros((num_vy, num_vx))
    u[0,:], u[:,0], u[-1,:] = u_value, u_value, u_value
    v = np.zeros((num_vy, num_vx))
    v[0,:], v[:,0], v[-1,:] = v_value, v_value, v_value
    p = np.zeros((num_py, num_px))
    #flag_v, flag_p
    flag_v = np.zeros((num_vy, num_vx))
    flag_v[:, -1] = 2
    flag_v[0, :], flag_v[-1, :], flag_v[:, 0] = 1, 1, 1
    flag_p = np.zeros((num_py, num_px))
    flag_p[:, -1] = 2
    flag_p[0, :], flag_p[-1:, :], flag_p[:, 0] = 1, 1, 1
    #init object
    #u, v, p
    #flag_v, flag_p
    flag_v[object_starty:object_endy, object_startx:object_endx] = 1
    flag_v[object_starty+1:object_endy-1, object_startx+1:object_endx-1] = 3
    flag_p[object_starty+1:object_endy, object_startx+1:object_endx] = 2
    flag_p[object_starty+2:object_endy-1, object_startx+2:object_endx-1] = 3

    c = calc(num_gridx, object_startx, object_endx, object_starty, object_endy, num_vx, num_vy)
    #create A
    A = c.initA(flag_p)
    factor = cholesky(A)

    print(f"t=0 divergence v: {c.NablaV(u, v, flag_v)}")
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111)
    img = ax.imshow(np.sqrt(u*u + v*v), vmin=0, vmax=1.3, cmap="viridis")
    ax.set_title(f"t=0")

    def run_simulate(t):
        u_old = u.copy()
        v_old = v.copy()
        c.ConvectionTerm(u, v, flag_v, u_old, v_old)
        c.DiffusionTerm(u, u_old, flag_v)
        c.DiffusionTerm(v, v_old, flag_v)
        c.PressureTerm(p, u, v, flag_p, flag_v, factor)
        c.UpdateV(u, v, p, flag_v)
        print(f"t={t+1} divergence v: {c.NablaV(u, v, flag_v)}")
        if (t+1) % plot_interbal == 0:
            normv = np.sqrt(u*u + v*v)
            plt.cla()
            img = ax.imshow(np.sqrt(u*u + v*v), vmin=0, vmax=1.3, cmap="viridis")
            ax.set_title(f"t={t+1}")

            fig.savefig("./plot/%d_%d.png" %(args.mesh, t))
    ani = animation.FuncAnimation(fig, run_simulate, interval=1, frames=time_step, repeat=False)
    #ani.save("out.mp4", writer="ffmpeg")
    plt.show()


if __name__ == '__main__':
    main()

"""
not cython
160168382 function calls (157222198 primitive calls) in 380.648 seconds
ncalls  tottime  percall  cumtime  percall filename:lineno(function)
    2001   53.168    0.027   53.168    0.027 calc.py:49(DiverV)
    2001   53.035    0.027   53.035    0.027 calc.py:22(ConvectionTerm)
    4002   48.034    0.012   48.034    0.012 calc.py:41(DiffusionTerm)
    2002   34.963    0.017   34.963    0.017 calc.py:103(NablaV)
    2001   34.169    0.017   34.169    0.017 calc.py:95(UpdateV)
    2001   20.806    0.010   20.834    0.010 calc.py:72(Cholesky)

cython
160141726 function calls (157195527 primitive calls) in 353.306 seconds

ncalls  tottime  percall  cumtime  percall filename:lineno(function)
    2001  214.241    0.107  262.213    0.131 cfd.py:61(run_simulate)
    3701097    7.723    0.000    7.723    0.000 {built-in method numpy.array}
    1    6.862    6.862  352.567  352.567 {built-in method exec_}
    4002    5.618    0.001    5.786    0.001 {built-in method matplotlib._image.resample}
    8841493/7466597    3.735    0.000    6.587    0.000 artist.py:217(stale)
    3002    3.457    0.001    3.457    0.001 {method 'take' of 'numpy.ndarray' objects}

"""