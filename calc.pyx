import numpy as np
cimport numpy as cnp
import scipy.sparse

class calc:
    def __init__(self, num_gridx, object_startx, object_endx, object_starty, object_endy, num_vx, num_vy):
        #define

        self.delta = num_gridx/512
        self.delta_t = 0.01
        self.Re = 100
        self.object_startx = object_startx
        self.object_endx =  object_endx
        self.object_starty = object_starty
        self.object_endy =  object_endy
        self.num_vx = num_vx
        self.num_vy = num_vy
        self.num_px = self.num_vx+1
        self.num_py = self.num_vy+1
        self.num_Ax, self.num_Ay = self.num_px-2, self.num_py-2

    def ConvectionTerm(self, u, v, flag_v, u_old, v_old):
        for i in range(1, self.num_vy-1):
            for j in range(1, self.num_vx-1):
                if flag_v[i, j] >= 1: continue
                if u_old[i, j] >= 0 and v_old[i, j] >= 0:
                    u[i, j] -= (u_old[i, j] * (u_old[i, j] - u_old[i, j-1]) + v_old[i, j] * (u_old[i, j] - u_old[i-1, j])) * self.delta_t / self.delta
                    v[i, j] -= (u_old[i, j] * (v_old[i, j] - v_old[i, j-1]) + v_old[i, j] * (v_old[i, j] - v_old[i-1, j])) * self.delta_t / self.delta
                elif u_old[i, j] < 0 and v_old[i, j] >= 0:
                    u[i, j] -= (u_old[i, j] * (u_old[i, j+1] - u_old[i, j]) + v_old[i, j] * (u_old[i, j] - u_old[i-1, j])) * self.delta_t / self.delta
                    v[i, j] -= (u_old[i, j] * (v_old[i, j+1] - v_old[i, j]) + v_old[i, j] * (v_old[i, j] - v_old[i-1, j])) * self.delta_t / self.delta
                elif u_old[i, j] >= 0 and v_old[i, j] < 0:
                    u[i, j] -= (u_old[i, j] * (u_old[i, j] - u_old[i, j-1]) + v_old[i, j] * (u_old[i+1, j] - u_old[i, j])) * self.delta_t / self.delta
                    v[i, j] -= (u_old[i, j] * (v_old[i, j] - v_old[i, j-1]) + v_old[i, j] * (v_old[i+1, j] - v_old[i, j])) * self.delta_t / self.delta
                else:
                    u[i, j] -= (u_old[i, j] * (u_old[i, j+1] - u_old[i, j]) + v_old[i, j] * (u_old[i+1, j] - u_old[i, j])) * self.delta_t / self.delta
                    v[i, j] -= (u_old[i, j] * (v_old[i, j+1] - v_old[i, j]) + v_old[i, j] * (v_old[i+1, j] - v_old[i, j])) * self.delta_t / self.delta
        return


    def DiffusionTerm(self, v, v_old, flag_v):
        for i in range(1, self.num_vy-1):
            for j in range(1, self.num_vx-1):
                if flag_v[i, j] >= 1: continue
                v[i, j] += (v_old[i+1, j] + v_old[i-1, j] + v_old[i, j+1] + v_old[i, j-1] - 4*v_old[i, j]) * self.delta_t / (self.delta**2 * self.Re)
        return


    def DiverV(self, s, u, v, flag_v):
        for i in range(1, self.num_py-1):
            for j in range(1, self.num_px-1):
                if flag_v[i, j] >= 3 or flag_v[i-1, j] >= 3 or flag_v[i-1, j-1] >= 3 or flag_v[i, j-1] >= 3:
                    continue
                if flag_v[i, j] == 2:
                    if i == self.num_py-2: u[i, j], v[i, j] = u[i-1, j], v[i-1, j]
                    else:  u[i, j], v[i, j] = u[i, j-1], v[i, j-1]
                if flag_v[i-1, j] == 2:
                    if i-1 == 0: u[i-1, j], v[i-1, j] = u[i, j], v[i, j]
                    else: u[i-1, j], v[i-1, j] = u[i-1, j-1], v[i-1, j-1]
                if flag_v[i, j-1] == 2:
                    if i == self.num_py-2: u[i, j-1], v[i, j-1] = u[i-1, j-1], v[i-1, j-1]
                    else: u[i, j-1], v[i, j-1] = u[i, j], v[i, j]
                if flag_v[i-1, j-1] == 2:
                    if i-1 == 0: u[i-1, j-1], v[i-1, j-1] = u[i, j-1], v[i, j-1]
                    else: u[i-1, j-1], v[i-1, j-1] = u[i-1, j], v[i-1, j]
                s[i, j] = (
                    u[i-1, j] - u[i-1, j-1] + u[i, j] - u[i, j-1] + \
                    v[i, j-1] - v[i-1, j-1] + v[i, j] - v[i-1, j]
                ) * self.delta / (2*self.delta_t)
        return

    def Cholesky(self, p, s, flag_p, factor):
        s = s[1:-1, 1:-1] * -1
        b = s.reshape((-1,))
        x = factor(b)
        x = x.reshape((self.num_Ay, self.num_Ax))
        for i in range(1, self.num_py-1):
            for j in range(1, self.num_px-1):
                if flag_p[i, j] >= 1: continue
                p[i, j] = x[i-1, j-1]
                if flag_p[i+1, j] == 2: p[i+1, j] = x[i-1, j-1]
                if flag_p[i-1, j] == 2: p[i-1, j] = x[i-1, j-1]
                if flag_p[i, j+1] == 2: p[i, j+1] = x[i-1, j-1]
                if flag_p[i, j-1] == 2: p[i, j-1] = x[i-1, j-1]
        return


    def PressureTerm(self, p, u, v, flag_p, flag_v, factor):
        s = np.zeros((self.num_py, self.num_px))
        self.DiverV(s, u, v, flag_v)
        self.Cholesky(p, s, flag_p, factor)
        return


    def UpdateV(self, u, v, p, flag_v):
        for i in range(1, self.num_vy-1):
            for j in range(1, self.num_vx-1):
                if flag_v[i, j] >= 1: continue
                u[i, j] -= (p[i, j+1] - p[i, j] + p[i+1, j+1] - p[i+1, j]) * self.delta_t / (2*self.delta)
                v[i, j] -= (p[i+1, j] - p[i, j] + p[i+1, j+1] - p[i, j+1]) * self.delta_t / (2*self.delta)
        return

    def NablaV(self, u, v, flag_v):
        nablav = 0
        for i in range(1, self.num_py-1):
            for j in range(1, self.num_px-1):
                if flag_v[i, j] >= 3: continue
                elif flag_v[i-1, j] >= 3: continue
                elif flag_v[i-1, j-1] >= 3: continue
                elif flag_v[i, j-1] >= 3: continue
                nablav += u[i-1, j] - u[i-1, j-1] + u[i, j] - u[i, j-1] + \
                    v[i, j-1] - v[i-1, j-1] + v[i, j] - v[i-1, j]
        return nablav

    def initA(self, flag_p):
        N = self.num_Ax * self.num_Ay
        A = np.eye(N) * 4
        for i in range(N):
            if flag_p[i // self.num_Ax + 1, i % self.num_Ax + 1] >= 2:
                A[i, i] = 1
                continue
            Apij = [
                [i, i-self.num_Ax, i // self.num_Ax, i % self.num_Ax + 1],
                [i, i-1, i // self.num_Ax + 1, i % self.num_Ax],
                [i, i+1, i // self.num_Ax + 1, i% self.num_Ax + 2],
                [i, i+self.num_Ax, i // self.num_Ax + 2, i % self.num_Ax + 1]
            ]
            for Ai, Aj, pi, pj in Apij:
                if flag_p[pi, pj] == 2:
                    A[Ai, Ai] -= 1
                if Aj < 0 or Aj >= N:
                    continue
                if flag_p[pi, pj] >= 1 or pi == 0 or pi == self.num_py-1 \
                        or pj == 0 or pj == self.num_px-1:
                    A[Ai, Aj] = 0
                else:
                    A[Ai, Aj] = -1
        A = scipy.sparse.csc_matrix(A)
        return A