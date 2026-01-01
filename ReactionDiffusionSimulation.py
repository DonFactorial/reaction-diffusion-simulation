import numpy as np
from scipy.integrate import solve_ivp
from numpy.fft import fft2, ifft2
from math import pi
import time
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import imageio

# The below method isn't mine. It's a translated version of MATLAB code provided in 
# page 266-267 of Data-Driven Modeling & Scientific Computation by J. Nathan Kutz.
# For a given number of points, N, this method returns the points in the domain, x,
# and a derivative operator, D, for Chebyshev polynomials.
def cheb(N):
    if N == 0:
        return np.array([[0.]]), np.array([1.])
    x = np.cos(pi * np.arange(0, N + 1) / N).reshape(-1, 1)  # column
    c = np.hstack(([2.], np.ones(N - 1), [2.])) * ((-1) ** np.arange(0, N + 1))
    X = np.tile(x, (1, N + 1))
    dX = X - X.T
    C = c.reshape(-1, 1) @ (1.0 / c).reshape(1, -1)
    D = C / (dX + np.eye(N + 1))
    D = D - np.diag(np.sum(D, axis=1))
    return D, x.flatten()

beta = 1
D_1 = 0.1
D_2 = 0.1
n_fft = 64
L = 20
tspan = np.arange(0,90+0.5,0.5)

# FFT method
x = np.linspace(-L/2,L/2,n_fft+1)
x = x[:n_fft]
y=x
X_fft, Y_fft = np.meshgrid(x, y, indexing='xy')

m=1 # number of spirals
u=np.tanh(np.sqrt(X_fft**2+Y_fft**2))*np.cos(m*np.angle(X_fft+1j*Y_fft)-(np.sqrt(X_fft**2+Y_fft**2)))
v=np.tanh(np.sqrt(X_fft**2+Y_fft**2))*np.sin(m*np.angle(X_fft+1j*Y_fft)-(np.sqrt(X_fft**2+Y_fft**2)))
uf = np.fft.fft2(u)
vf = np.fft.fft2(v)
ufvec = uf.reshape(n_fft**2, order='F')
vfvec = vf.reshape(n_fft**2, order='F')
ufvfvec0 = np.concatenate((np.real(ufvec), np.imag(ufvec), np.real(vfvec), np.imag(vfvec)))

kx = (2*np.pi/L)*np.concatenate((np.arange(0, n_fft//2), np.arange(-n_fft//2, 0)))
kx[0] = 10**-6
ky = kx
KX, KY = np.meshgrid(kx, ky, indexing='xy')
K = KX**2 + KY**2

def fftrhs(t, ufvfvec, beta, D_1, D_2, K, n):
    Re_ut = ufvfvec[0:n**2]
    Im_ut = ufvfvec[n**2:2 * n**2]
    Re_vt = ufvfvec[2 * n**2:3 * n**2]
    Im_vt = ufvfvec[3 * n**2:4 * n**2]

    ut = (Re_ut + 1j * Im_ut).reshape((n, n), order='F')
    vt = (Re_vt + 1j * Im_vt).reshape((n, n), order='F')

    u = np.real(ifft2(ut))
    v = np.real(ifft2(vt))

    A2 = u ** 2 + v ** 2
    lam = 1 - A2
    omega = -beta * A2

    dudt_hat = fft2(lam * u - omega * v) - (K * D_1) * ut
    dvdt_hat = fft2(omega * u + lam * v) - (K * D_2) * vt

    dudt_flat = dudt_hat.reshape(n * n, order='F')
    dvdt_flat = dvdt_hat.reshape(n * n, order='F')
    return np.concatenate((np.real(dudt_flat), np.imag(dudt_flat), np.real(dvdt_flat), np.imag(dvdt_flat)))

sol_fft = solve_ivp(fftrhs, (tspan[0], tspan[-1]), y0=ufvfvec0, t_eval=tspan, args=(beta, D_1, D_2, K, n_fft))

# Chebyshev Polynomials
n_cheb = 46
D, x = cheb(n_cheb-1)
D = D/(L/2)
D2 = D@D
D2[0,:] = np.zeros(n_cheb)
D2[n_cheb-1,:] = np.zeros(n_cheb)

x = (L/2)*x
y=x
X_cheb,Y_cheb = np.meshgrid(x,y,indexing='xy')
u=np.tanh(np.sqrt(X_cheb**2+Y_cheb**2))*np.cos(m*np.angle(X_cheb+1j*Y_cheb)-(np.sqrt(X_cheb**2+Y_cheb**2)))
v=np.tanh(np.sqrt(X_cheb**2+Y_cheb**2))*np.sin(m*np.angle(X_cheb+1j*Y_cheb)-(np.sqrt(X_cheb**2+Y_cheb**2)))
uvec = u.reshape(n_cheb**2, order='F')
vvec = v.reshape(n_cheb**2, order='F')
uvvec0 = np.concatenate((uvec, vvec))

I = np.eye(D2.shape[0])
Lap = np.kron(D2,I) + np.kron(I,D2)

def chebrhs(t, uvvec, beta, D_1, D_2, Lap, n):
    uvec = uvvec[0:n**2]
    vvec = uvvec[n**2:]
    A2 = uvec ** 2 + vvec ** 2
    lam = 1 - A2
    omega = -beta * A2
    ut = lam*uvec-omega*vvec+D_1*(Lap@uvec)
    vt = omega*uvec+lam*vvec+D_2*(Lap@vvec)
    return np.concatenate((ut,vt))

sol_cheb = solve_ivp(chebrhs, (tspan[0], tspan[-1]), y0=uvvec0, t_eval=tspan, args=(beta, D_1, D_2, Lap, n_cheb))

# Plot solutions
fig, axs = plt.subplots(1,2)
frames = []
for i in range(tspan.size):
    s = sol_fft.y.shape[0]
    u_fft_anim = np.real(np.fft.ifft2((sol_fft.y[:s//4,i]+1j*sol_fft.y[s//4:s//2,i]).reshape((n_fft,n_fft), order='F')))
    axs[0].pcolormesh(X_fft,Y_fft,u_fft_anim)
    s2 = sol_cheb.y.shape[0]
    u_cheb_anim = sol_cheb.y[:s2//2,i].reshape((n_cheb,n_cheb), order='F')
    axs[1].pcolormesh(X_cheb,Y_cheb,u_cheb_anim)
    axs[0].axis('off')
    axs[1].axis('off')
    axs[0].set_title('Periodic B.C.')
    axs[1].set_title('Dirichlet B.C.')
    fig.suptitle(f'$U$ with $\\beta={beta}$, $D_1={D_1}$, $D_2={D_2}$')
    fig.tight_layout()
    fig.savefig('frame.png', dpi=100)
    plt.close()
    frames.append(imageio.v2.imread('frame.png'))
        
imageio.mimsave('ReactionDiffusion.gif', frames, duration=0.5, loop=0)