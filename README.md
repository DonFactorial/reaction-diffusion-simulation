# reaction-diffusion-simulation
This code solves a reaction-diffusion system with periodic and Dirichlet boundary conditions. Reaction-diffusion systems describe the concentration of a chemical substance at each point in space and time as they diffuse and react with each other. Here, we have two substances, $U$ and $V$, represented by the equations
```math
U_t=\lambda(A)U-\omega(A)V+D_1\nabla^2U
```

```math
V_t=\omega(A)U+\lambda(A)V+D_2\nabla^2V
```

with $\lambda(A)=1-A^2$ and $\omega(A)=-\beta A^2$. The equations are numerically solved with periodic boundary conditions via Fast Fourier Transform and with Dirichlet boundary conditions via Chebyshev polynomials (which can be parameterized with $\cos$ to become similar to a Fourier Cosine Transform).

With a one-armed spiral initial condition and focusing only on $U$ (the visualization for $V$ is similar), we get nice-looking animations like this:
![Reaction-diffusion visualization with one-armed spiral initial condition](https://github.com/DonFactorial/reaction-diffusion-simulation/blob/main/ReactionDiffusion_stable.gif?raw=true)

By instead taking $\beta=2$, we can get more interesting behavior like this:
![Reaction-diffusion visualization with one-armed spiral initial condition and unstable behavior](https://github.com/DonFactorial/reaction-diffusion-simulation/blob/main/ReactionDiffusion_unstable.gif?raw=true)

We can also change the initial condition to a two-armed spiral:
![Reaction-diffusion visualization with two-armed spiral initial condition and unstable behavior](https://github.com/DonFactorial/reaction-diffusion-simulation/blob/main/ReactionDiffusion_unstable2.gif?raw=true)
