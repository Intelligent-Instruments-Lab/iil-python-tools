import taichi as ti
from iipyper import OSC, run, repeat
from math import log

ti.init(arch=ti.gpu)

N = 20
dt = 1e-5

x = ti.Vector.field(2, dtype=ti.f32, shape=N,
                    needs_grad=True)  # particle positions
v = ti.Vector.field(2, dtype=ti.f32, shape=N)  # particle velocities
U = ti.field(dtype=ti.f32, shape=(), needs_grad=True)  # potential energy
s = ti.Vector.field(2, dtype=ti.f32, shape=N)  # sinebank

@ti.kernel
def compute_U():
    for i, j in ti.ndrange(N, N):
        r = x[i] - x[j]
        # r.norm(1e-3) is equivalent to ti.sqrt(r.norm()**2 + 1e-3)
        # This is to prevent 1/0 error which can cause wrong derivative
        U[None] += -1 / r.norm(1e-3)  # U += -1 / |r|


@ti.kernel
def advance():
    for i in x:
        v[i] += dt * -x.grad[i]  # dv/dt = -dU/dx
    for i in x:
        x[i] += dt * v[i]  # dx/dt = v


def substep():
    with ti.ad.Tape(loss=U):
        # Kernel invocations in this scope will later contribute to partial derivatives of
        # U with respect to input variables such as x.
        compute_U(
        )  # The tape will automatically compute dU/dx and save the results in x.grad
    advance()
    scale()


@ti.kernel
def init():
    for i in x:
        x[i] = [ti.random(), ti.random()]

@ti.kernel
def scale():
    for i in x:
        # logratio = ti.log(40) / ti.log(20000)
        # power = ((logratio-1.) * x[i][0])+1.
        # r[i][0] = ti.pow(40, power)#40**power # frequency
        s[i][0] = x[i][0] * 100 + 40 # frequency
        s[i][1] = x[i][1] * 0.05 + 0.001 # gain

def main(host="192.168.7.1", receive_port=7563, send_port=7562):
    osc = OSC(host, receive_port, verbose=False)
    osc.create_client("bela", host="192.168.7.2", port=send_port)

    init()
    gui = ti.GUI('Autodiff gravity')
    osc_send_every = 1
    frame_count = 0
    while gui.running:
        for i in range(50):
            substep()
        _x = x.to_numpy()
        gui.circles(_x, radius=3)
        gui.show()
        frame_count+=1
        if frame_count == osc_send_every:
            osc.send('/sinebank', *s.to_numpy().astype(float).flatten())
            frame_count = 0

if __name__ == "__main__":
    run(main)
