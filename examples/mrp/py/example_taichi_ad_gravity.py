import taichi as ti
from iipyper import OSC, run, repeat, cleanup
from math import log
from mrp import MRP

ti.init(arch=ti.gpu)

N = 16
dt = 1e-6

x = ti.Vector.field(2, dtype=ti.f32, shape=N,
                    needs_grad=True)  # particle positions
v = ti.Vector.field(2, dtype=ti.f32, shape=N)  # particle velocities
U = ti.field(dtype=ti.f32, shape=(), needs_grad=True)  # potential energy
r = ti.Vector.field(3, dtype=ti.f32, shape=N)  # resonators

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
    # scale()


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
        r[i][0] = x[i][0] * 1000 + 40 # frequency
        r[i][1] = x[i][1] * 0.5 # gain
        r[i][2] = 0.5 # decay

def main(host="127.0.0.1", receive_port=7563, send_port=7770):

    osc = OSC(host, receive_port, verbose=False)
    osc.create_client("mrp", host="127.0.0.1", port=send_port)
    mrp = MRP(osc)
    notes = [
        24, 29, 31, 
        36, 41, 43, 
        48, 53, 55, 
        60, 65, 67, 
        72, 77, 79, 
        84
    ]
    [mrp.note_on(n) for n in notes]

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
            for i, n in enumerate(notes):
                mrp.qualities_update(n, {
                    'intensity':  _x[i][0],
                    'brightness': _x[i][1],
                    'pitch':      1-_x[i][0]
                    # 'harmonic':   1-_x[i][1],
                })
            # osc.send('/resonators', *_x.astype(float).flatten())
            # osc.send('/resonators', *r.to_numpy().astype(float).flatten())
            frame_count = 0
    mrp.all_notes_off()

    @cleanup
    def _():
        mrp.all_notes_off()

if __name__ == "__main__":
    run(main)
