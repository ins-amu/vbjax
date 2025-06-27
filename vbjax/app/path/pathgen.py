import tqdm
import numpy as np
import pylab as pl
import jax, jax.numpy as jp

def loss(c):
    v1 = c[:-2] - c[1:-1]
    v2 = c[2:] - c[1:-1]
    nv1 = jp.linalg.norm(v1, axis=1)
    nv2 = jp.linalg.norm(v2, axis=1)
    n12 = nv1 * nv2 + 1e-6
    dv = (v1 * v2).sum(axis=1) / n12
    return (dv).sum()

def opt(c0):
    c = jp.array(c0)
    vgloss = jax.jit(jax.value_and_grad(loss))
    for i in range(20):
        v, g = vgloss(c)
        c -= 0.01 * g
    return c # np.array(c)

def make_snake():
    num_anchors = 3
    num_subgrid = 3
    ax, ay = axy = np.random.rand(2, num_anchors)
    bx = np.random.rand(num_subgrid, num_anchors - 1)
    cxy = np.diff(axy, axis=-1).T * bx[..., None] + axy.T[:-1]
    cxy = np.concatenate([axy.T[:-1][None], cxy], axis=0)
    cx, cy = cxy = np.append(cxy.reshape(-1, 2).T, axy[:, -1:], axis=1)
    c = cxy.T
    ox, oy = opt(cxy.T).T
    return ox, oy

def make_snakes(n):
    num_anchors = 2
    num_subgrid = 3
    axy = np.random.rand(n, 2, num_anchors)
    bx = np.random.rand(n, num_subgrid, num_anchors - 1)
    def setup(axy, bx):
        import jax.numpy as np
        cxy = np.diff(axy, axis=-1).T * bx[..., None] + axy.T[:-1]
        cxy = np.concatenate([axy.T[:-1][None], cxy], axis=0)
        cx, cy = cxy = np.append(cxy.reshape(-1, 2).T, axy[:, -1:], axis=1)
        c = cxy.T
        return opt(cxy.T).T
    oxy = jax.vmap(setup)(axy, bx)
    return np.array(oxy)

import numpy as np

def mpl_to_numpy_agg(fig):
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    canvas = FigureCanvasAgg(fig)
    canvas.draw()
    buf_rgba = canvas.buffer_rgba()
    width, height = fig.canvas.get_width_height()
    image_array = np.asarray(buf_rgba).reshape(height, width, 4) # RGBA
    return image_array


if __name__ == "__main__":
    # np.random.seed(46)
    pl.rcParams["lines.dashed_pattern"] = [5, 4]
    lopts = {'linewidth': 2.0, 'antialiased': False}  # , 'dashes': [5,3,1]}

    num_im = 32768
    oxy = make_snakes(num_im*2)
    ims = []
    masks = []
    for i in tqdm.trange(num_im):
        fig = pl.figure(figsize=(2, 2), dpi=32, frameon=False)
        ox, oy = oxy[i*2]
        mx, my = oxy[i*2+1]
        pl.plot(ox, oy, 'k--', **lopts)
        pl.plot(ox[[0, -1]], oy[[0, -1]], 'ko', markersize=20, **lopts)
        mask = np.random.rand() < 0.5
        masks.append(mask)
        if mask:
            i_mask = ox.size // 2
            pl.plot(ox[i_mask], oy[i_mask], 'wo', markersize=60, **lopts)
        # pl.plot(mx, my, 'k--', **lopts)
        pl.gca().set_axis_off()
        pl.tight_layout()
        im = mpl_to_numpy_agg(fig)
        ims.append(im[..., 0])
        pl.close(fig)
        # pl.savefig(f'/tmp/path-{int(mask)}-{i:04d}.png')

    ims = np.array(ims)
    masks = np.array(masks)
    np.save('ims.npy', ims)
    np.save('masks.npy', masks)