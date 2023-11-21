import numpy as np
import pylab as pl
import tqdm
import jax, jax.numpy as jp, jax.example_libraries.optimizers as jopt
import vbjax as vb

# load data
SC = np.loadtxt('/Users/duke/Downloads/hcp-data/connectomes/'
                '100-Schaefer17Networks/1StructuralConnectivity/000/Counts.csv')
SC = np.log(1+SC)
FC = np.loadtxt('/Users/duke/Downloads/hcp-data/connectomes/'
                '100-Schaefer17Networks/2FunctionalConnectivity/000/EmpCorrFC_REST1-LR.csv')
pl.figure()
pl.subplot(121); pl.imshow(SC)
pl.subplot(122); pl.imshow(FC); pl.colorbar()
pl.suptitle('HCP data')

# make network model
def net(rv, theta):
    r, v = rv
    k, mpr_theta, ec = theta
    cr = k * ec @ r
    return vb.mpr_dfun(rv, (cr, 0), mpr_theta)

# do a short simulation
dt = 0.1 # ms
ntime = int(1e2 / dt)
rv0 = jp.zeros((2, SC.shape[0]))
_, loop = vb.make_sde(dt=dt, dfun=net, gfun=1e-3)
noise = vb.randn(ntime, *rv0.shape)
theta = 1e-2, vb.mpr_default_theta._replace(eta=-4.5), SC
rv_t = loop(rv0, noise, theta)

# show it
pl.figure()
pl.subplot(221); pl.plot(rv_t[::10,0], 'k')
pl.subplot(222); pl.imshow( np.corrcoef(rv_t[100:,0].T) ); pl.colorbar()

# apply a bold monitor
# ok this api is not so convenient yet, but monitors are online,
# so there's a step function to consume some states at every step
# and a sample function to "read" a sample out of the monitor buffer
bold_buf, bold_step, bold_sample = vb.make_bold(shape=(SC.shape[0],),
                                                dt=dt, p=vb.bold_default_theta)
# we make it "offline" to process an array of states at once
bold_sample = vb.make_offline(step_fn=bold_step, sample_fn=bold_sample)
# now window the MPR r variable into windows 100 time points
windowed_r = rv_t[:,0].reshape((-1, 100, SC.shape[0]))
# scan that array of windows w/ bold
bold_buf, bold = jax.lax.scan(bold_sample, bold_buf, windowed_r)
# now show it
pl.subplot(223); pl.plot(bold, 'k')
pl.subplot(224); pl.imshow( np.corrcoef(bold.T) )
pl.suptitle('simulated time series & FC')

# for optimization we write a loss function
def run_sim_for_params(opt_params):
    k, eta, ec = opt_params
    theta = k, vb.mpr_default_theta._replace(eta=eta), ec
    rv_t = loop(rv0, noise, theta)
    windowed_r = rv_t[:, 0].reshape((-1, 100, SC.shape[0]))
    bold_buf, _, _ = vb.make_bold(
        shape=(SC.shape[0],), dt=dt, p=vb.bold_default_theta)
    _, bold = jax.lax.scan(bold_sample, bold_buf, windowed_r)
    sim_fc = jp.corrcoef(bold.T)
    return sim_fc
def loss(opt_params):
    sim_fc = run_sim_for_params(opt_params)
    # the empirical FC has a zero diagonal (?)
    sim_fc = jp.fill_diagonal(sim_fc, 0, inplace=False)
    return jp.sum(jp.square(sim_fc - FC))

# and do some optimization steps
params = 1e-2, -4.0 * np.ones(SC.shape[0]), SC
print('initial loss', loss(params))  # good to check this is finite!
# choose learning rate / optimizer step size
lr = 1e-3
# create the optimizer (it returns 3 functions)
opt_init, opt_step, opt_get = jopt.adam(lr)
# create optimizer state
opt = opt_init(params)
# compile the loss function w/ gradients
vgloss = jax.jit(jax.value_and_grad(loss))
# do some steps
trace_loss = []
for i in (pbar := tqdm.trange(501)):
    v, g = vgloss(opt_get(opt))
    trace_loss.append(v)
    g = jax.tree_map(lambda g: jp.clip(g, -1, 1), g) # clip stablizes optimization
    opt = opt_step(i, g, opt)
    pbar.set_description(f'loss {v:0.3f}')
k_final, eta_final, ec_final = params_final = opt_get(opt)
pl.figure(); pl.plot(trace_loss); pl.title('trace loss')

# now show the opt sim fc found
sim_fc_final = run_sim_for_params(params_final)
pl.figure()
pl.subplot(221); pl.imshow(sim_fc_final); pl.title('sim_fc_final')
pl.subplot(222); pl.imshow(FC); pl.title('FC')
pl.subplot(223); pl.imshow(ec_final); pl.title('ec_final')
pl.subplot(223); pl.imshow(SC); pl.title('SC')
pl.title(f'k={k_final:0.3g} eta={eta_final.mean():0.3f}')

pl.show()