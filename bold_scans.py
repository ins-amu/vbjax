import jax, jax.numpy as jp, vbjax as vb

def make_run2(total_time, dt=0.01, TR=1800.):

    def network(x, p):
        c = p['k']*x.sum(axis=1)
        return vb.mpr_dfun(x, c, p['mp'])

    def g(x, p):
        return p['sig']

    step, _ = vb.make_sde(dt=dt, dfun=network, gfun=g, adhoc=vb.mpr_r_positive)
    
    nta = int(1. / dt)  # temporal average
    ts = jp.r_[:nta]
    
    ntb = int(TR / (nta * dt))  # repetition time bold
    tsb = jp.r_[:ntb]

    ntrs = int(total_time / (ntb*nta*dt))
    tstr = jp.r_[:ntrs]
    
    def scan(f, init, xs, length=None):
        if xs is None:
            xs = [None] * length
        carry = init
        ys = []
        for x in xs:
            carry, y = f(carry, x)
            ys.append(y)
        return carry, jp.stack(ys)

    scan = jax.lax.scan

    def run_bold(theta, x0, key):
        k, sig = theta
        p = {'k': k, 'mp': vb.mpr_default_theta, 'sig': sig}
        ta_buf, ta_update, ta_sample = vb.make_timeavg(x0.shape)
        bold_buf, bold_update, bold_sample = vb.make_timeavg(x0[0].shape) #, dt*nta, vb.bold_default_theta)

        state = {
            'key': key,
            'x': x0,
            'tavg': ta_buf,
            'bold': bold_buf,
        }

        def step_tavg(state, i):
            key1, state['key'] = jax.random.split(state['key'])
            z_t = vb.randn(2, 32, key=key1)
            state['x'] = step(state['x'], z_t, p)
            state['tavg'] = ta_update(state['tavg'], state['x'])
            return state, 0
            
        def step_bold(state, i):
            state, _ = scan(step_tavg, state, ts)
            state['tavg'], ta_value = ta_sample(state['tavg'])
            state['bold'] = bold_update(state['bold'], ta_value[0])
            return state, ta_value

        def step_bold_decimate(state, i):
            state, ta_value = scan(step_bold, state, tsb)
            state['bold'], bold_val = bold_sample(state['bold'])
            return state, bold_val # (bold_val, ta_value[-1], state['x'])

        state, vals = scan(step_bold_decimate, state, tstr)
        return vals
        
    return run_bold

jax.config.update("jax_debug_nans", False)

key = jax.random.PRNGKey(42)
run2 = make_run2(5000.0, TR=1800.0)
run2 = jax.jit(run2)

x0 = jp.c_[0.1, -2.0].T + jp.zeros((2, 32))

# yb, yt, yr = run2((0.,0.), x0, key)
yb = run2((0.,0.), x0, key)
yb.shape
