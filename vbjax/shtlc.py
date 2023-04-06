"SHT based local coupling functions."

import numpy as np
import scipy.special as sp
try:
    import shtns
except ImportError:
    print('** shtns is not available')


# Grid functions

def make_grid_shtns(lmax, nlat, nlon, D):
    "Create shtns object and grid as in `make_grid`."
    sht = shtns.sht(lmax)
    sht.set_grid(nlat=nlat, nphi=nlon)
    theta = np.r_[:2*np.pi:1j*(nlon + 1)][:-1]
    theta += (theta[1] - theta[0]) /2
    phi = np.r_[:np.pi:1j*(nlat+1)][:-1]
    phi += (phi[1] - phi[0])/2
    gw = np.sin(phi)/phi.size/2
    return sht, phi, theta, gw


def make_lm(lmax: int):
    lm = np.array([(l,m) for m in range(lmax+1) for l in range(lmax+1) if m<=l]).T
    return lm


def make_grid(nlat, nlon):
    "Create grid for SHT, phi latitude, theta longitude."
    points, weights = np.polynomial.legendre.leggauss(nlat)
    phi = np.arccos(-points)
    gw = weights[::-1]
    theta = np.array([i*2*np.pi/nlon for i in range(nlon)])
    return phi, theta, gw


def grid_pairwise_distance(theta, phi):
    "Compute pairwise distances on grid.  Memory intensive for large grids."
    phi1 = phi.astype('f')[:,None]
    phi2 = phi1[:,None,None]
    theta1 = theta.astype('f')
    theta2 = theta1[:,None,None]
    dphi = phi2 - phi1
    dtheta = theta2 - theta1
    ds = 2*np.arcsin(np.sqrt(np.sin(dphi/2)**2 + np.cos(phi1-np.pi/2)*np.cos(phi2-np.pi/2)*np.sin(dtheta/2)**2))
    return ds


def randn_relaxed(sht):
    "For shtns sht return random spatial array captured by sht."
    return sht.synth(sht.analys(np.random.randn(*sht.spat_array().shape)))

# cortex has 0.12 m2 surface, convert to radians on circumference
C_cm = 2*np.pi*np.sqrt(0.12/(4*np.pi)) * 100
rad_to_cm = C_cm / (2 * np.pi)

# Kernel functions

def kernel_diff(D, l):
    "Compute diffusion kernel, l in shtns order."
    return D * l * (l + 1)


def kernel_dist_origin(theta, phi):
    "Compute distance to origin on grid."
    phi1 = phi[:,None]
    theta1 = theta
    theta0 = 0
    phi0 = 0
    dphi = np.abs(phi0 - phi1)
    dtheta = np.abs(theta0 - theta1)
    ds = np.arccos(np.sin(phi1-np.pi/2)*np.sin(phi0-np.pi/2) + np.cos(phi1-np.pi/2)*np.cos(phi0-np.pi/2)*np.cos(dtheta))
    return ds


def kernel_sh_normalized(sht, k):
    K = sht.analys(k)
    auc = np.sum(np.abs(K))
    return K / auc


def kernel_laplace(sht, theta, phi, size):
    "Compute spatial & spectral coefficients for Laplacian kernel."
    ds = kernel_dist_origin(theta, phi)
    return np.exp(-np.abs(ds/size))


def kernel_gaussian(sht, theta, phi, size):
    "Compute spatial & spectral coefficients for Gaussian kernel."
    ds = kernel_dist_origin(theta, phi)
    return np.sqrt(1/(size*np.pi))*np.exp(-ds**2/size)


def kernel_mexican_hat(sht, theta, phi, size):
    "Compute spatial & spectral coefficients for Mexican hat kernel."
    ds = kernel_dist_origin(theta, phi)
    k = 2/(np.sqrt(3*size)*np.pi**0.25)*(1-(ds/size)**2)*np.exp(-(ds**2/(2*size**2)))
    return k


def kernel_conv_prep(sht, k):
    "Prepares evaluated kernel for SHT convolution."
    # Driscoll & Healy 1994, Theorem 1
    K = kernel_sh_normalized(sht, k)
    im0 = np.array([sht.idx(int(_l), 0) for _l in sht.l])
    return 2*np.pi*np.sqrt(4*np.pi/(2*sht.l+1)) * K[im0]


def kernel_estimate_shtns(sht, k, x0):
    "Estimate effective kernel for kernel k with state x0 for shtns object sht."
    # uses finite differences
    f = lambda x: sht.synth(k * sht.analys(x))
    f0 = f(x0)
    dfs = np.zeros(x0.shape+x0.shape)
    for i in range(x0.shape[0]):
        for j in range(x0.shape[1]):
            x1 = x0.copy()
            x1[i,j] += 0.001
            dfs[i,j] = (f(x1) - f0) / 0.001
    gw = sht.gauss_wts()
    gw_ = np.r_[gw, gw[::-1]]
    dfs /= gw_[:,None,None,None]
    return dfs


# SHT implementation

def make_shtdiff_np(lmax, nlat, nlon, D, return_L=False, np=np):
    "Construct SHT diff implementation in plain NumPy."
    # setup grid
    lm = make_lm(lmax)
    phi, _, gw = make_grid(nlat, nlon)
    L = []
    for m in range(lmax):
        l = lm[0,lm[0]>=m]
        fwd = gw[None, :] * sp.sph_harm(m, l[:, None], 0, phi[None, :]).conjugate()
        bwd = sp.sph_harm(m, l[None, :], 0, phi[:, None])
        dll = D * l * (l + 1)
        L.append(bwd.dot(dll[:, None] * fwd))
    L = np.array(L)

    def f(x):
        X = np.fft.rfft(x, axis=1)
        X[:,:lmax] = np.einsum('abc,ca->ba',L,X[:,:lmax])
        X[:,lmax:] = 0.0
        y = np.fft.irfft(X, axis=1).real
        return y

    return (f, L) if return_L else f


def make_shtdiff(nlat, lmax=None, nlon=None, D=0.0004, return_L=False):
    "Construct SHT diff implementation with Jax."

    # imports here keeps it readable
    from numpy.testing import assert_equal
    import jax.numpy as np
    import jax

    # defaults
    nlon = nlon or 2*nlat
    lmax = lmax or nlat - 1

    # do some checking
    assert lmax < nlat
    assert nlon > nlat

    # reuse numpy impl
    _, L = make_shtdiff_np(lmax=lmax, nlat=nlat, nlon=nlon, D=D, return_L=True)
    assert_equal(0, L.imag)
    L = np.array(L.real.astype('f'))

    # create computational kernel closing over the `L` array
    def f(x):
        X = np.fft.rfft(x, axis=1)
        # can't in-place set w/ Jax so hstack padding
        X = np.hstack(
            (np.einsum('abc,ca->ba', L, X[:,:lmax]),
             np.zeros((X.shape[0], X.shape[1] - lmax), np.complex64)
            )
        )
        y = np.real(np.fft.irfft(X, axis=1))
        return y

    return (f, L) if return_L else f
