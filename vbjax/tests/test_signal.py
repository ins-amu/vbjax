import numpy as np
import scipy.signal
import jax.numpy as jnp
import vbjax

def test_hilbert():
    # Test case 1: Simple sine wave
    t = np.linspace(0, 10, 100)
    x = np.sin(t)

    # Expected result from scipy.signal.hilbert
    expected = scipy.signal.hilbert(x)

    # Result from vbjax.hilbert
    result = vbjax.hilbert(x)

    # Check if results are close
    np.testing.assert_allclose(result, expected, atol=1e-5)

    # Test case 2: N parameter
    N = 120
    expected_padded = scipy.signal.hilbert(x, N=N)
    result_padded = vbjax.hilbert(x, N=N)
    np.testing.assert_allclose(result_padded, expected_padded, atol=1e-5)

    # Test case 3: 1D check
    try:
        vbjax.hilbert(np.zeros((10, 10)))
        assert False, "Should raise NotImplementedError for ndim > 1"
    except NotImplementedError:
        pass

    # Test case 4: Real input check
    try:
        vbjax.hilbert(np.array([1j, 2j]))
        assert False, "Should raise ValueError for complex input"
    except ValueError:
        pass

if __name__ == "__main__":
    test_hilbert()
    print("All tests passed!")
