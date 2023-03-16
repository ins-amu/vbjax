import jax


def make_dense_layers(in_dim, latent_dims=[10], init_scl=0.1, extra_in=0,
               act_fn=jax.nn.leaky_relu, key=jax.random.PRNGKey(42)):
    """Make a dense neural network with the given latent layer sizes."""
    small = lambda shape: jax.random.normal(key, shape=shape)*init_scl
    
    weights = [small((latent_dims[0], in_dim + extra_in))]
    biases = [small((latent_dims[0], 1))]
    nlayers = len(latent_dims)
    
    for i in range(nlayers - 1):
        weights.append(small((latent_dims[i+1], latent_dims[i])))
        biases.append(small((latent_dims[i+1], 1)))
        
    weights.append(small((in_dim, latent_dims[-1])))
    biases.append(small((in_dim, 1)))
    
    def fwd(params, x):
        weights, biases = params
        for i in range(nlayers):
            x = act_fn(weights[i] @ x + biases[i])
        return weights[-1] @ x + biases[-1]
    
    return (weights, biases), fwd