from typing import Callable
import jax
import jax.numpy as jnp
from flax import linen as nn

def get_masks(input_dim, hidden_dim=64, num_hidden=1):
    masks = []
    input_degrees = jnp.arange(input_dim)
    degrees = [input_degrees]

    for n_h in range(num_hidden + 1):
        degrees += [jnp.arange(hidden_dim) % (input_dim - 1)]
    degrees += [input_degrees % input_dim - 1]

    for (d0, d1) in zip(degrees[:-1], degrees[1:]):
        masks += [jnp.transpose(jnp.expand_dims(d1, -1) >= jnp.expand_dims(d0, 0)).astype(jnp.float32)]
    return masks

class MaskedDense(nn.Module):
    n_dim: int
    n_hidden: int
    kernel_init: Callable = nn.initializers.lecun_normal()
    bias_init: Callable = nn.initializers.zeros

    @nn.compact
    def __call__(self, x, mask):
        weight = self.param('weights', self.kernel_init, (self.n_dim, self.n_hidden))
        bias = self.param('bias', self.bias_init, (self.n_hidden,))
        return jnp.dot(x, weight * mask) + bias

class MaskedAutoEncoder(nn.Module):
    n_dim: int
    n_hidden: int

    def setup(self):
        self.mask = get_masks(self.n_dim, self.n_hidden)
        self.up = MaskedDense(self.n_dim, self.n_hidden)
        self.mid = MaskedDense(self.n_hidden, self.n_hidden)
        self.down = MaskedDense(self.n_hidden, 2*self.n_dim)

    def __call__(self, inputs):
        log_weight, bias = self.forward(inputs)
        outputs = (inputs - bias)*jnp.exp(-log_weight)
        log_jacobian = -jnp.sum(log_weight, axis=-1)
        return outputs, log_jacobian

    def forward(self, inputs):
        x = self.up(inputs, self.mask[0])
        x = nn.swish(x)
        x = self.mid(x, self.mask[1])
        x = nn.swish(x)
        log_weight, bias = self.down(x, self.mask[2].tile(2)).split(2, -1)
        return log_weight, bias

    def inverse(self, inputs):
        outputs = jnp.zeros_like(inputs)
        for i_col in range(inputs.shape[1]):
            log_weight, bias = self.forward(outputs)
            outputs = jax.ops.index_update(
                outputs, jax.ops.index[:, i_col], inputs[:, i_col] * jnp.exp(log_weight[:, i_col]) + bias[:, i_col]
            )
        log_det_jacobian = -log_weight.sum(-1)
        return outputs, log_det_jacobian

class MaskedAutoregressiveFlow(nn.Module):
    n_dim: int
    n_hidden: int
    n_layer: int

    def setup(self):
        self.layers = [MaskedAutoEncoder(self.n_dim, self.n_hidden) for _ in range(self.n_layer)]
    
    def __call__(self, inputs):
        log_jacobian = 0
        for layer in self.layers:
            inputs, log_jacobian_ = layer(inputs)
            inputs = inputs[:,::-1]
            log_jacobian += log_jacobian_
        return inputs, log_jacobian

    def inverse(self, inputs):
        # Be careful about flipping the inputs when inverting the flow.
        log_jacobian = 0
        for layer in reversed(self.layers):
            inputs, log_jacobian_ = layer.inverse(inputs)
            inputs = inputs[:,::-1]
            log_jacobian += log_jacobian_
        return inputs, log_jacobian
    
    def sample(self, rng_key, n_samples, params):
        mean = jnp.zeros((n_samples,self.n_dim))
        cov = jnp.repeat(jnp.eye(self.n_dim)[None,:],n_samples,axis=0)
        gaussian = jax.random.multivariate_normal(rng_key, mean, cov)
        samples = self.apply({'params': params},gaussian,method=self.inverse)
        return samples