from flowMC.resource.base import Resource
import optax
import equinox as eqx


class Optimizer(Resource):
    optim: optax.GradientTransformation
    optim_state: optax.OptState

    def __repr__(self):
        return "Optimizer"

    def __init__(
        self,
        model: eqx.Module,
        learning_rate: float = 1e-3,
        momentum: float = 0.9,
    ):
        self.optim = optax.chain(
            optax.clip_by_global_norm(1.0),
            optax.adamw(learning_rate=learning_rate, b1=momentum),
        )
        self.optim_state = self.optim.init(eqx.filter(model, eqx.is_array))

    def __call__(self, params, grads):
        raise NotImplementedError

    def print_parameters(self):
        raise NotImplementedError

    def save_resource(self, path: str):
        raise NotImplementedError

    def load_resource(self, path: str):
        raise NotImplementedError
