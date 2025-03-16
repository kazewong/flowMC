import jax.numpy as jnp
from flowMC.resource.buffers import Buffer


class TestBuffer:
    def test_buffer(self):
        buffer = Buffer("test", (10, 10), cursor_dim=1)
        assert buffer.name == "test"
        assert buffer.data.shape == (10, 10)
        assert buffer.cursor == 0
        assert buffer.cursor_dim == 1

    def test_update_buffer(self):
        buffer = Buffer("test", (10, 10), cursor_dim=0)
        buffer.update_buffer(jnp.ones((10, 10)))
        assert (buffer.data == jnp.ones((10, 10))).all()
