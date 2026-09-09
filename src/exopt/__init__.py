import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)
EPS = float(jnp.sqrt(jnp.finfo(float).eps))  # square-root machine precision
MAX_TORQUE = 13.0  # [Nm] hardware torque limit per motor
