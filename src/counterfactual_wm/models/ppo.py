# a simple ppo jax implementation with flax nnx
from typing import NamedTuple, Tuple
from flax import nnx
import jax
import jax.numpy as jnp
import optax
import gymnax
from gymnax.wrappers.purerl import LogWrapper, FlattenObservationWrapper
from jaxtyping import Array, Float

import distrax
# turn off warnings for now
import warnings
warnings.filterwarnings("ignore")

class MLP(nnx.Module):
  def __init__(self, din: int, dmid: int, dout: int, rngs: nnx.Rngs):
    self.linear = nnx.Linear(din, dmid, rngs=rngs)
    self.bn = nnx.BatchNorm(dmid, rngs=rngs)
    self.dropout = nnx.Dropout(0.1, rngs=rngs)
    self.linear_out = nnx.Linear(dmid, dout, rngs=rngs)

  def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
    x = nnx.relu(self.dropout(self.bn(self.linear(x))))
    return self.linear_out(x)

  
class ActorCritic(nnx.Module):
    def __init__(self, obs_dim: int, mid_dim: int, act_dim: int, rngs: nnx.Rngs): 
        super().__init__()
        self.actor = MLP(obs_dim, mid_dim, act_dim, rngs=rngs)
        self.critic = MLP(obs_dim, mid_dim, 1, rngs=rngs) 
    
    def __call__(self, x: Float[Array, "n_envs obs_dim"]) -> Tuple[Float[Array, "n_envs act_dim"], Float[Array, "n_envs"]]:
        """
        x: observation
        Returns: log_actions, values
        """
        pi = distrax.Categorical(logits=self.actor(x))
        return pi, self.critic(x).squeeze()


class Transition(NamedTuple):
    done: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray
    log_prob: jnp.ndarray
    obs: jnp.ndarray
    info: jnp.ndarray

class PPO():
    def __init__(self, config: dict):
        self.rng = jax.random.PRNGKey(0)
        self.config = config
        env, self.env_params = gymnax.make(config["env_name"])
        env = FlattenObservationWrapper(env)
        self.env = LogWrapper(env)
        self.obs_dim = self.env.observation_space(self.env_params).shape[0]
        self.action_dim = self.env.action_space(self.env_params).n 
        mid_dim = 64
        rngs = nnx.Rngs(jax.random.PRNGKey(0))
        model = ActorCritic(self.obs_dim, mid_dim, self.action_dim, rngs=rngs)
        self.optimizer = nnx.Optimizer(model, optax.adam(1e-3))

    def train(self):
        # from purejaxrl
        # INIT ENV
        rng, _rng = jax.random.split(self.rng)
        reset_rng = jax.random.split(_rng, self.config["num_envs"])
        obsv, env_state = jax.vmap(self.env.reset, in_axes=(0, None))(reset_rng, self.env_params)

        self.config["num_updates"] = self.config["total_timesteps"] // self.config["num_steps"] // self.config["num_envs"]

        # TRAIN LOOP
        def _update_step(runner_state, unused):

            # COLLECT TRAJECTORIES
            def _env_step(runner_state, unused):
                env_state, last_obs, optimizer, rng = runner_state

                # SELECT ACTION
                rng, _rng = jax.random.split(rng)
                pi, value = optimizer.model(last_obs)
                action = pi.sample(seed=_rng)
                log_prob = pi.log_prob(action)

                # STEP ENV
                rng, _rng = jax.random.split(rng)
                rng_step = jax.random.split(_rng, self.config["num_envs"])
                obsv, env_state, reward, done, info = jax.vmap(self.env.step, in_axes=(0,0,0,None))(
                    rng_step, env_state, action, self.env_params
                )
                transition = Transition(
                    done, action, value, reward, log_prob, last_obs, info
                )
                runner_state = (env_state, obsv, optimizer, rng)
                return runner_state, transition

            s_env_step = nnx.scan(
                _env_step, length=self.config["num_steps"]
            )
            runner_state, traj_batch = s_env_step(runner_state, None)

            # CALCULATE ADVANTAGE
            env_state, last_obs, optimizer, rng = runner_state
            # last_obs shape: (2, 400) (n_envs, obs_dim)
            # _, last_val = network.apply(train_state.params, last_obs)
            _, last_val = optimizer.model(last_obs) 
            # last_val shape: (2,) (n_envs,) 
            # traj_batch contains arrays with (5,2) shape -> (num_steps, num_envs) 
            # obs is (128,4,4) (n_steps, n_envs, obs_dim)
            # rest is (128,4) (n_steps, n_envs)

            def _calculate_gae(traj_batch, last_val):
                gamma = 0.99
                gae_lambda = 0.95
                def _get_advantages(gae_and_next_value, transition):
                    gae, next_value = gae_and_next_value #(2,1), (2,1)
                    done, value, reward = (
                        transition.done, #(5,2) (num_steps, num_envs)
                        transition.value, #(5,2)
                        transition.reward, #(5,2)
                    )
                    delta = reward + gamma * next_value * (1 - done) - value
                    gae = (
                        delta
                        + gamma * gae_lambda * (1 - done) * gae
                    )
                    return (gae, value), gae

                _, advantages = jax.lax.scan(
                    _get_advantages,
                    (jnp.zeros_like(last_val), last_val),
                    traj_batch,
                    reverse=True,
                    unroll=16,
                )
                return advantages, advantages + traj_batch.value

            # goal: calculate gae for each env and each step -> output shape: (num_steps, num_envs)
            advantages, targets = _calculate_gae(traj_batch, last_val)

            # UPDATE NETWORK
            def _update_epoch(update_state, unused):
                self.config["minibatch_size"] = self.config["num_envs"] * self.config["num_steps"] // self.config["num_minibatches"]

                def _update_minibatch(optimizer, batch_info):
                    traj_batch, advantages, targets = batch_info
                    # should all have shape (n_steps), which they do
                    clip_eps = 0.2
                    vf_coef = 0.5
                    ent_coef = 0.01


                    def _loss_fn(model: ActorCritic, traj_batch, gae, targets):
                        # RERUN NETWORK
                        pi, value = model(traj_batch.obs) 
                        log_prob = pi.log_prob(traj_batch.action)
                        # CALCULATE VALUE LOSS
                        value_pred_clipped = traj_batch.value + (
                            value - traj_batch.value
                        ).clip(-clip_eps, clip_eps)
                        value_losses = jnp.square(value - targets)
                        value_losses_clipped = jnp.square(value_pred_clipped - targets)
                        value_loss = (
                            0.5 * jnp.maximum(value_losses, value_losses_clipped).mean()
                        )

                        # CALCULATE ACTOR LOSS
                        ratio = jnp.exp(log_prob - traj_batch.log_prob)
                        gae = (gae - gae.mean()) / (gae.std() + 1e-8)
                        loss_actor1 = ratio * gae
                        loss_actor2 = (
                            jnp.clip(
                                ratio,
                                1.0 - clip_eps,
                                1.0 + clip_eps,
                            )
                            * gae
                        )
                        loss_actor = -jnp.minimum(loss_actor1, loss_actor2)
                        loss_actor = loss_actor.mean()
                        entropy = pi.entropy().mean()

                        total_loss = (
                            loss_actor
                            + vf_coef * value_loss
                            - ent_coef * entropy
                        )
                        return total_loss, (value_loss, loss_actor, entropy)

                    grad_fn = nnx.value_and_grad(_loss_fn, has_aux=True)
                    total_loss, grads = grad_fn(optimizer.model, traj_batch, advantages, targets)

                    optimizer.update(grads)

                    return optimizer, total_loss

                traj_batch, advantages, targets, optimizer, rng = update_state
                rng, _rng = jax.random.split(rng)
                batch_size = self.config["minibatch_size"] * self.config["num_minibatches"]
                assert (
                    batch_size == self.config["num_steps"] * self.config["num_envs"]
                ), "batch size must be equal to number of steps * number of envs"
                permutation = jax.random.permutation(_rng, batch_size)
                batch = (traj_batch, advantages, targets)
                batch = jax.tree_util.tree_map(
                    lambda x: x.reshape((batch_size,) + x.shape[2:]), batch
                )
                shuffled_batch = jax.tree_util.tree_map(
                    lambda x: jnp.take(x, permutation, axis=0), batch
                )
                minibatches = jax.tree_util.tree_map(
                    lambda x: jnp.reshape(
                        x, [self.config["num_minibatches"], -1] + list(x.shape[1:])
                    ),
                    shuffled_batch,
                )
                # minibatches consists of (Transition, Float[Array, "n_envs n_steps"], Float[Array, "n_envs n_steps"])
                # so traj_batch, advantages, targets
                # each array in traj_batch has also shape (n_envs, n_steps)

                # TODO: might want to use vmap instead?
                s_update_minibatch = nnx.scan(
                    _update_minibatch 
                )
                _, total_loss = s_update_minibatch(optimizer, minibatches)
                update_state = (traj_batch, advantages, targets, optimizer, rng)
                return update_state, total_loss

            update_state = (traj_batch, advantages, targets, optimizer, rng)
            s_update_epoch = nnx.scan(
                _update_epoch, length=self.config["update_epochs"]
            )
            update_state, loss_info = s_update_epoch(update_state, None)

            metric = traj_batch.info
            rng = update_state[-1]

            runner_state = (env_state, last_obs, optimizer, rng)
            return runner_state, metric

        rng, _rng = jax.random.split(rng)
        runner_state = (env_state, obsv, self.optimizer, _rng)
        s_update = nnx.scan(
            _update_step, length=self.config["num_updates"]
        )
        runner_state, metric = s_update(runner_state, None)
        return {"runner_state": runner_state, "metrics": metric}


# ppo = PPO("Breakout-MinAtar")
from counterfactual_wm.utils.config_parser import parse_config
config = parse_config("configs/ppo.yaml")
print(config)
ppo = PPO(config)
# Can be jitted
out_dict = ppo.train()
print(out_dict)