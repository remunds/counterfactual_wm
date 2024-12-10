# a simple ppo jax implementation with flax nnx
from typing import NamedTuple
from flax import nnx
import jax
import jax.numpy as jnp
import optax
import gymnax
from gymnax.wrappers.purerl import LogWrapper, FlattenObservationWrapper

import distrax
# turn off warnings for now
import warnings
warnings.filterwarnings("ignore")
from jax import debug

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
    
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        x: observation
        Returns: log_actions, values
        """
        pi = distrax.Categorical(logits=self.actor(x))
        return pi, self.critic(x)


class Transition(NamedTuple):
    done: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray
    log_prob: jnp.ndarray
    obs: jnp.ndarray
    info: jnp.ndarray

class PPO():
    def __init__(self, env_name):
        self.rng = jax.random.PRNGKey(0)
        self.n_envs = 2
        self.env_name = env_name
        env, self.env_params = gymnax.make(env_name)
        env = FlattenObservationWrapper(env)
        self.env = LogWrapper(env)
        self.obs_dim = self.env.observation_space(self.env_params).shape[0]
        self.action_dim = self.env.action_space(self.env_params).n 
        mid_dim = 64
        rngs = nnx.Rngs(jax.random.PRNGKey(0))
        self.ac = ActorCritic(self.obs_dim, mid_dim, self.action_dim, rngs=rngs)
        self.optimizer = nnx.Optimizer(self.ac, optax.adam(1e-3))

    def train(self, num_steps):
        # from purejaxrl
        # INIT ENV
        rng, _rng = jax.random.split(self.rng)
        reset_rng = jax.random.split(_rng, self.n_envs)
        obsv, env_state = jax.vmap(self.env.reset, in_axes=(0, None))(reset_rng, self.env_params)

        # TRAIN LOOP
        def _update_step(runner_state, unused):

            # COLLECT TRAJECTORIES
            def _env_step(runner_state, unused):
                env_state, last_obs, rng = runner_state

                # SELECT ACTION
                rng, _rng = jax.random.split(rng)
                # pi, value = network.apply(train_state.params, last_obs)
                pi, value = self.ac(last_obs)
                action = pi.sample(seed=_rng)
                log_prob = pi.log_prob(action)

                # STEP ENV
                rng, _rng = jax.random.split(rng)
                rng_step = jax.random.split(_rng, self.n_envs)
                obsv, env_state, reward, done, info = jax.vmap(self.env.step, in_axes=(0,0,0,None))(
                    rng_step, env_state, action, self.env_params
                )
                transition = Transition(
                    done, action, value, reward, log_prob, last_obs, info
                )
                runner_state = (env_state, obsv, rng)
                return runner_state, transition

            runner_state, traj_batch = jax.lax.scan(
                _env_step, runner_state, None, num_steps
            )

            # CALCULATE ADVANTAGE
            env_state, last_obs, rng = runner_state
            # _, last_val = network.apply(train_state.params, last_obs)
            _, last_val = self.ac(last_obs) 

            #TODO: Error here...
            def _calculate_gae(traj_batch, last_val):
                gamma = 0.99
                gae_lambda = 0.95
                def _get_advantages(gae_and_next_value, transition):
                    gae, next_value = gae_and_next_value
                    done, value, reward = (
                        transition.done,
                        transition.value,
                        transition.reward,
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

            advantages, targets = _calculate_gae(traj_batch, last_val)
            print(advantages.shape, targets.shape)

            # UPDATE NETWORK
        #     def _update_epoch(update_state, unused):
        #         def _update_minbatch(train_state, batch_info):
        #             traj_batch, advantages, targets = batch_info

        #             def _loss_fn(params, traj_batch, gae, targets):
        #                 # RERUN NETWORK
        #                 pi, value = network.apply(params, traj_batch.obs)
        #                 log_prob = pi.log_prob(traj_batch.action)

        #                 # CALCULATE VALUE LOSS
        #                 value_pred_clipped = traj_batch.value + (
        #                     value - traj_batch.value
        #                 ).clip(-config["CLIP_EPS"], config["CLIP_EPS"])
        #                 value_losses = jnp.square(value - targets)
        #                 value_losses_clipped = jnp.square(value_pred_clipped - targets)
        #                 value_loss = (
        #                     0.5 * jnp.maximum(value_losses, value_losses_clipped).mean()
        #                 )

        #                 # CALCULATE ACTOR LOSS
        #                 ratio = jnp.exp(log_prob - traj_batch.log_prob)
        #                 gae = (gae - gae.mean()) / (gae.std() + 1e-8)
        #                 loss_actor1 = ratio * gae
        #                 loss_actor2 = (
        #                     jnp.clip(
        #                         ratio,
        #                         1.0 - config["CLIP_EPS"],
        #                         1.0 + config["CLIP_EPS"],
        #                     )
        #                     * gae
        #                 )
        #                 loss_actor = -jnp.minimum(loss_actor1, loss_actor2)
        #                 loss_actor = loss_actor.mean()
        #                 entropy = pi.entropy().mean()

        #                 total_loss = (
        #                     loss_actor
        #                     + config["VF_COEF"] * value_loss
        #                     - config["ENT_COEF"] * entropy
        #                 )
        #                 return total_loss, (value_loss, loss_actor, entropy)

        #             grad_fn = jax.value_and_grad(_loss_fn, has_aux=True)
        #             total_loss, grads = grad_fn(
        #                 train_state.params, traj_batch, advantages, targets
        #             )
        #             train_state = train_state.apply_gradients(grads=grads)
        #             return train_state, total_loss

        #         train_state, traj_batch, advantages, targets, rng = update_state
        #         rng, _rng = jax.random.split(rng)
        #         batch_size = config["MINIBATCH_SIZE"] * config["NUM_MINIBATCHES"]
        #         assert (
        #             batch_size == config["NUM_STEPS"] * config["NUM_ENVS"]
        #         ), "batch size must be equal to number of steps * number of envs"
        #         permutation = jax.random.permutation(_rng, batch_size)
        #         batch = (traj_batch, advantages, targets)
        #         batch = jax.tree_util.tree_map(
        #             lambda x: x.reshape((batch_size,) + x.shape[2:]), batch
        #         )
        #         shuffled_batch = jax.tree_util.tree_map(
        #             lambda x: jnp.take(x, permutation, axis=0), batch
        #         )
        #         minibatches = jax.tree_util.tree_map(
        #             lambda x: jnp.reshape(
        #                 x, [config["NUM_MINIBATCHES"], -1] + list(x.shape[1:])
        #             ),
        #             shuffled_batch,
        #         )
        #         train_state, total_loss = jax.lax.scan(
        #             _update_minbatch, train_state, minibatches
        #         )
        #         update_state = (train_state, traj_batch, advantages, targets, rng)
        #         return update_state, total_loss

        #     update_state = (train_state, traj_batch, advantages, targets, rng)
        #     update_state, loss_info = jax.lax.scan(
        #         _update_epoch, update_state, None, config["UPDATE_EPOCHS"]
        #     )
        #     train_state = update_state[0]
        #     metric = traj_batch.info
        #     rng = update_state[-1]

        #     runner_state = (train_state, env_state, last_obs, rng)
        #     return runner_state, metric

        rng, _rng = jax.random.split(rng)
        runner_state = (env_state, obsv, _rng)
        runner_state, metric = jax.lax.scan(
            _update_step, runner_state, None, 1 #TODO: change to num_update_steps
        )
        return {"runner_state": runner_state, "metrics": metric}

        




ppo = PPO("Breakout-MinAtar")
ppo.train(5)