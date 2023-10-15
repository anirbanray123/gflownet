import tensorflow as tf
import numpy as np
from collections import defaultdict

# Replace torch.device with the desired GPU/CPU configuration
device = 'cpu'

def make_mlp(layer_sizes, activation=tf.nn.leaky_relu, output_activation=None):
    model = tf.keras.Sequential()
    for i in range(len(layer_sizes) - 1):
        model.add(tf.keras.layers.Dense(layer_sizes[i], activation=activation))
    if output_activation:
        model.add(tf.keras.layers.Dense(layer_sizes[-1], activation=output_activation))
    else:
        model.add(tf.keras.layers.Dense(layer_sizes[-1]))
    return model

def Categorical(logits):
    return tfp.distributions.Categorical(logits=logits)

class SplitCategorical:
    def __init__(self, n, logits):
        self.cats = [Categorical(logits=logits[..., :n]), Categorical(logits=logits[..., n:])]
        self.n = n
        self.logits = logits

    def sample(self):
        split = tf.random.uniform(self.logits.shape[:-1]) < 0.5
        return tf.where(split, self.cats[0].sample(), self.n + self.cats[1].sample())

    def log_prob(self, a):
        split = a < self.n
        log_one_half = np.log(0.5)
        return tf.where(split, self.cats[0].log_prob(tf.minimum(a, self.n - 1)), self.cats[1].log_prob(tf.maximum(a - self.n, 0))) + log_one_half

    def entropy(self):
        return tfp.distributions.Categorical(probs=tf.concat([self.cats[0].probs, self.cats[1].probs], axis=-1) * 0.5).entropy()

class FlowNetAgent:
    def __init__(self, args, envs):
        self.model = make_mlp([args.horizon * args.ndim] +
                              [args.n_hid] * args.n_layers +
                              [args.ndim + 1])
        self.model.build((None, args.horizon * args.ndim))
        self.target = tf.keras.models.clone_model(self.model)
        self.target.set_weights(self.model.get_weights())
        self.envs = envs
        self.ndim = args.ndim
        self.tau = args.bootstrap_tau
        self.replay = ReplayBuffer(args, envs[0])

    def sample_many(self, mbsize, all_visited):
        batch = []
        batch += self.replay.sample()
        s = np.stack([i.reset()[0] for i in self.envs])
        done = np.array([False] * mbsize)
        while not all(done):
            acts = Categorical(logits=self.model(s)).sample()
            step = []
            for i in range(len(done)):
                if not done[i]:
                    step.append(self.envs[i].step(acts[i]))
                else:
                    step.append((None, None, None))
            p_a = [self.envs[0].parent_transitions(step[i][3], acts[i] == self.ndim) for i in range(len(step))]
            batch += [[p, a, [r], [sp], [d]] for (p, a), (_, r, d, sp) in zip(p_a, step)]
            done = np.logical_or(done, [d or step[i][2] for i, d in enumerate(done)])
            s = np.stack([step[i][3][0] for i in range(len(step)) if not step[i][2]])
        return batch

    def learn_from(self, it, batch):
        loginf = tf.constant([1000.0])
        batch_idxs = np.array([i for i, (parents, _, _, _, _) in enumerate(batch)
                               for _ in parents])
        parents, actions, r, sp, done = map(np.concatenate, zip(*batch))
        parents_Qsa = self.model(parents)[:, actions.astype(int)]
        in_flow = tf.math.log(tf.reduce_sum(np.exp(parents_Qsa), axis=1))
        next_q = self.target(sp) if self.tau > 0 else self.model(sp)
        next_qd = next_q * (1 - done)[:, np.newaxis] + done[:, np.newaxis] * -loginf
        out_flow = tf.reduce_logsumexp(np.column_stack([np.log(r), next_qd]), axis=1)
        loss = tf.reduce_mean((in_flow - out_flow) ** 2)
        term_loss = tf.reduce_sum((in_flow - out_flow) * done) / (tf.reduce_sum(done) + 1e-2)
        loss += self.tau * term_loss
        if it > 1:
            for w, tw in zip(self.model.trainable_variables, self.target.trainable_variables):
                tw.assign(w + 1e-3 * (w - tw))
        return loss

class ReplayBuffer:
    def __init__(self, args, env):
        self.dims = env.observation_space.shape
        self.ndim = args.ndim
        self.obs = np.empty((args.replay_size, *self.dims), dtype=np.float32)
        self.actions = np.empty((args.replay_size, 1), dtype=np.float32)
        self.rewards = np.empty((args.replay_size, 1), dtype=np.float32)
        self.next_obs = np.empty((args.replay_size, *self.dims), dtype=np.float32)
        self.dones = np.empty((args.replay_size, 1), dtype=np.float32)
        self.num_in_buffer = 0
        self.np_random, seed = gym.utils.seeding.np_random(args.seed)
        self.indices = self.np_random.permutation(args.replay_size)

    def store(self, obs, actions, rewards, next_obs, dones):
        self.obs[self.indices[self.num_in_buffer]] = obs
        self.actions[self.indices[self.num_in_buffer]] = actions
        self.rewards[self.indices[self.num_in_buffer]] = rewards
        self.next_obs[self.indices[self.num_in_buffer]] = next_obs
        self.dones[self.indices[self.num_in_buffer]] = dones
        self.num_in_buffer += 1
        if self.num_in_buffer >= args.replay_size:
            self.num_in_buffer = 0
            self.indices = self.np_random.permutation(args.replay_size)

    def sample(self):
        if self.num_in_buffer < 5000:
            return []
        return [self.obs[self.indices[0:self.num_in_buffer]],
                self.actions[self.indices[0:self.num_in_buffer]],
                self.rewards[self.indices[0:self.num_in_buffer]],
                self.next_obs[self.indices[0:self.num_in_buffer]],
                self.dones[self.indices[0:self.num_in_buffer]]]

class SACAgent:
    def __init__(self, args, envs):
        self.horizon = args.horizon
        self.ndim = args.ndim
        self.envs = envs
        self.replay = ReplayBuffer(args, envs[0])
        self.temp = args.temp
        self.actors = make_mlp([self.horizon * args.ndim] +
                              [args.n_hid] * args.n_layers +
                              [args.ndim], output_activation=tf.nn.tanh)
        self.actors.build((None, self.horizon * args.ndim))
        self.actors_trainable = make_mlp([self.horizon * args.ndim] +
                              [args.n_hid] * args.n_layers +
                              [args.ndim], output_activation=tf.nn.tanh)
        self.actors_trainable.build((None, self.horizon * args.ndim))
        self.q_critic = make_mlp([self.horizon * args.ndim + 1] +
                               [args.n_hid] * args.n_layers +
                               [args.ndim])
        self.q_critic.build((None, self.horizon * args.ndim + 1))
        self.q_critic_trainable = make_mlp([self.horizon * args.ndim + 1] +
                               [args.n_hid] * args.n_layers +
                               [args.ndim])
        self.q_critic_trainable.build((None, self.horizon * args.ndim + 1))
        self.target_q = make_mlp([self.horizon * args.ndim + 1] +
                              [args.n_hid] * args.n_layers +
                              [args.ndim])
        self.target_q.build((None, self.horizon * args.ndim + 1))
        self.actors.optimizer = tf.optimizers.Adam(learning_rate=1e-3)
        self.q_critic.optimizer = tf.optimizers.Adam(learning_rate=1e-3)
        self.discount = args.discount
        self.loss = []

    def learn(self, it):
        obs = np.stack([i.reset()[0] for i in self.envs])
        done = np.array([False] * self.ndim)
        self.loss = []
        state_noise = self.q_critic_trainable(obs)[:, -1]
        # Note: For SAC, it's common to add small noise to the actions to encourage exploration
        entropy = 0.0
        for _ in range(100):
            acts = self.actors_trainable(obs) + np.random.normal(0, 0.2, self.horizon * self.ndim)
            q_critic = self.q_critic_trainable(tf.concat([obs, acts], axis=1))
            policy_a = self.actors(obs)
            logp = self.get_logp(acts, policy_a, state_noise)
            q_targets = tf.reduce_logsumexp(q_critic, axis=1)
            q_loss = -tf.reduce_sum(q_targets - logp) + entropy
            q_critic_grads = tf.gradients(q_loss, self.q_critic.trainable_variables)
            self.q_critic.optimizer.apply_gradients(zip(q_critic_grads, self.q_critic.trainable_variables))
            policy_a_noise = self.actors_trainable(obs)
            q1_pred = self.q_critic_trainable(tf.concat([obs, policy_a_noise], axis=1))
            policy_loss = -tf.reduce_sum(q1_pred) + entropy
            actors_grads = tf.gradients(policy_loss, self.actors.trainable_variables)
            self.actors.optimizer.apply_gradients(zip(actors_grads, self.actors.trainable_variables))
            self.q_critic_trainable.set_weights(self.q_critic.get_weights())
            self.actors_trainable.set_weights(self.actors.get_weights())

            self.loss.append(q_loss)

        return np.mean(self.loss)

    def get_logp(self, acts, policy_a, state_noise):
        return self.q_critic_trainable(tf.concat([self.target_q(acts), acts], axis=1))[:, -1] - state_noise

def main():
    # Define your argparse configuration here.
    # You need to adjust the TensorFlow code according to your specific requirements.
    pass

if __name__ == "__main__":
    main()
