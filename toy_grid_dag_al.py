import tensorflow as tf
import numpy as np
from collections import defaultdict

# Define the UCB class in TensorFlow
class UCB:
    def __init__(self, model, kappa):
        self.model = model
        self.kappa = kappa

    def __call__(self, x):
        t_x = tf.constant(np.array([[x]], dtype=np.float32))
        output = self.model(t_x, training=False)
        mean, std = output.mean(), tf.sqrt(output.variance)
        return tf.maximum(mean + self.kappa * std, 0.0).numpy()[0]

    def many(self, x):
        output = self.model(tf.constant(x, dtype=tf.float32), training=False)
        mean, std = output.mean(), tf.sqrt(output.variance)
        return (mean + self.kappa * std).numpy()

# Define the get_init_data function in TensorFlow
def get_init_data(args, func):
    env = GridEnv(args.horizon, args.ndim, func=func)
    td, end_states, true_r = env.true_density()
    idx = np.random.choice(len(end_states), args.num_init_points, replace=False)
    end_states = np.array(end_states)
    true_r = np.array(true_r)
    states, y = end_states[idx], true_r[idx]
    x = np.array([env.s2x(s) for s in states], dtype=np.float32)
    init_data = x, y
    return init_data, td, end_states, true_r, env

# Define the update_proxy function in TensorFlow
def update_proxy(args, data):
    train_x, train_y = data
    model = SingleTaskGP(train_x, train_y, covar_module=gpytorch.kernels.ScaleKernel(gpytorch.kernels.MaternKernel(nu=0.5)))
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(model.likelihood, model)
    fit_gpytorch_model(mll)
    return model

# Define the diverse_topk_mean_reward function in TensorFlow
def diverse_topk_mean_reward(args, d_prev, d):
    topk_new_indices = tf.nn.top_k(d[1], k=args.reward_topk).indices
    topk_old_indices = tf.nn.top_k(d_prev[1], k=args.reward_topk).indices
    topk_new = tf.reduce_mean(tf.gather(d[1], topk_new_indices))
    topk_old = tf.reduce_mean(tf.gather(d_prev[1], topk_old_indices))
    new_reward = topk_new + args.reward_lambda * get_pairwise_distances(d[0][topk_new_indices].numpy())
    old_reward = topk_old + args.reward_lambda * get_pairwise_distances(d_prev[0][topk_old_indices].numpy())
    return (new_reward - old_reward).numpy()

# Define the get_pairwise_distances function in TensorFlow
def get_pairwise_distances(arr):
    return np.mean(np.tril(np.linalg.norm(arr[:, np.newaxis] - arr, axis=-1))) * 2 / (arr.shape[0] * (arr.shape[0] - 1))

# Define the main function using TensorFlow
def main(args):
    set_device(args.dev)
    f = {
        'default': None,
        'cos_N': func_cos_N,
        'corners': func_corners,
        'corners_floor_A': func_corners_floor_A,
        'corners_floor_B': func_corners_floor_B,
    }[args.func]

    init_data, td, end_states, true_r, env = get_init_data(args, f)
    all_x, all_y = tf.constant(end_states, dtype=tf.float32), tf.constant(true_r, dtype=tf.float32)
    init_x, init_y = tf.constant(init_data[0], dtype=tf.float32), tf.constant(init_data[1], dtype=tf.float32)
    if not os.path.exists(args.save_path):
        os.mkdir(args.save_path)

    reward = []
    base_path = os.path.join(args.save_path, args.method)
    if not os.path.exists(base_path):
        os.mkdir(base_path)

    dataset = tf.data.Dataset.from_tensor_slices((init_x, init_y))
    model = update_proxy(args, dataset)
    metrics = []
    for i in range(args.num_iter):
        model.fit(train_x=init_x, train_y=init_y)
        model.eval()
        dataset = tf.data.Dataset.from_tensor_slices((init_x, init_y))
        func = UCB(model, args.kappa) if args.use_model else f
        agent, _metrics = train_generative_model(args, func)
        metrics.append(_metrics)
        new_dataset = generate_batch(args, agent, dataset, env)
        reward.append(diverse_topk_mean_reward(args, dataset, new_dataset))
        print(reward)
        dataset = new_dataset
        model = update_proxy(args, dataset)
        pickle.dump({
            'metrics': metrics,
            'rewards': reward,
            'args': args
        }, gzip.open(os.path.join(base_path, 'result.pkl.gz'), 'wb'))

# Define the train_generative_model function using TensorFlow
def train_generative_model(args, f):
    args.is_mcmc = args.method in ['mars', 'mcmc']

    env = GridEnv(args.horizon, args.ndim, func=f, allow_backward=args.is_mcmc)
    envs = [GridEnv(args.horizon, args.ndim, func=f, allow_backward=args.is_mcmc)
            for i in range(args.bufsize)]
    ndim = args.ndim

    if args.method == 'flownet':
        agent = FlowNetAgent(args, envs)
    elif args.method == 'mars':
        agent = MARSAgent(args, envs)
    elif args.method == 'mcmc':
        agent = MHAgent(args, envs)
    elif args.method == 'ppo':
        agent = PPOAgent(args, envs)
    elif args.method == 'random_traj':
        agent = RandomTrajAgent(args, envs)

    opt = make_opt(agent.trainable_variables, args)

    # metrics
    all_losses = []
    all_visited = []
    empirical_distrib_losses = []
    ttsr = max(int(args.train_to_sample_ratio), 1)
    sttr = max(int(1 / args.train_to_sample_ratio), 1)  # sample to train ratio

    if args.method == 'ppo':
        ttsr = args.ppo_num_epochs
        sttr = args.ppo_epoch_size

    for i in range(args.n_train_steps + 1):
        data = []
        for j in range(sttr):
            data += agent.sample_many(args.mbsize, all_visited)
        for j in range(ttsr):
            losses = agent.learn_from(i * ttsr + j, data)  # returns (opt loss, *metrics)
            if losses is not None:
                # Apply gradients
                gradients = tape.gradient(losses[0], agent.trainable_variables)
                opt.apply_gradients(zip(gradients, agent.trainable_variables))
                all_losses.append([i.numpy() for i in losses])

        if not i % 100:
            empirical_distrib_losses.append(
                compute_empirical_distribution_error(env, all_visited[-args.num_empirical_loss:]))
            if args.progress:
                k1, kl = empirical_distrib_losses[-1]
                print('empirical L1 distance', k1, 'KL', kl)
                if len(all_losses):
                    print(*[f'{np.mean([i[j].numpy() for i in all_losses[-100:]]):.3f}'
                            for j in range(len(all_losses[0]))])

    metrics = {'losses': np.float32(all_losses),
               'model': agent.model if agent.model else None,
               'visited': np.int8(all_visited),
               'emp_dist_loss': empirical_distrib_losses}
    return agent, metrics

if __name__ == '__main__':
    args = parser.parse_args()
    tf.config.set_visible_devices([], 'GPU')
    main(args)
