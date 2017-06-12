import gym

from baselines import deepq
from baselines.common.atari_wrappers_deprecated import wrap_dqn, ScaledFloatFrame


def main():
    env = gym.make("MontezumaRevengeNoFrameskip-v4")
    env = gym.wrappers.Monitor(env, "results/gym", force=True)
    env = ScaledFloatFrame(wrap_dqn(env))
    model = deepq.models.cnn_to_mlp(
        convs=[(32, 8, 4), (64, 4, 2), (64, 3, 1)],
        hiddens=[256],
        dueling=True
    )
    inv_act_model = deepq.models.states_to_action(
        hidden_size=256
    )
    phi_tp1_loss_model = deepq.models.state_action_to_phi_tp1_loss(
        hidden_size=256, output_size=288
    )

    act = deepq.learn(
        env,
        q_func=model,
        inv_act_func=inv_act_model,
        phi_tp1_loss_func=phi_tp1_loss_model,
        lr=1e-4,
        max_timesteps=10000000,
        buffer_size=10000,
        exploration_fraction=0.1,
        exploration_final_eps=0.01,
        train_freq=4,
        learning_starts=10000,
        target_network_update_freq=1000,
        gamma=0.99,
        prioritized_replay=True
    )
    act.save("results/montezuma_model_icm.pkl")
    env.close()


if __name__ == '__main__':
    main()
