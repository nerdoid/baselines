import tensorflow as tf
import tensorflow.contrib.layers as layers


def _mlp(hiddens, inpt, num_actions, scope, reuse=False):
    with tf.variable_scope(scope, reuse=reuse):
        out = inpt
        for hidden in hiddens:
            out = layers.fully_connected(out, num_outputs=hidden, activation_fn=tf.nn.relu)
        out = layers.fully_connected(out, num_outputs=num_actions, activation_fn=None)
        return out


def mlp(hiddens=[]):
    """This model takes as input an observation and returns values of all actions.

    Parameters
    ----------
    hiddens: [int]
        list of sizes of hidden layers

    Returns
    -------
    q_func: function
        q_function for DQN algorithm.
    """
    return lambda *args, **kwargs: _mlp(hiddens, *args, **kwargs)


def _cnn_to_mlp(convs, hiddens, dueling, inpt, num_actions, scope, reuse=False):
    with tf.variable_scope(scope, reuse=reuse):
        out = inpt
        with tf.variable_scope("convnet"):
            for num_outputs, kernel_size, stride in convs:
                out = layers.convolution2d(out,
                                           num_outputs=num_outputs,
                                           kernel_size=kernel_size,
                                           stride=stride,
                                           activation_fn=tf.nn.relu)
        out = layers.flatten(out)
        with tf.variable_scope("action_value"):
            action_out = out
            for hidden in hiddens:
                action_out = layers.fully_connected(action_out, num_outputs=hidden, activation_fn=tf.nn.relu)
            action_scores = layers.fully_connected(action_out, num_outputs=num_actions, activation_fn=None)

        if dueling:
            with tf.variable_scope("state_value"):
                state_out = out
                for hidden in hiddens:
                    state_out = layers.fully_connected(state_out, num_outputs=hidden, activation_fn=tf.nn.relu)
                state_score = layers.fully_connected(state_out, num_outputs=1, activation_fn=None)
            action_scores_mean = tf.reduce_mean(action_scores, 1)
            action_scores_centered = action_scores - tf.expand_dims(action_scores_mean, 1)
            return state_score + action_scores_centered
        else:
            return action_scores
        return out


def cnn_to_mlp(convs, hiddens, dueling=False):
    """This model takes as input an observation and returns values of all actions.

    Parameters
    ----------
    convs: [(int, int int)]
        list of convolutional layers in form of
        (num_outputs, kernel_size, stride)
    hiddens: [int]
        list of sizes of hidden layers
    dueling: bool
        if true double the output MLP to compute a baseline
        for action scores

    Returns
    -------
    q_func: function
        q_function for DQN algorithm.
    """

    return lambda *args, **kwargs: _cnn_to_mlp(convs, hiddens, dueling, *args, **kwargs)


def _state_to_phi(state, reuse=None):
    """Builds shared convolutional layers to process states into intermediate
    representations for inverse action prediction.
    """
    with tf.variable_scope('phi', reuse=reuse) as scope:
        out = state
        for i in range(5):
            out = layers.convolution2d(
                out,
                num_outputs=32,
                kernel_size=[3, 3],
                stride=2,
                activation_fn=tf.nn.elu
            )

        return layers.flatten(out)


def _states_to_action(hidden_size, num_actions, s_t, s_tp1):
    phi_t = _state_to_phi(s_t)
    phi_tp1 = _state_to_phi(s_tp1, reuse=True)

    phies = tf.concat([phi_t, phi_tp1], 1)
    phies = layers.flatten(phies)

    inverse_hidden = layers.fully_connected(
        phies, num_outputs=hidden_size, activation_fn=tf.nn.elu
    )
    inverse_out = layers.fully_connected(
        inverse_hidden, num_outputs=num_actions
    )

    return (inverse_out, phi_t, phi_tp1)


def states_to_action(hidden_size):
    """This model takes as input two consecutive observations and returns
    the action that was likely executed to arrive at the second observation.

    Parameters
    ----------
    hidden_size: int
        size of the hidden layer

    Returns
    -------
    inv_act_func: function
        function that returns one-hot prediction op for the action just
        executed.
    """

    return lambda *args, **kwargs: _states_to_action(hidden_size, *args, **kwargs)


def _state_action_to_phi_tp1_loss(hidden_size, output_size, phi_t, phi_tp1,
                                  a_t):
    inpt = tf.concat([phi_t, a_t], 1)

    forward_hidden = layers.fully_connected(
        inpt, num_outputs=hidden_size, activation_fn=tf.nn.elu
    )
    phi_tp1_pred = layers.fully_connected(
        forward_hidden, num_outputs=output_size
    )

    phi_tp1_error = phi_tp1_pred - phi_tp1
    phi_tp1_loss = tf.nn.l2_loss(phi_tp1_error)

    return phi_tp1_loss


def state_action_to_phi_tp1_loss(hidden_size, output_size):
    """This model takes as input the current observation and chosen action
    and returns the predicted next observation latent features (phi_tp1).

    Parameters
    ----------
    hidden_size: int
        size of the hidden layer

    Returns
    -------
    phi_tp1_loss_func: function
        function that returns one-hot prediction op for the action just
        executed.
    """

    return lambda *args, **kwargs: _state_action_to_phi_tp1_loss(hidden_size, output_size, *args, **kwargs)
