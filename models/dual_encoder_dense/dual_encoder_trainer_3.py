import os

os.environ["KERAS_BACKEND"] = 'tensorflow'

from models.dual_encoder_dense.model_dual_encoder_dense import dot_semantic_nn
from dataset.ubuntu_dialogue_corpus import UDCDataset
from test_tube.log import Experiment
from tensorflow.contrib.keras.api.keras.utils import Progbar
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import keras
from keras import backend as K
from keras.models import Model
from keras.layers import Input, Dense, Lambda, Layer


class MyLayer(Layer):

    def __init__(self, output_dim, **kwargs):
        self.output_dim = output_dim
        super(MyLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.

        # These are the 3 trainable weights for word_embedding, lstm_output1 and lstm_output2
        self.kernel1 = self.add_weight(name='kernel1',
                                       shape=(3,),
                                       initializer='uniform',
                                       trainable=True)
        # This is the bias weight
        self.kernel2 = self.add_weight(name='kernel2',
                                       shape=(),
                                       initializer='uniform',
                                       trainable=True)
        super(MyLayer, self).build(input_shape)

    def call(self, x):
        # Get all the outputs of elmo_model
        elmo_model = hub.Module("https://tfhub.dev/google/elmo/2", trainable=True)
        model = elmo_model(x, signature="default", as_dict=True)

        # Embedding activation output
        activation1 = model["word_emb"]

        # First LSTM layer output
        activation2 = model["lstm_outputs1"]

        # Second LSTM layer output
        activation3 = model["lstm_outputs2"]
        activation2 = tf.reduce_mean(activation2, axis=1)
        activation3 = tf.reduce_mean(activation3, axis=1)

        mul1 = tf.scalar_mul(self.kernel1[0], activation1)
        mul2 = tf.scalar_mul(self.kernel1[1], activation2)
        mul3 = tf.scalar_mul(self.kernel1[2], activation3)

        sum_vector = tf.add(mul2, mul3)

        return tf.scalar_mul(self.kernel2, sum_vector)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_dim)


def prep(lst, num):
    list1 = np.array_split(lst, num)
    list2 = [l.tolist() for l in list1]
    s = '<\s>'
    list3 = [s.join(l) for l in list2]

    return list3


def train_main(hparams):
    """
    Main training routine for the dot semantic network bot
    :return:
    """

    # -----------------------
    # INIT EXPERIMENT
    # ----------------------
    exp = Experiment(name=hparams.exp_name,
                     debug=hparams.debug,
                     description=hparams.exp_desc,
                     autosave=False,
                     save_dir=hparams.test_tube_dir)

    exp.add_meta_tags(vars(hparams))

    # -----------------------
    # LOAD DATASET
    # ----------------------
    udc_dataset = UDCDataset(vocab_path=hparams.vocab_path,
                             train_path=hparams.dataset_train_path,
                             test_path=hparams.dataset_test_path,
                             val_path=hparams.dataset_val_path,
                             max_seq_len=hparams.max_seq_len)

    # -----------------------
    # INIT TF VARS
    # ----------------------
    # context holds chat history
    # utterance holds our responses
    # labels holds the ground truth labels
    context_ph = tf.placeholder(dtype="string", shape=[hparams.batch_size,], name='context_seq_in')
    utterance_ph = tf.placeholder(dtype="string", shape=[hparams.batch_size,], name='utterance_seq_in')

    # ----------------------
    # EMBEDDING LAYER
    # ----------------------
    # you can preload your own or learn in the network
    # in this case we'll just learn it in the network
    # embedding_layer = tf.Variable(tf.random_uniform([udc_dataset.vocab_size, hparams.embedding_dim], -1.0, 1.0), name='embedding')
    #x = prep(udc_dataset.train, hparams.batch_size)
    #print(type(x))
    #print(len(x))

    # elmo_model = hub.Module("https://tfhub.dev/google/elmo/2", trainable=True)
    sess = tf.Session()

    K.set_session(sess)
    # Initialize sessions
    sess.run(tf.global_variables_initializer())
    sess.run(tf.tables_initializer())
    # print('elmo')
    # context = list(udc_dataset['Context'])
    # elmo_text = elmo(context, signature="default", as_dict=True)
    # input_text = Input(shape=(100,), tensor= ,dtype="string")
    #custom_layer = MyLayer(output_dim=1024, trainable=True)(tf.convert_to_tensor(x, dtype='string'))
    # embedding = Lambda(ELMoEmbedding, output_shape=(1024, ))(input_text)

    # elmo_text = elmo(tf.squeeze(tf.cast(x, tf.string)), signature="default", as_dict=True)["default"]
    #embedding_layer = Dense(256, activation='relu', kernel_regularizer=keras.regularizers.l2(0.001))(custom_layer)

    print('embedding_layer')
    # ----------------------
    # RESOLVE EMBEDDINGS
    # ----------------------
    # look up embeddings
    #context_embedding_custom = MyLayer(output_dim=1024, trainable=True)(tf.slice(context_ph, [0], [1]))
    #utterance_embedding_custom = MyLayer(output_dim=1024, trainable=True)(tf.slice(utterance_ph, [0], [1]))
    #context_embedding = Dense(hparams.embedding_dim, activation='relu',
    #                                    kernel_regularizer=keras.regularizers.l2(0.001))(
    #                                   tf.expand_dims(context_embedding_custom, 0))
                                             #utterance_embedding = Dense(hparams.embedding_dim, activation='relu',
                                             #                                       kernel_regularizer=keras.regularizers.l2(0.001))(
                                             # tf.expand_dims(utterance_embedding_custom, 0))
    context_embedding_custom = Lambda(MyLayer(output_dim=1024, trainable=True),output_shape=(1024,))(context_ph)
    utterance_embedding_custom = Lambda(MyLayer(output_dim=1024, trainable=True),output_shape=(1024,))(utterance_ph)
    context_embedding = tf.reduce_mean(context_embedding_custom, axis=1)
    utterance_embedding = tf.reduce_mean(utterance_embedding_custom, axis=1)
    context_embedding_summed = Dense(hparams.embedding_dim, activation='relu',
                                     kernel_regularizer=keras.regularizers.l2(0.001))(context_embedding)
    context_embedding_summed = Dense(hparams.embedding_dim, activation='relu',
                            kernel_regularizer=keras.regularizers.l2(0.001))(context_embedding)
        #for batch_num in range(1,hparams.batch_size):

        #context_embedding_custom = MyLayer(output_dim=1024, trainable=True)(tf.slice(context_ph, [batch_num], [1]))
        #utterance_embedding_custom = MyLayer(output_dim=1024, trainable=True)(tf.slice(utterance_ph, [batch_num], [1]))
        #context_embedding = tf.concat([context_embedding,
        #Dense(hparams.embedding_dim, activation='relu',
        #kernel_regularizer=keras.regularizers.l2(0.001))(tf.expand_dims(context_embedding_custom, 0))],
        #axis=0)
                                          #utterance_embedding = tf.concat([utterance_embedding,
                                          #Dense(hparams.embedding_dim, activation='relu',
                                          #kernel_regularizer=keras.regularizers.l2(0.001))(tf.expand_dims(utterance_embedding_custom, 0))],
                                          #axis=0)


    # avg all embeddings (sum works better?)
    # this generates 1 vector per training example
    #context_embedding_summed = tf.reduce_mean(context_embedding, axis=1)
    #utterance_embedding_summed = tf.reduce_mean(utterance_embedding, axis=1)

    # ----------------------
    # OPTIMIZATION PROBLEM
    # ----------------------
    model, _, _, pred_opt = dot_semantic_nn(context=context_embedding_summed,
                                            utterance=utterance_embedding_summed,
                                            tng_mode=hparams.train_mode)

    # allow optiizer to be changed through hyper params
    optimizer = get_optimizer(hparams=hparams, minimize=model)

    # ----------------------
    # TF ADMIN (VAR INIT, SESS)
    # ----------------------
    sess = tf.Session()
    init_vars = tf.global_variables_initializer()
    sess.run(init_vars)

    # Add ops to save and restore all the variables.
    saver = tf.train.Saver()

    # ----------------------
    # TRAINING ROUTINE
    # ----------------------
    # admin vars
    nb_batches_served = 0
    eval_every_n_batches = hparams.eval_every_n_batches

    train_err = 1000
    precission_at_1 = 0
    precission_at_2 = 0

    # iter for the needed epochs
    print('\n\n', '-' * 100, '\n  {} TRAINING\n'.format(hparams.exp_name.upper()), '-' * 100, '\n\n')
    for epoch in range(hparams.nb_epochs):
        print('training epoch:', epoch + 1)
        progbar = Progbar(target=udc_dataset.nb_tng, width=50)
        train_gen = udc_dataset.train_generator(batch_size=hparams.batch_size, max_epochs=1)

        # mini batches
        for batch_context, batch_utterance in train_gen:


            feed_dict = {
                context_ph: batch_context,
                utterance_ph: batch_utterance
            }
            print("optimizer!")
            # OPT: run one step of optimization
            optimizer.run(session=sess, feed_dict=feed_dict)
            # update loss metrics
            if nb_batches_served % eval_every_n_batches == 0:
                # calculate test error
                train_err = model.eval(session=sess, feed_dict=feed_dict)
                precission_at_1 = test_precision_at_k(pred_opt, feed_dict, k=1, sess=sess)
                precission_at_2 = test_precision_at_k(pred_opt, feed_dict, k=2, sess=sess)

                # update prog bar
                exp.add_metric_row({'tng loss': train_err, 'P@1': precission_at_1, 'P@2': precission_at_2})

            nb_batches_served += 1

            progbar.add(n=len(batch_context), values=[('train_err', train_err),
                                                      ('P@1', precission_at_1),
                                                      ('P@2', precission_at_2)])

        # ----------------------
        # END OF EPOCH PROCESSING
        # ----------------------
        # calculate the val loss
        print('\nepoch complete...\n')
        check_val_stats(model, pred_opt, udc_dataset, hparams, context_ph, utterance_ph, exp, sess, epoch)

        # save model
        save_model(saver=saver, hparams=hparams, sess=sess, epoch=epoch)

        # save exp data


def evaluate_recall(y, y_test, k=1):
    num_examples = float(len(y))
    num_correct = 0
    for predictions, label in zip(y, y_test):
        if label in predictions[:k]:
            num_correct += 1
    return num_correct / num_examples


def test_precision_at_k(pred_opt, feed_dict, k, sess):
    sims = pred_opt.eval(session=sess, feed_dict=feed_dict)
    labels = range(0, len(sims))
    for i, pred_vector in enumerate(sims):
        sims[i] = [i[0] for i in sorted(enumerate(pred_vector), key=lambda x: x[1])][::-1]

    recall_score = evaluate_recall(sims, labels, k)
    return recall_score


def new_evaluate_recall(y, k=1):
    num_correct = 0
    if 0 in y[0][:k]:
        num_correct += 1
    return num_correct


def new_test_precision_at_k(pred_opt, feed_dict, k, sess):
    sims = pred_opt.eval(session=sess, feed_dict=feed_dict)
    for i, pred_vector in enumerate(sims):
        sims[i] = [i[0] for i in sorted(enumerate(pred_vector), key=lambda x: x[1])][::-1]

    recall_score = new_evaluate_recall(sims, k)
    return recall_score


def get_optimizer(hparams, minimize):
    opt = None
    name = hparams.optimizer_name
    if name == 'sgd':
        opt = tf.train.GradientDescentOptimizer(learning_rate=hparams.lr_1).minimize(minimize)
    if name == 'adam':
        opt = tf.train.AdamOptimizer(learning_rate=hparams.lr_1).minimize(minimize)
    if name == 'rmsprop':
        opt = tf.train.RMSPropOptimizer(learning_rate=hparams.lr_1).minimize(minimize)

    return opt


def save_model(saver, hparams, sess, epoch):
    print('saving model...')

    # create path if not around
    model_save_path = hparams.model_save_dir + '/{}/epoch_{}'.format(hparams.exp_name, epoch)
    if not os.path.isdir(model_save_path):
        os.makedirs(model_save_path)

    model_name = '{}/model.cpkt'.format(model_save_path)

    save_path = saver.save(sess, model_name)
    print('model saved at', save_path, '\n\n')


def check_val_stats(model, pred_opt, data, hparams, X_ph, Y_ph, exp, sess, epoch):
    """
    Runs through validation data to check the overall mean loss
    :param model:
    :param data:
    :param hparams:
    :param X_ph:
    :param Y_ph:
    :param exp:
    :param sess:
    :param epoch:
    :return:
    """
    print('checking val loss...')
    max_val_batches = 100
    val_gen = data.val_generator(batch_size=hparams.batch_size, max_epochs=100)

    overall_err = []
    overall_p_1 = []
    overall_p_2 = []
    progbar = Progbar(target=max_val_batches, width=50)
    for batch_nb in range(max_val_batches):
        batch_X, batch_Y = next(val_gen)
        if len(batch_X) == 0:
            continue

        # aggregate data
        feed_dict = {
            X_ph: batch_X,
            Y_ph: batch_Y
        }
        sims = pred_opt.eval(session=sess, feed_dict=feed_dict)
        file = open("result.txt", "a")
        for ban, paras in enumerate(zip(batch_X, batch_Y)):
            pred_num = [i[0] for i in sorted(enumerate(sims[ban]), key=lambda x: x[1])][::-1][0]
            file.write("Question \n")
            file.writelines(paras[0] + "\n")
            file.writelines("\n")
            file.write("Right Answer \n")
            file.writelines(paras[1] + "\n")
            file.writelines("\n")
            file.write("Predicted Answer \n")
            file.writelines(batch_Y[pred_num] + "\n")
            file.writelines("*************************************\n")

        # calculate metrics
        val_err = model.eval(session=sess, feed_dict=feed_dict)
        precission_at_1 = test_precision_at_k(pred_opt, feed_dict, k=1, sess=sess)
        precission_at_2 = test_precision_at_k(pred_opt, feed_dict, k=2, sess=sess)

        # track metrics for means
        overall_err.append(val_err)
        overall_p_1.append(precission_at_1)
        overall_p_2.append(precission_at_2)

        # update exp and progbar
        exp.add_metric_row({'val loss': val_err, 'val P@1': precission_at_1, 'val P@2': precission_at_2})
        progbar.add(n=1)

    # log and save val metrics
    overall_val_mean_err = np.asarray(overall_err).mean()
    overall_p_1_mean = np.asarray(overall_p_1).mean()
    overall_p_2_mean = np.asarray(overall_p_2).mean()
    exp.add_metric_row({'epoch_mean_err': overall_val_mean_err,
                        'epoch_P@1_mean': overall_p_1_mean,
                        'epoch_P@2_mean': overall_p_2_mean,
                        'epoch': epoch + 1})

    print('\nval loss: ', overall_val_mean_err,
          'epoch_P@1_mean: ', overall_p_1_mean,
          'epoch_P@2_mean: ', overall_p_2_mean)
    print('-' * 100)
