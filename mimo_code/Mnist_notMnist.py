fimport functools
import os
import time
from absl import app
from absl import flags
from absl import logging

import mnist_model  # ADDED
# REMOVED from experimental.mimo import cifar_model  # local file import
import robustness_metrics as rm
import tensorflow as tf
import tensorflow_datasets as tfds
import uncertainty_baselines as ub
import baselines.utils_new as utils  # ADDED this!
# from uncertainty_baselines.baselines.cifar import utils
import uncertainty_metrics as um
flags.DEFINE_integer('seed', 0, 'Random seed.')

flags.DEFINE_string(
    'NN_dir', '/home/jupyter/mnist/WRN28-2/M4', 'The directory where the model weights and '
                                'training/evaluation summaries are stored.')
# Accelerator flags.
flags.DEFINE_bool('use_gpu', True, 'Whether to run on GPU or otherwise TPU.')
flags.DEFINE_bool('use_bfloat16', False, 'Whether to use mixed precision.')
flags.DEFINE_integer('num_cores', 8, 'Number of TPU cores or number of GPUs.')
flags.DEFINE_string('tpu', None,
                    'Name of the TPU. Only used if use_gpu is False.')
FLAGS = flags.FLAGS


def main(argv):
    del argv  # unused arg

    tf.random.set_seed(FLAGS.seed)

    if FLAGS.use_gpu:
        logging.info('Use GPU')
        strategy = tf.distribute.MirroredStrategy()
    else:
        logging.info('Use TPU at %s',
                     FLAGS.tpu if FLAGS.tpu is not None else 'local')
        resolver = tf.distribute.cluster_resolver.TPUClusterResolver(tpu=FLAGS.tpu)
        tf.config.experimental_connect_to_cluster(resolver)
        tf.tpu.experimental.initialize_tpu_system(resolver)
        strategy = tf.distribute.TPUStrategy(resolver)

    ds_info = tfds.builder(FLAGS.dataset).info
    train_batch_size = FLAGS.per_core_batch_size * FLAGS.num_cores // FLAGS.batch_repetitions
    test_batch_size = FLAGS.per_core_batch_size * FLAGS.num_cores
    train_dataset_size = ds_info.splits['train'].num_examples
    steps_per_epoch = train_dataset_size // train_batch_size
    steps_per_eval = ds_info.splits['test'].num_examples // test_batch_size
    num_classes = ds_info.features['label'].num_classes
    
    if FLAGS.dataset == 'cifar10':
        dataset_builder_class = ub.datasets.Cifar10Dataset
    elif FLAGS.dataset == 'cifar10':
        dataset_builder_class = ub.datasets.Cifar100Dataset
    elif FLAGS.dataset == 'mnist':
        dataset_builder_class = ub.datasets.MnistDataset
    train_dataset_builder = dataset_builder_class(
        split=tfds.Split.TRAIN)
        #use_bfloat16=FLAGS.use_bfloat16)
    # train_dataset = train_dataset_builder.load(batch_size=train_batch_size)
    # train_dataset = strategy.experimental_distribute_dataset(train_dataset)
    clean_test_dataset_builder = dataset_builder_class(
        split=tfds.Split.TEST)
        #use_bfloat16=FLAGS.use_bfloat16)
    clean_test_dataset = clean_test_dataset_builder.load(
        batch_size=test_batch_size)
    test_datasets = {
        'clean': strategy.experimental_distribute_dataset(clean_test_dataset),
    }
    if FLAGS.corruptions_interval > 0:
        if FLAGS.dataset == 'cifar10':
            load_c_dataset = utils.load_cifar10_c
        elif FLAGS.dataset == 'cifar10':
            load_c_dataset = functools.partial(utils.load_cifar100_c,
                                               path=FLAGS.cifar100_c_path)
        elif FLAGS.dataset == 'mnist':
            load_c_dataset = utils.load_mnist_c

        corruption_types, max_intensity = utils.load_corrupted_test_info(
            FLAGS.dataset)
        for corruption in corruption_types:
            for intensity in range(1, max_intensity + 1):
                dataset = load_c_dataset(
                    corruption_name=corruption,
                    corruption_intensity=intensity,
                    batch_size=test_batch_size,
                    use_bfloat16=FLAGS.use_bfloat16)
                test_datasets['{0}_{1}'.format(corruption, intensity)] = (
                    strategy.experimental_distribute_dataset(dataset))

    if FLAGS.use_bfloat16:
        tf.keras.mixed_precision.set_global_policy('mixed_bfloat16')

    summary_writer = tf.summary.create_file_writer(
        os.path.join(FLAGS.output_dir, 'summaries'))
    print([FLAGS.ensemble_size] +
                        list(ds_info.features['image'].shape))
    with strategy.scope():
        logging.info('Building Keras model')
        model = mnist_model.wide_resnet(
            input_shape=[FLAGS.ensemble_size] +
                        list(ds_info.features['image'].shape),
            depth=28,
            width_multiplier=FLAGS.width_multiplier,
            num_classes=num_classes,
            ensemble_size=FLAGS.ensemble_size)
        
        #model.summary()
        logging.info('Model input shape: %s', model.input_shape)
        logging.info('Model output shape: %s', model.output_shape)
        logging.info('Model number of weights: %s', model.count_params())
        # Linearly scale learning rate and the decay epochs by vanilla settings.
        base_lr = FLAGS.base_learning_rate * train_batch_size / 128
        lr_decay_epochs = [(int(start_epoch_str) * FLAGS.train_epochs) // 200
                           for start_epoch_str in FLAGS.lr_decay_epochs]
        lr_schedule = ub.schedules.WarmUpPiecewiseConstantSchedule(
            steps_per_epoch,
            base_lr,
            decay_ratio=FLAGS.lr_decay_ratio,
            decay_epochs=lr_decay_epochs,
            warmup_epochs=FLAGS.lr_warmup_epochs)
        optimizer = tf.keras.optimizers.SGD(
            lr_schedule, momentum=0.9, nesterov=True)
        metrics = {
            'train/negative_log_likelihood': tf.keras.metrics.Mean(),
            'train/accuracy': tf.keras.metrics.SparseCategoricalAccuracy(),
            'train/loss': tf.keras.metrics.Mean(),
            'train/ece': um.ExpectedCalibrationError(num_bins=FLAGS.num_bins),
            'test/negative_log_likelihood': tf.keras.metrics.Mean(),
            'test/accuracy': tf.keras.metrics.SparseCategoricalAccuracy(),
            'test/ece': um.ExpectedCalibrationError(num_bins=FLAGS.num_bins),
            'test/diversity': rm.metrics.AveragePairwiseDiversity(),
        }
        if FLAGS.corruptions_interval > 0:
            corrupt_metrics = {}
            for intensity in range(1, max_intensity + 1):
                for corruption in corruption_types:
                    dataset_name = '{0}_{1}'.format(corruption, intensity)
                    corrupt_metrics['test/nll_{}'.format(dataset_name)] = (
                        tf.keras.metrics.Mean())
                    corrupt_metrics['test/accuracy_{}'.format(dataset_name)] = (
                        tf.keras.metrics.SparseCategoricalAccuracy())
                    corrupt_metrics['test/ece_{}'.format(dataset_name)] = (
                        um.ExpectedCalibrationError(num_bins=FLAGS.num_bins))

        for i in range(FLAGS.ensemble_size):
            metrics['test/nll_member_{}'.format(i)] = tf.keras.metrics.Mean()
            metrics['test/accuracy_member_{}'.format(i)] = (
                tf.keras.metrics.SparseCategoricalAccuracy())

        checkpoint = tf.train.Checkpoint(model=model, optimizer=optimizer)
        latest_checkpoint = tf.train.latest_checkpoint(FLAGS.output_dir)
        initial_epoch = 0
        if latest_checkpoint:
            # checkpoint.restore must be within a strategy.scope() so that optimizer
            # slot variables are mirrored.
            checkpoint.restore(latest_checkpoint)
            logging.info('Loaded checkpoint %s', latest_checkpoint)
            initial_epoch = optimizer.iterations.numpy() // steps_per_epoch