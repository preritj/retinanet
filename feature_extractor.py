"""NASNet RetinaNet feature extraction
author : Prerit Jaiswal
This code is largely an adaptation of Tensorflow object detection
that uses NASNet-A-large backbone with Faster RCNN detector [2].
This code instead uses NASNet-mobile backbone with RetinaNet detector.
Pre-trained NASNet-A-mobile backbone taken from [1]

Paper refs:
[1] NASNet : https://arxiv.org/abs/1707.07012
[2] RetinaNet : https://arxiv.org/abs/1708.02002

Code ref:
[1] https://github.com/tensorflow/models/blob/master/research/slim/nets/nasnet
[2] https://github.com/tensorflow/models/blob/master/research/object_detection/models/faster_rcnn_nas_feature_extractor.py"""


import tensorflow as tf
from slim.nets.nasnet import nasnet
from slim.nets.nasnet import nasnet_utils
import retinanet_meta_arch

arg_scope = tf.contrib.framework.arg_scope
slim = tf.contrib.slim


def nasnet_mobile_arg_scope_for_detection(is_batch_norm_training=False):
    """Defines the default arg scope for the NASNet-A Mobile for object detection.
    This provides a small edit to switch batch norm training on and off.
    Args:
      is_batch_norm_training: Boolean indicating whether to train with batch norm.
    Returns:
      An `arg_scope` to use for the NASNet Mobile Model.
    """
    imagenet_scope = nasnet.nasnet_mobile_arg_scope()
    with arg_scope(imagenet_scope):
        with arg_scope([slim.batch_norm], is_training=is_batch_norm_training) as sc:
            return sc


# Note: This is largely a copy of _build_nasnet_base inside nasnet.py but
# with special edits to remove instantiation of the stem and the special
# ability to receive as input a pair of hidden states.
def _build_nasnet_base(hidden_previous,
                       hidden,
                       normal_cell,
                       reduction_cell,
                       hparams,
                       true_cell_num,
                       start_cell_num):
    """Constructs a NASNet image model."""

    # Find where to place the reduction cells or stride normal cells
    reduction_indices = nasnet_utils.calc_reduction_layers(
        hparams.num_cells, hparams.num_reduction_layers)

    # Note: The None is prepended to match the behavior of _imagenet_stem()
    cell_outputs = [None, hidden_previous, hidden]
    net = hidden

    # NOTE: In the nasnet.py code, filter_scaling starts at 1.0. We instead
    # start at 2.0 because 1 reduction cell has been created which would
    # update the filter_scaling to 2.0.
    filter_scaling = 2.0

    # Run the cells
    for cell_num in range(start_cell_num, hparams.num_cells):
        stride = 1
        if hparams.skip_reduction_layer_input:
            prev_layer = cell_outputs[-2]
        if cell_num in reduction_indices:
            filter_scaling *= hparams.filter_scaling_rate
            net = reduction_cell(
                net,
                scope='reduction_cell_{}'.format(reduction_indices.index(cell_num)),
                filter_scaling=filter_scaling,
                stride=2,
                prev_layer=cell_outputs[-2],
                cell_num=true_cell_num)
            true_cell_num += 1
            cell_outputs.append(net)
        if not hparams.skip_reduction_layer_input:
            prev_layer = cell_outputs[-2]
        net = normal_cell(
            net,
            scope='cell_{}'.format(cell_num),
            filter_scaling=filter_scaling,
            stride=stride,
            prev_layer=prev_layer,
            cell_num=true_cell_num)
        true_cell_num += 1
        cell_outputs.append(net)
    return net


def build_custom_nasnet(images, is_training=True):
    """Build custom NASNet Mobile model"""
    hparams = nasnet._mobile_imagenet_config()
    # Calculate the total number of cells in the network
    # -- Add 2 for the reduction cells.
    total_num_cells = hparams.num_cells + 2
    # -- And add 2 for the stem cells for ImageNet training.
    total_num_cells += 2

    normal_cell = nasnet_utils.NasNetANormalCell(
        hparams.num_conv_filters, hparams.drop_path_keep_prob,
        total_num_cells, hparams.total_training_steps)
    reduction_cell = nasnet_utils.NasNetAReductionCell(
        hparams.num_conv_filters, hparams.drop_path_keep_prob,
        total_num_cells, hparams.total_training_steps)

    net, end_points = nasnet.build_nasnet_mobile(
        images, num_classes=None,
        is_training=is_training,
        final_endpoint='Cell_11')

    # hidden_previous = end_points['Cell_6']
    # hidden = end_points['Cell_7']
    # start_cell_num = 8
    # true_cell_num = 11
    # net = _build_nasnet_base(hidden_previous,
    #                          hidden,
    #                          normal_cell=normal_cell,
    #                          reduction_cell=reduction_cell,
    #                          hparams=hparams,
    #                          true_cell_num=true_cell_num,
    #                          start_cell_num=start_cell_num)
    # end_points['Reduction_Cell_1'] = net
    return net, end_points


class RetinaNetNASFeatureExtractor(retinanet_meta_arch.RetinaNetFeatureExtractor):
    """RetinaNet with NASNet-A feature extractor implementation"""
    def __init__(self,
                 is_training,
                 batch_norm_trainable=False,
                 reuse_weights=None,
                 weight_decay=0.0):
        """Constructor.
        Args:
          is_training: See base class.
          batch_norm_trainable: See base class.
          reuse_weights: See base class.
          weight_decay: See base class.
        """
        super(RetinaNetNASFeatureExtractor, self).__init__(
            is_training, batch_norm_trainable, reuse_weights, weight_decay)

    def preprocess(self, resized_inputs):
        """RetinaNet with NAS preprocessing.
        Maps pixel values to the range [-1, 1].
        Args:
          resized_inputs: A [batch, height_in, width_in, channels] float32 tensor
            representing a batch of images with values between 0 and 255.0.
        Returns:
          preprocessed_inputs: A [batch, height_out, width_out, channels] float32
            tensor representing a batch of images.
        """
        return (2.0 / 255.0) * resized_inputs - 1.0

    def _extract_proposal_features(self, preprocessed_inputs, scope):
        """Extracts RPN features.
        Extracts features using the first half of the NASNet network.
        We construct the network in `align_feature_maps=True` mode, which means
        that all VALID paddings in the network are changed to SAME padding so that
        the feature maps are aligned.
        Args:
          preprocessed_inputs: A [batch, height, width, channels] float32 tensor
            representing a batch of images.
          scope: A scope name.
        Returns:
          rpn_feature_map: A dictionary with pyramid layers as keys and
          tensors with shape [batch, height, width, depth] as values
        Raises:
          ValueError: If the created network is missing the required activation.
        """
        del scope

        if len(preprocessed_inputs.get_shape().as_list()) != 4:
            raise ValueError('`preprocessed_inputs` must be 4 dimensional, got a '
                             'tensor of shape %s' % preprocessed_inputs.get_shape())

        with slim.arg_scope(nasnet_mobile_arg_scope_for_detection(
                is_batch_norm_training=self._train_batch_norm)):
            with arg_scope([slim.conv2d,
                            slim.batch_norm,
                            slim.separable_conv2d],
                           reuse=self._reuse_weights):
                _, end_points = build_custom_nasnet(
                    preprocessed_inputs, is_training=self._is_training)

        pyramid_layers = ['Cell_3', 'Cell_7', 'Cell_11']
        rpn_feature_map = {l: end_points[l] for l in pyramid_layers}

        # nasnet.py does not maintain the batch size in the first dimension.
        # This work around permits us retaining the batch for below.
        for l in pyramid_layers:
            batch = preprocessed_inputs.get_shape().as_list()[0]
            shape_without_batch = rpn_feature_map[l].get_shape().as_list()[1:]
            rpn_feature_map_shape = [batch] + shape_without_batch
            rpn_feature_map[l].set_shape(rpn_feature_map_shape)

        return rpn_feature_map

    def restore_from_classification_checkpoint_fn(self, feature_extractor_scope):
        """Returns a map of variables to load from a foreign checkpoint.
        Args:
          feature_extractor_scope: A scope name for the feature extractor.
        Returns:
          A dict mapping variable names (to load from a checkpoint) to variables in
          the model graph.
        """
        variables_to_restore = {}
        for variable in tf.global_variables():
            if variable.op.name.startswith(feature_extractor_scope):
                var_name = variable.op.name.replace(
                    feature_extractor_scope + '/', '')
                var_name += '/ExponentialMovingAverage'
                variables_to_restore[var_name] = variable
        return variables_to_restore