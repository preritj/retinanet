import tensorflow as tf
from abc import abstractmethod
from object_detection.core import model
from object_detection.utils import shape_utils
from object_detection.anchor_generators import multiscale_grid_anchor_generator
from object_detection.core import box_list_ops

slim = tf.contrib.slim


class RetinaNetFeatureExtractor(object):
    """RetinaNet Feature Extractor definition."""

    def __init__(self,
                 is_training,
                 batch_norm_trainable=False,
                 reuse_weights=None,
                 weight_decay=0.0,
                 feature_depth=256):
        """Constructor.
        Args:
          is_training: A boolean indicating whether the training version of the
            computation graph should be constructed.
          batch_norm_trainable: Whether to update batch norm parameters during
            training or not. When training with a relative large batch size
            (e.g. 8), it could be desirable to enable batch norm update.
          reuse_weights: Whether to reuse variables. Default is None.
          weight_decay: float weight decay for feature extractor (default: 0.0).
          feature_depth: Feature depth
        """
        self._is_training = is_training
        self._train_batch_norm = (batch_norm_trainable and is_training)
        self._reuse_weights = reuse_weights
        self._weight_decay = weight_decay
        self._feature_depth = feature_depth

    @abstractmethod
    def preprocess(self, resized_inputs):
        """Feature-extractor specific preprocessing (minus image resizing)."""
        pass

    @abstractmethod
    def extract_features(self, preprocessed_inputs):
        """Extracts features from preprocessed inputs.
        This function is responsible for extracting feature maps from preprocessed
        images (to be overridden).
        Args:
          preprocessed_inputs: a [batch, height, width, channels] float tensor
            representing a batch of images.
        Returns:
          feature_maps: a list of tensors where the ith tensor has shape
            [batch, height_i, width_i, depth_i]
        """
        pass

    @abstractmethod
    def restore_from_classification_checkpoint_fn(self, feature_extractor_scope):
        """Returns a map of variables to load from a foreign checkpoint.
        To be overridden
        """
        pass


class RetinaNetMetaArch(model.DetectionModel):
    """Retina net Meta-architecture definition."""

    def __init__(self,
                 is_training,
                 num_classes,
                 image_resizer_fn,
                 feature_extractor,
                 anchor_generator,
                 parallel_iterations=16):
        """RetinaNetMetaArch Constructor.
            Args:
              is_training: A boolean indicating whether the training version of the
                computation graph should be constructed.
              num_classes: Number of classes.  Note that num_classes *does not*
                include the background category, so if groundtruth labels take values
                in {0, 1, .., K-1}, num_classes=K (and not K+1, even though the
                assigned classification targets can range from {0,... K}).
              image_resizer_fn: A callable for image resizing.  This callable
                takes a rank-3 image tensor of shape [height, width, channels]
                (corresponding to a single image), an optional rank-3 instance mask
                tensor of shape [num_masks, height, width] and returns a resized rank-3
                image tensor, a resized mask tensor if one was provided in the input. In
                addition this callable must also return a 1-D tensor of the form
                [height, width, channels] containing the size of the true image, as the
                image resizer can perform zero padding. See protos/image_resizer.proto.
              feature_extractor: A RetinaNetFeatureExtractor object
              anchor_generator: An anchor_generator.AnchorGenerator object
                (note that currently we only support
                multiscale_grid_anchor_generator.MultiscaleGridAnchorGenerator objects)
              parallel_iterations: (Optional) The number of iterations allowed to run
                in parallel for calls to tf.map_fn.

            Raises:
              ValueError: If first_stage_anchor_generator is not of type
                grid_anchor_generator.GridAnchorGenerator.
        """
        super(RetinaNetMetaArch, self).__init__(num_classes=num_classes)
        if not isinstance(anchor_generator,
                          multiscale_grid_anchor_generator.MultiscaleGridAnchorGenerator):
            raise ValueError('first_stage_anchor_generator must be of type '
                             'multiscale_grid_anchor_generator.MultiscaleGridAnchorGenerator')
        self._is_training = is_training
        self._image_resizer_fn = image_resizer_fn
        # Needed for fine-tuning from classification checkpoints whose
        # variables do not have the feature extractor scope.
        self._feature_extractor = feature_extractor
        self._parallel_iterations = parallel_iterations
        self._anchor_generator = anchor_generator

    @property
    def feature_extractor_scope(self):
        return 'FeatureExtractor'

    def preprocess(self, inputs):
        """Feature-extractor specific preprocessing.
        See base class.
        For Retina net, we perform image resizing in the base class --- each
        class subclassing RetinaNetMetaArch is responsible for any additional
        preprocessing (e.g., scaling pixel values to be in [-1, 1]).
        Args:
          inputs: a [batch, height_in, width_in, channels] float tensor representing
            a batch of images with values between 0 and 255.0.
        Returns:
          preprocessed_inputs: a [batch, height_out, width_out, channels] float
            tensor representing a batch of images.
          true_image_shapes: int32 tensor of shape [batch, 3] where each row is
            of the form [height, width, channels] indicating the shapes
            of true images in the resized images, as resized images can be padded
            with zeros.
        Raises:
          ValueError: if inputs tensor does not have type tf.float32
        """
        if inputs.dtype is not tf.float32:
            raise ValueError('`preprocess` expects a tf.float32 tensor')
        with tf.name_scope('Preprocessor'):
            outputs = shape_utils.static_or_dynamic_map_fn(
                self._image_resizer_fn,
                elems=inputs,
                dtype=[tf.float32, tf.int32],
                parallel_iterations=self._parallel_iterations)
            resized_inputs = outputs[0]
            true_image_shapes = outputs[1]
            return (self._feature_extractor.preprocess(resized_inputs),
                    true_image_shapes)

    def _extract_rpn_feature_maps(self, preprocessed_inputs):
        """Extracts RPN features.
        This function extracts two feature maps: a feature map to be directly
        fed to a box predictor (to predict location and objectness scores for
        proposals) and a feature map from which to crop regions which will then
        be sent to the second stage box classifier.
        Args:
          preprocessed_inputs: a [batch, height, width, channels] image tensor.
        Returns:
          rpn_box_predictor_features: A 4-D float32 tensor with shape
            [batch, height, width, depth] to be used for predicting proposal boxes
            and corresponding objectness scores.
          rpn_features_to_crop: A 4-D float32 tensor with shape
            [batch, height, width, depth] representing image features to crop using
            the proposals boxes.
          anchors: A BoxList representing anchors (for the RPN) in
            absolute coordinates.
          image_shape: A 1-D tensor representing the input image shape.
        """
        image_shape = tf.shape(preprocessed_inputs)
        rpn_features = self._feature_extractor.extract_features(
            preprocessed_inputs, scope=self.feature_extractor_scope)
        feature_map_shape_list = []

        for layer, feat in rpn_features.items():
            feature_map_shape = tf.shape(rpn_features[layer])
            feature_map_shape_list.append((feature_map_shape[1],
                                           feature_map_shape[2]))

        anchors = self._anchor_generator.generate(feature_map_shape_list)


