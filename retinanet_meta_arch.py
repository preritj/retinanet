import tensorflow as tf
from abc import abstractmethod
from object_detection.anchor_generators import multiscale_grid_anchor_generator
from object_detection.meta_architectures import ssd_meta_arch
from object_detection.core import losses

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
          scope: feature extractor scope
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


class RetinaNetMetaArch(ssd_meta_arch.SSDMetaArch):
    """Retina net Meta-architecture definition."""

    def __init__(self,
                 is_training,
                 anchor_generator,
                 box_predictor,
                 box_coder,
                 feature_extractor,
                 matcher,
                 region_similarity_calculator,
                 encode_background_as_zeros,
                 negative_class_weight,
                 image_resizer_fn,
                 non_max_suppression_fn,
                 score_conversion_fn,
                 classification_loss,
                 localization_loss,
                 classification_loss_weight,
                 localization_loss_weight,
                 normalize_loss_by_num_matches,
                 hard_example_miner,
                 add_summaries=True,
                 normalize_loc_loss_by_codesize=False):
        """RetinaNetMetaArch Constructor.
        Args:
          is_training: A boolean indicating whether the training version of the
            computation graph should be constructed.
          anchor_generator: an anchor_generator.AnchorGenerator object.
          box_predictor: a box_predictor.BoxPredictor object.
          box_coder: a box_coder.BoxCoder object.
          feature_extractor: a RetinaNetFeatureExtractor object.
          matcher: a matcher.Matcher object.
          region_similarity_calculator: a
            region_similarity_calculator.RegionSimilarityCalculator object.
          encode_background_as_zeros: boolean determining whether background
            targets are to be encoded as an all zeros vector or a one-hot
            vector (where background is the 0th class).
          negative_class_weight: Weight for confidence loss of negative anchors.
          image_resizer_fn: a callable for image resizing.  This callable always
            takes a rank-3 image tensor (corresponding to a single image) and
            returns a rank-3 image tensor, possibly with new spatial dimensions and
            a 1-D tensor of shape [3] indicating shape of true image within
            the resized image tensor as the resized image tensor could be padded.
            See builders/image_resizer_builder.py.
          non_max_suppression_fn: batch_multiclass_non_max_suppression
            callable that takes `boxes`, `scores` and optional `clip_window`
            inputs (with all other inputs already set) and returns a dictionary
            hold tensors with keys: `detection_boxes`, `detection_scores`,
            `detection_classes` and `num_detections`. See `post_processing.
            batch_multiclass_non_max_suppression` for the type and shape of these
            tensors.
          score_conversion_fn: callable elementwise nonlinearity (that takes tensors
            as inputs and returns tensors).  This is usually used to convert logits
            to probabilities.
          classification_loss: an object_detection.core.losses.Loss object.
          localization_loss: a object_detection.core.losses.Loss object.
          classification_loss_weight: float
          localization_loss_weight: float
          normalize_loss_by_num_matches: boolean
          hard_example_miner: a losses.HardExampleMiner object (can be None)
          add_summaries: boolean (default: True) controlling whether summary ops
            should be added to tensorflow graph.
          normalize_loc_loss_by_codesize: whether to normalize localization loss
            by code size of the box encoder.

        Raises:
          ValueError: If feature_extractor is not of type
            RetinaNetFeatureExtractor.
          ValueError: If anchor_generator is not of type
            grid_anchor_generator.GridAnchorGenerator.
          ValueError: If box_predictor is not of type
            box_predictor.WeightSharedConvolutionalBoxPredictor
        """
        super(RetinaNetMetaArch, self).__init__(
            is_training,
            anchor_generator,
            box_predictor,
            box_coder,
            feature_extractor,
            matcher,
            region_similarity_calculator,
            encode_background_as_zeros,
            negative_class_weight,
            image_resizer_fn,
            non_max_suppression_fn,
            score_conversion_fn,
            classification_loss,
            localization_loss,
            classification_loss_weight,
            localization_loss_weight,
            normalize_loss_by_num_matches,
            hard_example_miner,
            add_summaries=add_summaries,
            normalize_loc_loss_by_codesize=normalize_loc_loss_by_codesize)

        if not isinstance(feature_extractor, RetinaNetFeatureExtractor):
            raise ValueError('feature_extractor must be of type '
                             'RetinaNetFeatureExtractor')
        if not isinstance(anchor_generator,
                          multiscale_grid_anchor_generator.MultiscaleGridAnchorGenerator):
            raise ValueError('anchor_generator must be of type '
                             'multiscale_grid_anchor_generator.MultiscaleGridAnchorGenerator')
        if not isinstance(box_predictor,
                          box_predictor.WeightSharedConvolutionalBoxPredictor):
            raise ValueError('box_predictor must be of type  '
                             'box_predictor.WeightSharedConvolutionalBoxPredictor')
        if not isinstance(classification_loss,
                          losses.SigmoidFocalClassificationLoss):
            raise ValueError('classification_loss must be of type  '
                             'losses.SigmoidFocalClassificationLoss')


