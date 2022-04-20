#!/usr/bin/env python2

from __future__ import print_function
from collections import OrderedDict
import os.path
import hashlib
# import wget

import sys
import argparse
import onnx
from onnx import helper
from onnx import TensorProto
import numpy as np


class DarkNetParser(object):

    def __init__(self, supported_layers):
        """Initializes a DarkNetParser object.

        Keyword argument:
        supported_layers -- a string list of supported layers in DarkNet naming convention,
        parameters are only added to the class dictionary if a parsed layer is included.
        """

        # A list of YOLOv3 layers containing dictionaries with all layer
        # parameters:
        self.layer_configs = OrderedDict()
        self.supported_layers = supported_layers
        self.layer_counter = 0

    def parse_model_config(self, path):
        """Parses the yolo-v3 layer configuration file and returns module definitions"""

        file = open(path, 'r')
        lines = file.read().replace("\r", "\n").split('\n')
        lines = [x for x in lines if x and not x.startswith('#')]
        lines = [x.replace(" ", "") for x in lines]
        module_defs = OrderedDict()
        layernum = 0  # 0._fill(3)
        for line in lines:
            if line.startswith('['):  # This marks the start of a new block
                new_layer = {}
                layerprefix = str(layernum).zfill(3)
                layernum = layernum + 1
                type = line[1:-1].rstrip()
                new_layer['type'] = type
                module_defs[layerprefix + "_" + type] = new_layer
                if module_defs[module_defs.keys()[-1]]['type'] == 'convolutional':
                    module_defs[module_defs.keys()[-1]]['batch_normalize'] = 0
            else:
                try:
                    key, value = line.split('=')
                except Exception as e:
                    print(layernum)
                    print(line)
                    print(key)
                    print(value)
                    print(e)
                    break
                # finally:
                #     key = key.strip()
                #     value = value.strip()

                # parameter parser
                if key == 'layers':
                    layer_indexes = list()
                    for index in value.split(','):
                        layer_indexes.append(int(index))
                    param_value = layer_indexes
                elif isinstance(value, str) and not value.isalpha():
                    condition_param_value_positive = value.isdigit()
                    condition_param_value_negative = value[0] == '-' and \
                                                     value[1:].isdigit()
                    if condition_param_value_positive or condition_param_value_negative:
                        param_value = int(value)
                    else:
                        try:
                            param_value = float(value)
                        except ValueError:
                            param_value = [float(x) for x in value.split(",")]
                else:
                    param_value = str(value)

                module_defs[module_defs.keys()[-1]][key.rstrip()] = param_value
        self.layer_counter = layernum - 1
        self.layer_configs = module_defs
        del module_defs
        file.close()
        return self.layer_configs

    def parse_cfg_file(self, cfg_file_path):

        with open(cfg_file_path, 'rb') as cfg_file:
            remainder = cfg_file.read()
            while remainder is not None:
                layer_dict, layer_name, remainder = self._next_layer(remainder)
                if layer_dict is not None:
                    self.layer_configs[layer_name] = layer_dict
        return self.layer_configs

    def _next_layer(self, remainder):

        remainder = remainder.split('[', 1)
        if len(remainder) == 2:
            remainder = remainder[1]
        else:
            return None, None, None
        remainder = remainder.split(']', 1)
        if len(remainder) == 2:
            layer_type, remainder = remainder
        else:
            return None, None, None
        if remainder.replace(' ', '')[0] == '#':
            remainder = remainder.split('\n', 1)[1]

        layer_param_block, remainder = remainder.split('\n\n', 1)
        layer_param_lines = layer_param_block.split('\n')[1:]
        layer_name = str(self.layer_counter).zfill(3) + '_' + layer_type
        layer_dict = dict(type=layer_type)
        if layer_type in self.supported_layers:
            for param_line in layer_param_lines:
                if param_line[0] == '#':
                    continue
                param_type, param_value = self._parse_params(param_line)
                layer_dict[param_type] = param_value
        self.layer_counter += 1
        return layer_dict, layer_name, remainder

    def _parse_params(self, param_line):

        param_line = param_line.replace(' ', '')
        param_type, param_value_raw = param_line.split('=')
        param_value = None
        if param_type == 'layers':
            layer_indexes = list()
            for index in param_value_raw.split(','):
                layer_indexes.append(int(index))
            param_value = layer_indexes
        elif isinstance(param_value_raw, str) and not param_value_raw.isalpha():
            condition_param_value_positive = param_value_raw.isdigit()
            condition_param_value_negative = param_value_raw[0] == '-' and \
                                             param_value_raw[1:].isdigit()
            if condition_param_value_positive or condition_param_value_negative:
                param_value = int(param_value_raw)
            else:
                param_value = float(param_value_raw)
        else:
            param_value = str(param_value_raw)
        return param_type, param_value


class MajorNodeSpecs(object):

    def __init__(self, name, channels):
        """ Initialize a MajorNodeSpecs object.
        Keyword arguments:
        name -- name of the ONNX node
        channels -- number of output channels of this node
        """
        self.name = name
        self.channels = channels
        self.created_onnx_node = False
        if name is not None and isinstance(channels, int) and channels > 0:
            self.created_onnx_node = True


class ConvParams(object):

    def __init__(self, node_name, batch_normalize, conv_weight_dims):

        self.node_name = node_name
        self.batch_normalize = batch_normalize
        assert len(conv_weight_dims) == 4
        self.conv_weight_dims = conv_weight_dims

    def generate_param_name(self, param_category, suffix):
        """Generates a name based on two string inputs,
        and checks if the combination is valid."""
        assert suffix
        assert param_category in ['bn', 'conv']
        assert (suffix in ['scale', 'mean', 'var', 'weights', 'bias'])
        if param_category == 'bn':
            assert self.batch_normalize
            assert suffix in ['scale', 'bias', 'mean', 'var']
        elif param_category == 'conv':
            assert suffix in ['weights', 'bias']
            if suffix == 'bias':
                assert not self.batch_normalize
        param_name = self.node_name + '_' + param_category + '_' + suffix
        return param_name


class ResizeParams(object):

    def __init__(self, node_name, value):
        self.node_name = node_name
        self.value = value

    def generate_param_name(self):
        """Generates the scale parameter name for the Resize node."""
        param_name = self.node_name + '_' + "scale"
        return param_name


class WeightLoader(object):

    def __init__(self, weights_file_path):
        """Initialized with a path to the YOLOv3 .weights file.

        Keyword argument:
        weights_file_path -- path to the weights file.
        """
        self.weights_file = self._open_weights_file(weights_file_path)

    def load_resize_scales(self, resize_params):
        """Returns the initializers with the value of the scale input
        tensor given by resize_params.

        Keyword argument:
        resize_params -- a ResizeParams object
        """
        initializer = list()
        inputs = list()
        name = resize_params.generate_param_name()
        shape = resize_params.value.shape
        data = resize_params.value
        scale_init = helper.make_tensor(
            name, TensorProto.FLOAT, shape, data)
        scale_input = helper.make_tensor_value_info(
            name, TensorProto.FLOAT, shape)
        initializer.append(scale_init)
        inputs.append(scale_input)
        return initializer, inputs

    def load_conv_weights(self, conv_params):
        """Returns the initializers with weights from the weights file and
        the input tensors of a convolutional layer for all corresponding ONNX nodes.

        Keyword argument:
        conv_params -- a ConvParams object
        """
        initializer = list()
        inputs = list()
        if conv_params.batch_normalize:
            bias_init, bias_input = self._create_param_tensors(
                conv_params, 'bn', 'bias')
            bn_scale_init, bn_scale_input = self._create_param_tensors(
                conv_params, 'bn', 'scale')
            bn_mean_init, bn_mean_input = self._create_param_tensors(
                conv_params, 'bn', 'mean')
            bn_var_init, bn_var_input = self._create_param_tensors(
                conv_params, 'bn', 'var')
            initializer.extend(
                [bn_scale_init, bias_init, bn_mean_init, bn_var_init])
            inputs.extend([bn_scale_input, bias_input,
                           bn_mean_input, bn_var_input])
        else:
            bias_init, bias_input = self._create_param_tensors(
                conv_params, 'conv', 'bias')
            initializer.append(bias_init)
            inputs.append(bias_input)
        conv_init, conv_input = self._create_param_tensors(
            conv_params, 'conv', 'weights')
        initializer.append(conv_init)
        inputs.append(conv_input)
        return initializer, inputs

    def _open_weights_file(self, weights_file_path):
        """Opens a YOLOv3 DarkNet file stream and skips the header.

        Keyword argument:
        weights_file_path -- path to the weights file.
        """
        weights_file = open(weights_file_path, 'rb')
        length_header = 5
        np.ndarray(
            shape=(length_header,), dtype='int32', buffer=weights_file.read(
                length_header * 4))
        return weights_file

    def _create_param_tensors(self, conv_params, param_category, suffix):
        """Creates the initializers with weights from the weights file together with
        the input tensors.

        Keyword arguments:
        conv_params -- a ConvParams object
        param_category -- the category of parameters to be created ('bn' or 'conv')
        suffix -- a string determining the sub-type of above param_category (e.g.,
        'weights' or 'bias')
        """
        param_name, param_data, param_data_shape = self._load_one_param_type(
            conv_params, param_category, suffix)

        initializer_tensor = helper.make_tensor(
            param_name, TensorProto.FLOAT, param_data_shape, param_data)
        input_tensor = helper.make_tensor_value_info(
            param_name, TensorProto.FLOAT, param_data_shape)
        return initializer_tensor, input_tensor

    def _load_one_param_type(self, conv_params, param_category, suffix):
        """Deserializes the weights from a file stream in the DarkNet order.

        Keyword arguments:
        conv_params -- a ConvParams object
        param_category -- the category of parameters to be created ('bn' or 'conv')
        suffix -- a string determining the sub-type of above param_category (e.g.,
        'weights' or 'bias')
        """
        param_name = conv_params.generate_param_name(param_category, suffix)
        param_shape = []
        channels_out, channels_in, filter_h, filter_w = conv_params.conv_weight_dims
        if param_category == 'bn':
            param_shape = [channels_out]
        elif param_category == 'conv':
            if suffix == 'weights':
                param_shape = [channels_out, channels_in, filter_h, filter_w]
            elif suffix == 'bias':
                param_shape = [channels_out]
        param_size = np.product(np.array(param_shape))
        param_data = np.ndarray(
            shape=param_shape,
            dtype='float32',
            buffer=self.weights_file.read(param_size * 4))
        param_data = param_data.flatten().astype(float)
        return param_name, param_data, param_shape


class GraphBuilderONNX(object):

    def __init__(self):
        """Initialize with all DarkNet default parameters used creating YOLOv3,
        and specify the output tensors as an OrderedDict for their output dimensions
        with their names as keys.

        Keyword argument:
        output_tensors -- the output tensors as an OrderedDict containing the keys'
        output dimensions
        """
        # self.output_tensors = output_tensors

        self._nodes = list()
        self.graph_def = None
        self.input_tensor = None
        self.epsilon_bn = 1e-5
        self.momentum_bn = 0.99
        self.alpha_lrelu = 0.1
        self.param_dict = OrderedDict()
        self.major_node_specs = list()
        self.batch_size = 1

        self.outputs = []
        self.yolo_cfg = {}
        self.iter = 0
        self.input_resolution = None

    def build_onnx_graph(
            self,
            layer_configs,
            weights_file_path,
            verbose=True):

        for layer_name in layer_configs.keys():
            layer_dict = layer_configs[layer_name]
            major_node_specs = self._make_onnx_node(layer_name, layer_dict)
            if major_node_specs.name is not None:
                self.major_node_specs.append(major_node_specs)

        inputs = [self.input_tensor]  # info

        weight_loader = WeightLoader(weights_file_path)
        initializer = list()
        # If a layer has parameters, add them to the initializer and input lists.
        for layer_name in self.param_dict.keys():
            _, layer_type = layer_name.split('_', 1)
            params = self.param_dict[layer_name]
            if layer_type == 'convolutional':
                initializer_layer, inputs_layer = weight_loader.load_conv_weights(
                    params)
                initializer.extend(initializer_layer)
                inputs.extend(inputs_layer)
            elif layer_type == "upsample":
                initializer_layer, inputs_layer = weight_loader.load_resize_scales(
                    params)
                initializer.extend(initializer_layer)
                inputs.extend(inputs_layer)
        del weight_loader

        self.graph_def = helper.make_graph(
            nodes=self._nodes,
            name='YOLOv3-608',
            inputs=inputs,
            outputs=self.outputs,
            initializer=initializer
        )

        if verbose:
            print(helper.printable_graph(self.graph_def))

        model_def = helper.make_model(self.graph_def,
                                      producer_name='NVIDIA TensorRT sample')
        return model_def

    def _make_onnx_node(self, layer_name, layer_dict):

        layer_type = layer_dict['type']
        if self.input_tensor is None:
            if layer_type == 'net':
                major_node_output_name, major_node_output_channels = self._make_input_tensor(
                    layer_name, layer_dict)
                major_node_specs = MajorNodeSpecs(major_node_output_name,
                                                  major_node_output_channels)
            else:
                raise ValueError('The first node has to be of type "net".')
        else:
            node_creators = dict()
            node_creators['convolutional'] = self._make_conv_node
            node_creators['shortcut'] = self._make_shortcut_node
            node_creators['route'] = self._make_route_node
            node_creators['upsample'] = self._make_upsample_node
            node_creators['yolo'] = self._make_output_tensor
            node_creators['maxpool'] = self._make_maxpool_node

            if layer_type in node_creators.keys():
                major_node_output_name, major_node_output_channels = \
                    node_creators[layer_type](layer_name, layer_dict)
                major_node_specs = MajorNodeSpecs(major_node_output_name,
                                                  major_node_output_channels)
            else:
                print(
                    'Layer of type %s not supported, skipping ONNX node generation.' %
                    layer_type)
                major_node_specs = MajorNodeSpecs(layer_name, None)

        return major_node_specs

    def _make_maxpool_node(self, layer_name, layer_dict):

        previous_node_specs = self.major_node_specs[-1]
        channels = previous_node_specs.channels
        name = previous_node_specs.name
        k = int(layer_dict["stride"])

        maxpool_node = helper.make_node(
            'MaxPool',
            inputs=[name],
            outputs=[layer_name],
            kernel_shape=[k, k],
            auto_pad='SAME_UPPER'  # SAME_LOWER
        )
        self._nodes.append(maxpool_node)

        return layer_name, channels

    def _make_output_tensor(self, layer_name, layer_dict):

        previous_node_specs = self.major_node_specs[-1]
        name = previous_node_specs.name
        previous_channels = previous_node_specs.channels

        output_shape = [self.batch_size, previous_channels,
                        (2 ** self.iter) * (self.input_resolution // 32),
                        (2 ** self.iter) * (self.input_resolution // 32)]

        output_tensor = helper.make_tensor_value_info(
            name, TensorProto.FLOAT, output_shape)
        """
        b = 1
        yolo_node = helper.make_node(
            'Liner',
            inputs=[name,b],
            outputs=[layer_name],
            name=layer_name
        )
        self._nodes.append(yolo_node)
        """
        # self.output_tensors.add([layer_name])
        self.outputs.append(output_tensor)
        self.yolo_cfg[layer_name] = layer_dict
        self.iter += 1
        return layer_name, previous_channels

    def _make_input_tensor(self, layer_name, layer_dict):
        """Create an ONNX input tensor from a 'net' layer and store the batch size.

        Keyword arguments:
        layer_name -- the layer's name (also the corresponding key in layer_configs)   "net"
        layer_dict -- a layer parameter dictionary (one element of layer_configs)
        """
        batch_size = layer_dict['batch']
        channels = layer_dict['channels']
        height = layer_dict['height']
        width = layer_dict['width']
        self.input_resolution = width

        self.batch_size = batch_size
        input_tensor = helper.make_tensor_value_info(
            str(layer_name), TensorProto.FLOAT, [
                batch_size, channels, height, width])
        self.input_tensor = input_tensor
        return layer_name, channels

    def _get_previous_node_specs(self, target_index=-1):
        """Get a previously generated ONNX node (skip those that were not generated).
        Target index can be passed for jumping to a specific index.

        Keyword arguments:
        target_index -- optional for jumping to a specific index (default: -1 for jumping
        to previous element)
        """
        previous_node = None
        for node in self.major_node_specs[target_index::-1]:
            if node.created_onnx_node:
                previous_node = node
                break
        assert previous_node is not None
        return previous_node

    def _make_conv_node(self, layer_name, layer_dict):
        """Create an ONNX Conv node with optional batch normalization and
        activation nodes.

        Keyword arguments:
        layer_name -- the layer's name (also the corresponding key in layer_configs)
        layer_dict -- a layer parameter dictionary (one element of layer_configs)
        """
        previous_node_specs = self._get_previous_node_specs()

        inputs = [previous_node_specs.name]
        previous_channels = previous_node_specs.channels

        kernel_size = layer_dict['size']
        stride = layer_dict['stride']
        filters = layer_dict['filters']
        batch_normalize = False
        if 'batch_normalize' in layer_dict.keys(
        ) and layer_dict['batch_normalize'] == 1:
            batch_normalize = True

        kernel_shape = [kernel_size, kernel_size]
        weights_shape = [filters, previous_channels] + kernel_shape
        conv_params = ConvParams(layer_name, batch_normalize, weights_shape)

        strides = [stride, stride]
        dilations = [1, 1]
        weights_name = conv_params.generate_param_name('conv', 'weights')
        inputs.append(weights_name)
        if not batch_normalize:
            bias_name = conv_params.generate_param_name('conv', 'bias')
            inputs.append(bias_name)

        conv_node = helper.make_node(
            'Conv',
            inputs=inputs,
            outputs=[layer_name],
            kernel_shape=kernel_shape,
            strides=strides,
            auto_pad='SAME_LOWER',
            dilations=dilations,
            name=layer_name
        )
        self._nodes.append(conv_node)

        inputs = [layer_name]
        layer_name_output = layer_name

        if batch_normalize:
            layer_name_bn = layer_name + '_bn'
            bn_param_suffixes = ['scale', 'bias', 'mean', 'var']
            for suffix in bn_param_suffixes:
                bn_param_name = conv_params.generate_param_name('bn', suffix)
                inputs.append(bn_param_name)
            batchnorm_node = helper.make_node(
                'BatchNormalization',
                inputs=inputs,
                outputs=[layer_name_bn],
                epsilon=self.epsilon_bn,
                momentum=self.momentum_bn,
                name=layer_name_bn
            )
            self._nodes.append(batchnorm_node)
            inputs = [layer_name_bn]
            layer_name_output = layer_name_bn

        if layer_dict['activation'] == 'leaky':
            layer_name_lrelu = layer_name + '_lrelu'

            lrelu_node = helper.make_node(
                'LeakyRelu',
                inputs=inputs,
                outputs=[layer_name_lrelu],
                name=layer_name_lrelu,
                alpha=self.alpha_lrelu
            )
            self._nodes.append(lrelu_node)
            inputs = [layer_name_lrelu]
            layer_name_output = layer_name_lrelu
        elif layer_dict['activation'] == 'linear':
            pass
        elif layer_dict['activation'] == 'mish':
            layer_name_Celu = layer_name + '_Celu'
            Celu_node = onnx.helper.make_node(
                'Selu',
                inputs=inputs,
                outputs=[layer_name_Celu],
                alpha=2.0,
                gamma=3.0
            )
            self._nodes.append(Celu_node)
            layer_name_output = layer_name_Celu
        else:
            print('Activation not supported.')

        self.param_dict[layer_name] = conv_params
        return layer_name_output, filters

    def _make_shortcut_node(self, layer_name, layer_dict):
        """Create an ONNX Add node with the shortcut properties from
        the DarkNet-based graph.

        Keyword arguments:
        layer_name -- the layer's name (also the corresponding key in layer_configs)
        layer_dict -- a layer parameter dictionary (one element of layer_configs)
        """
        shortcut_index = layer_dict['from']
        activation = layer_dict['activation']
        assert activation == 'linear'

        first_node_specs = self._get_previous_node_specs()
        second_node_specs = self._get_previous_node_specs(
            target_index=shortcut_index)
        assert first_node_specs.channels == second_node_specs.channels
        channels = first_node_specs.channels
        inputs = [first_node_specs.name, second_node_specs.name]
        shortcut_node = helper.make_node(
            'Add',
            inputs=inputs,
            outputs=[layer_name],
            name=layer_name,
        )
        self._nodes.append(shortcut_node)
        return layer_name, channels

    def _make_route_node(self, layer_name, layer_dict):
        """If the 'layers' parameter from the DarkNet configuration is only one index, continue
        node creation at the indicated (negative) index. Otherwise, create an ONNX Concat node
        with the route properties from the DarkNet-based graph.

        Keyword arguments:
        layer_name -- the layer's name (also the corresponding key in layer_configs)
        layer_dict -- a layer parameter dictionary (one element of layer_configs)
        """
        route_node_indexes = layer_dict['layers']
        if len(route_node_indexes) == 1:
            split_index = route_node_indexes[0]
            if split_index > 0:
                split_index += 1
            major_node_specs = self.major_node_specs[split_index]
            layer_name = major_node_specs.name
            channels = major_node_specs.channels
        else:
            inputs = list()
            channels = 0
            for index in route_node_indexes:
                if index > 0:
                    # Increment by one because we count the input as a node (DarkNet
                    # does not)
                    index += 1
                route_node_specs = self.major_node_specs[index]
                inputs.append(route_node_specs.name)
                channels += route_node_specs.channels
            assert inputs
            assert channels > 0

            route_node = helper.make_node(
                'Concat',
                axis=1,
                inputs=inputs,
                outputs=[layer_name],
                name=layer_name,
            )
            self._nodes.append(route_node)
        return layer_name, channels

    def _make_upsample_node(self, layer_name, layer_dict):
        """Create an ONNX Resize node with the properties from
        the DarkNet-based graph.

        Keyword arguments:
        layer_name -- the layer's name (also the corresponding key in layer_configs)
        layer_dict -- a layer parameter dictionary (one element of layer_configs)
        """
        resize_scale_factors = float(layer_dict['stride'])
        # Create the scale factor array with node parameters
        scales = np.array([1.0, 1.0, resize_scale_factors, resize_scale_factors]).astype(np.float32)
        previous_node_specs = self._get_previous_node_specs()
        inputs = [previous_node_specs.name]

        channels = previous_node_specs.channels
        assert channels > 0
        resize_params = ResizeParams(layer_name, scales)
        scales_name = resize_params.generate_param_name()
        inputs.append(scales_name)

        resize_node = helper.make_node(
            'Upsample',
            mode='nearest',
            inputs=inputs,
            outputs=[layer_name],
            name=layer_name,
        )
        self._nodes.append(resize_node)
        self.param_dict[layer_name] = resize_params
        return layer_name, channels


def parse_args():
    parser = argparse.ArgumentParser(description="YOLO conversion.")
    parser.add_argument("--cfg", type=str,
                        default='./weights/yolov4-3l2c-1.cfg',
                        help="The configuration file of yolov3.")
    parser.add_argument("--weight", type=str, default='./weights/yolov4-3l2c-1.weights',
                        help="The width or input image")
    args = parser.parse_args()
    return args


def main():
    """Run the DarkNet-to-ONNX conversion for YOLOv3-608."""
    # # Have to use python 2 due to hashlib compatibility
    if sys.version_info[0] > 2:
        raise Exception(
            "This script is only compatible with python2, please re-run this script with python2. The rest of this sample can be run with either version of python.")

    # Download the config for YOLOv3 if not present yet, and analyze the checksum:
    args = parse_args()
    cfg_file_path = args.cfg

    # These are the only layers DarkNetParser will extract parameters from. The three layers of
    # type 'yolo' are not parsed in detail because they are included in the post-processing later:
    supported_layers = ['net', 'convolutional', 'shortcut',
                        'route', 'upsample',"yolo","maxpool"]
    # ['convolutional', 'yolo', 'route', 'maxpool', 'upsample', 'shortcut', 'net']
    # Create a DarkNetParser object, and the use it to generate an OrderedDict with all
    # layer's configs from the cfg file:
    parser = DarkNetParser(supported_layers)
    layer_configs = parser.parse_model_config(cfg_file_path)
    for i in layer_configs.keys():
        print(i)
        if "route" in i:
            print(layer_configs[i]["layers"])

    # We do not need the parser anymore after we got layer_configs:
    del parser

    builder = GraphBuilderONNX()

    weights_file_path = args.weight

    # Now generate an ONNX graph with weights from the previously parsed layer configurations
    # and the weights file:
    yolov3_model_def = builder.build_onnx_graph(
        layer_configs=layer_configs,
        weights_file_path=weights_file_path,
        verbose=True)
    # Once we have the model definition, we do not need the builder anymore:
    del builder
    for i, input in enumerate(yolov3_model_def.graph.input[0:1]):
        d = input.type.tensor_type.shape.dim
        print('in_dim:')
        print(d)

    for i, output in enumerate(yolov3_model_def.graph.output):
        d = output.type.tensor_type.shape.dim
        print("out_dim:")
        print(d)
    # Perform a sanity check on the ONNX model definition:
    onnx.checker.check_model(yolov3_model_def)
    output_file_path = './weights/%s.onnx' % cfg_file_path.split("/")[-1].split(".")[0]
    onnx.save(yolov3_model_def, output_file_path)
    print("{} saved !".format(output_file_path))


if __name__ == '__main__':
    main()
