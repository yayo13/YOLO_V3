#! /usr/bin/env python
# coding=utf-8

import core.common as common

def CSPDarknet53(input_data):
    input_data = common.convolutional(input_data, (3, 3,  3, 32), activate_type="mish")
    input_data = common.convolutional(input_data, (3, 3, 32, 64), downsample=True, activate="mish")
    input_data = common.cross_stage_partial(input_data, 64, 64, 32, 64, 1, "mish")

    input_data = common.convolutional(input_data, (3, 3, 64, 64, 128), downsample=True, activate_type="mish")
    input_data = common.cross_stage_partial(input_data, 128, 64, 64, 64, 2, "mish")

    input_data = common.convolutional(input_data, (3, 3, 128, 256), downsample=True, activate_type="mish")
    input_data = common.cross_stage_partial(input_data, 256, 128, 128, 128, 8, "mish")
    
    route_1    = input_data
    input_data = common.convolutional(input_data, (3, 3, 256, 512), downsample=True, activate_type="mish")
    input_data = common.cross_stage_partial(input_data, 512, 256, 256, 256, 8, "mish")

    route_2    = input_data
    input_data = common.convolutional(input_data, (3, 3, 512, 1024), downsample=True, activate_type="mish")
    input_data = common.cross_stage_partial(input_data, 1024, 512, 512, 512, 4, "mish")

    # neck
    input_data = common.convolutional(input_data, (1, 1, 1024, 512))
    input_data = common.convolutional(input_data, (3, 3, 512, 1024))
    input_data = common.convolutional(input_data, (1, 1, 1024, 512))

    input_data = common.spp_block(input_data, 512)
    input_data = common.convolutional(input_data, (1, 1, 2048, 512))
    input_data = common.convolutional(input_data, (3, 3, 512, 1024))
    input_data = common.convolutional(input_data, (1, 1, 1024, 512))

    return route_1, route_2, input_data