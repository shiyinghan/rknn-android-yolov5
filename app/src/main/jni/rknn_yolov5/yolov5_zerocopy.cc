// Copyright (c) 2023 by Rockchip Electronics Co., Ltd. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "yolov5_zerocopy.h"
#include "utils/common.h"
#include "utils/file_utils.h"
#include "utils/image_utils.h"

static void dump_tensor_attr(rknn_tensor_attr *attr) {
    LOGI("  index=%d, name=%s, n_dims=%d, dims=[%d, %d, %d, %d], n_elems=%d, size=%d, fmt=%s, type=%s, qnt_type=%s, "
         "zp=%d, scale=%f",
         attr->index, attr->name, attr->n_dims, attr->dims[0], attr->dims[1], attr->dims[2],
         attr->dims[3],
         attr->n_elems, attr->size, get_format_string(attr->fmt), get_type_string(attr->type),
         get_qnt_type_string(attr->qnt_type), attr->zp, attr->scale);
}

int init_yolov5_model_zerocopy(const char *model_path, rknn_app_context_t *app_ctx) {
    int ret;
    int model_len = 0;
    char *model;
    rknn_context ctx = 0;

    // Load RKNN Model
    model_len = read_data_from_file(model_path, &model);
    if (model == NULL) {
        LOGI("load_model fail!");
        return -1;
    }

    // 1.初始化模型
    ret = rknn_init(&ctx, model, model_len, 0, NULL);
    free(model);
    if (ret < 0) {
        LOGI("rknn_init fail! ret=%d", ret);
        return -1;
    }

    // 2.查询模型的输入输出属性

    // Get Model Input Output Number
    rknn_input_output_num io_num;
    ret = rknn_query(ctx, RKNN_QUERY_IN_OUT_NUM, &io_num, sizeof(io_num));
    if (ret != RKNN_SUCC) {
        LOGI("rknn_query fail! ret=%d", ret);
        return -1;
    }
    LOGI("model input num: %d, output num: %d", io_num.n_input, io_num.n_output);

    // 调用rknn_query接口，查询原始的输入tensor属性，输出的tensor属性，放到对应rknn_tensor_attr结构体对象
    // Get Model Input Info
    LOGI("input tensors:");
    rknn_tensor_attr input_attrs[io_num.n_input];
    memset(input_attrs, 0, sizeof(input_attrs));
    for (int i = 0; i < io_num.n_input; i++) {
        input_attrs[i].index = i;
        ret = rknn_query(ctx, RKNN_QUERY_INPUT_ATTR, &(input_attrs[i]), sizeof(rknn_tensor_attr));
        if (ret != RKNN_SUCC) {
            LOGI("rknn_query fail! ret=%d", ret);
            return -1;
        }
        dump_tensor_attr(&(input_attrs[i]));
    }

    // Get Model Output Info
    LOGI("output tensors:");
    rknn_tensor_attr output_attrs[io_num.n_output];
    memset(output_attrs, 0, sizeof(output_attrs));
    for (int i = 0; i < io_num.n_output; i++) {
        output_attrs[i].index = i;
        ret = rknn_query(ctx, RKNN_QUERY_OUTPUT_ATTR, &(output_attrs[i]), sizeof(rknn_tensor_attr));
        if (ret != RKNN_SUCC) {
            LOGI("rknn_query fail! ret=%d", ret);
            return -1;
        }
        dump_tensor_attr(&(output_attrs[i]));
    }

    // Set to context
    app_ctx->rknn_ctx = ctx;

    // TODO
    if (output_attrs[0].qnt_type == RKNN_TENSOR_QNT_AFFINE_ASYMMETRIC &&
        output_attrs[0].type != RKNN_TENSOR_FLOAT16) {
        app_ctx->is_quant = true;
    } else {
        app_ctx->is_quant = false;
    }

    app_ctx->io_num = io_num;
    app_ctx->input_attrs = (rknn_tensor_attr *) malloc(io_num.n_input * sizeof(rknn_tensor_attr));
    memcpy(app_ctx->input_attrs, input_attrs, io_num.n_input * sizeof(rknn_tensor_attr));
    app_ctx->output_attrs = (rknn_tensor_attr *) malloc(io_num.n_output * sizeof(rknn_tensor_attr));
    memcpy(app_ctx->output_attrs, output_attrs, io_num.n_output * sizeof(rknn_tensor_attr));

    if (input_attrs[0].fmt == RKNN_TENSOR_NCHW) {
        LOGI("model is NCHW input fmt");
        app_ctx->model_channel = input_attrs[0].dims[1];
        app_ctx->model_height = input_attrs[0].dims[2];
        app_ctx->model_width = input_attrs[0].dims[3];
    } else {
        LOGI("model is NHWC input fmt");
        app_ctx->model_height = input_attrs[0].dims[1];
        app_ctx->model_width = input_attrs[0].dims[2];
        app_ctx->model_channel = input_attrs[0].dims[3];
    }
    LOGI("model input height=%d, width=%d, channel=%d",
         app_ctx->model_height, app_ctx->model_width, app_ctx->model_channel);

    return 0;
}

int release_yolov5_model_zerocopy(rknn_app_context_t *app_ctx) {
    if (app_ctx->rknn_ctx != 0) {
        // 9.销毁 RKNN
        rknn_destroy(app_ctx->rknn_ctx);
        app_ctx->rknn_ctx = 0;
    }
    if (app_ctx->input_attrs != NULL) {
        free(app_ctx->input_attrs);
        app_ctx->input_attrs = NULL;
    }
    if (app_ctx->output_attrs != NULL) {
        free(app_ctx->output_attrs);
        app_ctx->output_attrs = NULL;
    }
    return 0;
}

static void copyDataToTensorMemory(uint8_t *data, rknn_tensor_mem *tensor_mem,
                                   rknn_tensor_attr *tensor_attr) {
    // Copy data to tensor memory
    int width = tensor_attr->dims[2];
    int stride = tensor_attr->w_stride;

    if (width == stride) {
        memcpy(tensor_mem->virt_addr, data,
               width * tensor_attr->dims[1] * tensor_attr->dims[3]);
    } else {
        int height = tensor_attr->dims[1];
        int channel = tensor_attr->dims[3];
        // copy from src to dst with stride
        uint8_t *src_ptr = data;
        uint8_t *dst_ptr = (uint8_t *) tensor_mem->virt_addr;
        // width-channel elements
        int src_wc_elems = width * channel;
        int dst_wc_elems = stride * channel;
        for (int h = 0; h < height; ++h) {
            memcpy(dst_ptr, src_ptr, src_wc_elems);
            src_ptr += src_wc_elems;
            dst_ptr += dst_wc_elems;
        }
    }
}

int inference_yolov5_model_zerocopy(rknn_app_context_t *app_ctx, image_buffer_t *img,
                                    object_detect_result_list *od_results) {
    int ret;
    image_buffer_t dst_img;
    letterbox_t letter_box;
//    rknn_input inputs[app_ctx->io_num.n_input];
//    rknn_output outputs[app_ctx->io_num.n_output];
    void *output_data[app_ctx->io_num.n_output];

    // 定义输入输出内存
    rknn_tensor_mem *input_mems[app_ctx->io_num.n_input];
    rknn_tensor_mem *output_mems[app_ctx->io_num.n_output];

    const float nms_threshold = NMS_THRESH;      // 默认的NMS阈值
    const float box_conf_threshold = BOX_THRESH; // 默认的置信度阈值
    int bg_color = 114;

    if ((!app_ctx) || !(img) || (!od_results)) {
        return -1;
    }

    memset(od_results, 0x00, sizeof(*od_results));
    memset(&letter_box, 0, sizeof(letterbox_t));
    memset(&dst_img, 0, sizeof(image_buffer_t));
//    memset(inputs, 0, sizeof(inputs));
//    memset(outputs, 0, sizeof(outputs));

    // 设置输入输出类型
    // 初始化输入输出内存
    for (int i = 0; i < app_ctx->io_num.n_input; i++) {
        //这里有个有意思的现象，这里模型输入的type格式默认为RKNN_TENSOR_INT8，这就意味着，
        //归一化及量化操作要在CPU侧进行处理，也就是读完数据后就进行操作，而如果
        //设置为RKNN_TENSOR_UINT8则归一化及量化操作都放到了NPU上进行。
        // default input type is int8 (normalize and quantize need compute in outside)
        // if set uint8, will fuse normalize and quantize to npu
        app_ctx->input_attrs[i].type = RKNN_TENSOR_UINT8;
        // default fmt is NHWC, npu only support NHWC in zero copy mode
        app_ctx->input_attrs[i].fmt = RKNN_TENSOR_NHWC;

        input_mems[i] = rknn_create_mem(app_ctx->rknn_ctx,
                                        app_ctx->input_attrs[i].size_with_stride);
    }

    for (int i = 0; i < app_ctx->io_num.n_output; i++) {
        //这里设置float32，反量化操作在NPU内进行。
        // default output type is depend on model, this require float32 to compute top5
        app_ctx->output_attrs[i].type = RKNN_TENSOR_FLOAT32;

        // allocate float32 output tensor
        int output_size = app_ctx->output_attrs[i].n_elems * sizeof(float);
        output_mems[i] = rknn_create_mem(app_ctx->rknn_ctx,
                                         output_size);
    }

    //强制输出float32，在NPU内进行反量化，后处理的nms会用到
    app_ctx->is_quant = false;

    // Pre Process
    dst_img.width = app_ctx->model_width;
    dst_img.height = app_ctx->model_height;
    dst_img.format = IMAGE_FORMAT_RGB888;
    dst_img.size = get_image_size(&dst_img);
    dst_img.virt_addr = (unsigned char *) malloc(dst_img.size);
    if (dst_img.virt_addr == NULL) {
        LOGI("malloc buffer size:%d fail!", dst_img.size);
        return -1;
    }

    // 3.对输入进行前处理
    // letterbox操作：在对图片进行resize时，保持原图的长宽比进行等比例缩放，当长边 resize 到需要的长度时，短边剩下的部分采用灰色填充。
    // letterbox
    ret = convert_image_with_letterbox(img, &dst_img, &letter_box, bg_color);
    if (ret < 0) {
        LOGI("convert_image_with_letterbox fail! ret=%d", ret);
        return -1;
    }

    // Set Input Data
//    inputs[0].index = 0;
//    inputs[0].type = RKNN_TENSOR_UINT8;
//    inputs[0].fmt = RKNN_TENSOR_NHWC;
//    inputs[0].size = app_ctx->model_width * app_ctx->model_height * app_ctx->model_channel;
//    inputs[0].buf = dst_img.virt_addr;

    // 4.设置输入数据
    for (int i = 0; i < app_ctx->io_num.n_input; i++) {
        // Copy input data to input tensor memory
        copyDataToTensorMemory(dst_img.virt_addr, input_mems[i], &app_ctx->input_attrs[i]);

        // rknn_set_io_mem函数设置输入tensor内存，NPU使用
        rknn_set_io_mem(app_ctx->rknn_ctx, input_mems[i], &app_ctx->input_attrs[i]);
    }

    for (int i = 0; i < app_ctx->io_num.n_output; i++) {
        // rknn_set_io_mem函数设置输出tensor内存，NPU使用
        rknn_set_io_mem(app_ctx->rknn_ctx, output_mems[i], &app_ctx->output_attrs[i]);
    }

    // 4.设置输入数据
//    ret = rknn_inputs_set(app_ctx->rknn_ctx, app_ctx->io_num.n_input, inputs);
//    if (ret < 0) {
//        LOGI("rknn_input_set fail! ret=%d", ret);
//        return -1;
//    }

    // 5.进行模型推理
    // Run
    LOGI("rknn_run");
    int64_t start_us = getCurrentTimeUs();
    ret = rknn_run(app_ctx->rknn_ctx, nullptr);
    int64_t elapse_us = getCurrentTimeUs() - start_us;
    LOGI("zerocopy Elapse Time = %.2fms, FPS = %.2f\n", elapse_us / 1000.f,
         1000.f * 1000.f / elapse_us);
    if (ret < 0) {
        LOGI("rknn_run fail! ret=%d", ret);
        return -1;
    }

    // Get Output
//    memset(outputs, 0, sizeof(outputs));
//    for (int i = 0; i < app_ctx->io_num.n_output; i++) {
//        outputs[i].index = i;
//        outputs[i].want_float = (!app_ctx->is_quant);
//    }
    // 6.获取推理结果数据
//    ret = rknn_outputs_get(app_ctx->rknn_ctx, app_ctx->io_num.n_output, outputs, NULL);
//    if (ret < 0) {
//        LOGI("rknn_outputs_get fail! ret=%d", ret);
//        goto out;
//    }

    for (int i = 0; i < app_ctx->io_num.n_output; i++) {
        output_data[i] = output_mems[i]->virt_addr;
    }

    // 7.对输出进行后处理
    // Post Process
    LOGI("post_process");
    post_process(app_ctx, output_data, &letter_box, box_conf_threshold, nms_threshold, od_results);

    // 8.释放输出数据内存
//    // Remeber to release rknn output
//    rknn_outputs_release(app_ctx->rknn_ctx, app_ctx->io_num.n_output, outputs);
    for (int i = 0; i < app_ctx->io_num.n_input; i++) {
        rknn_destroy_mem(app_ctx->rknn_ctx, input_mems[i]);
    }
    for (int i = 0; i < app_ctx->io_num.n_output; i++) {
        rknn_destroy_mem(app_ctx->rknn_ctx, output_mems[i]);
    }

    out:
    if (dst_img.virt_addr != NULL) {
        free(dst_img.virt_addr);
    }

    return ret;
}