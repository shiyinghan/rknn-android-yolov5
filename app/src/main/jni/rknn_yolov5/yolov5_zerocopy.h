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


#ifndef _RKNN_DEMO_YOLOV5_ZERO_COPY_H_
#define _RKNN_DEMO_YOLOV5_ZERO_COPY_H_

#include "utils/common.h"
#include "postprocess.h"


int init_yolov5_model_zerocopy(const char* model_path, rknn_app_context_t* app_ctx);

int release_yolov5_model_zerocopy(rknn_app_context_t* app_ctx);

int inference_yolov5_model_zerocopy(rknn_app_context_t* app_ctx, image_buffer_t* img, object_detect_result_list* od_results);

#endif //_RKNN_DEMO_YOLOV5_ZERO_COPY_H_