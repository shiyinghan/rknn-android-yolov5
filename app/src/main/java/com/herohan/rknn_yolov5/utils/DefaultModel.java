package com.herohan.rknn_yolov5.utils;

public enum DefaultModel {
    YOLOV5S_FP("yolov5s-fp.rknn"),
    YOLOV5S_INT8("yolov5s-int8.rknn");

    public final String name;

    DefaultModel(String name) {
        this.name = name;
    }
}
