package com.herohan.rknn_yolov5;

import android.graphics.Bitmap;

public class YoloV5Detect {
    static {
        System.loadLibrary("rknn_yolov5");
    }

    public native boolean init(String modelPath, String labelListPath, boolean useZeroCopy);

    public native boolean detect(Bitmap srtBitmap);

    public native boolean release();
}
