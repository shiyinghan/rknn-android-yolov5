LOCAL_PATH := $(call my-dir)

include $(CLEAR_VARS)
LOCAL_MODULE := turbojpeg-static
LOCAL_SRC_FILES := $(TARGET_ARCH_ABI)/libturbojpeg.a
LOCAL_EXPORT_C_INCLUDES := $(LOCAL_PATH)/include
include $(PREBUILT_STATIC_LIBRARY)