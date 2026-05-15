#!/usr/bin/env python3
import argparse
import time
import cv2
import numpy as np
import pyrealsense2 as rs


PROFILES = [
    # common D435/D435i profiles
    (640, 480, 30),
    (848, 480, 30),
    (640, 360, 30),
    (480, 270, 30),
]


def try_profile(width, height, fps, color_format):
    print("=" * 80)
    print(f"Trying depth z16 + color {color_format}: {width}x{height}@{fps}")

    pipeline = rs.pipeline()
    config = rs.config()

    config.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
    config.enable_stream(rs.stream.color, width, height, color_format, fps)

    try:
        profile = pipeline.start(config)
    except Exception as e:
        print("start failed:", repr(e))
        return False

    try:
        dev = profile.get_device()
        print("device:", dev.get_info(rs.camera_info.name))
        print("serial:", dev.get_info(rs.camera_info.serial_number))
        try:
            print("usb:", dev.get_info(rs.camera_info.usb_type_descriptor))
        except Exception:
            pass

        # warmup
        got = False
        for i in range(60):
            try:
                frames = pipeline.wait_for_frames(1000)
                depth = frames.get_depth_frame()
                color = frames.get_color_frame()
                if depth and color:
                    print("got first frames at attempt", i)
                    got = True
                    break
            except Exception as e:
                print("wait attempt failed:", i, repr(e))

        if not got:
            print("NO FRAMES")
            return False

        print("Press q to quit.")
        while True:
            frames = pipeline.wait_for_frames(5000)
            depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame()

            depth = np.asanyarray(depth_frame.get_data())
            color_raw = np.asanyarray(color_frame.get_data())

            if color_format == rs.format.rgb8:
                color = cv2.cvtColor(color_raw, cv2.COLOR_RGB2BGR)
            elif color_format == rs.format.bgr8:
                color = color_raw
            elif color_format == rs.format.yuyv:
                color = cv2.cvtColor(color_raw, cv2.COLOR_YUV2BGR_YUY2)
            else:
                color = color_raw

            depth_vis = cv2.convertScaleAbs(depth, alpha=0.03)
            depth_vis = cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)

            color_show = cv2.resize(color, (640, 480))
            depth_show = cv2.resize(depth_vis, (640, 480))

            cv2.imshow("RealSense color", color_show)
            cv2.imshow("RealSense depth", depth_show)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        return True

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", default="auto")
    args = parser.parse_args()

    formats = [rs.format.rgb8, rs.format.bgr8, rs.format.yuyv]

    for width, height, fps in PROFILES:
        for fmt in formats:
            ok = try_profile(width, height, fps, fmt)
            if ok:
                print("WORKING PROFILE:", width, height, fps, fmt)
                return

    print("No working profile found.")


if __name__ == "__main__":
    main()
