#!/usr/bin/env python3
import time
import pyrealsense2 as rs

ctx = rs.context()
devices = ctx.query_devices()

print("devices:", len(devices))

for dev in devices:
    print("=" * 80)
    print("name:", dev.get_info(rs.camera_info.name))
    print("serial:", dev.get_info(rs.camera_info.serial_number))
    print("firmware:", dev.get_info(rs.camera_info.firmware_version))
    try:
        print("usb:", dev.get_info(rs.camera_info.usb_type_descriptor))
    except Exception as e:
        print("usb: unavailable", e)

    for sensor in dev.query_sensors():
        print()
        print("sensor:", sensor.get_info(rs.camera_info.name))
        for p in sensor.get_stream_profiles():
            try:
                vp = p.as_video_stream_profile()
                print(
                    f"  stream={p.stream_type()} "
                    f"format={p.format()} "
                    f"{vp.width()}x{vp.height()} "
                    f"fps={p.fps()}"
                )
            except Exception:
                print(" ", p)

print()
print("Trying hardware reset...")
for dev in devices:
    try:
        dev.hardware_reset()
        print("reset sent")
    except Exception as e:
        print("reset failed:", e)

print("Wait 5 sec after reset...")
time.sleep(5)
