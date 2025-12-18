#!/usr/bin/env python3
#
# Copyright 2025 KETI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Authors: KETI AI Robot Team

"""
CSI Camera Node for Jetson Orin Nano
- nvjpegenc hardware JPEG encoding
- flip-method for camera orientation
- HTTP MJPEG server on port 8080
"""
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage
import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst, GLib
import threading
from http.server import HTTPServer, BaseHTTPRequestHandler


class MJPEGHandler(BaseHTTPRequestHandler):
    """MJPEG stream handler"""
    protocol_version = 'HTTP/1.0'

    def log_message(self, format, *args):
        pass

    def do_HEAD(self):
        self.send_response(200)
        self.send_header('Content-Type', 'multipart/x-mixed-replace; boundary=frame')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, HEAD, OPTIONS')
        self.end_headers()

    def do_GET(self):
        if self.path in ('/', '/stream'):
            self.send_response(200)
            self.send_header('Content-Type', 'multipart/x-mixed-replace; boundary=frame')
            self.send_header('Cache-Control', 'no-cache, no-store, must-revalidate')
            self.send_header('Pragma', 'no-cache')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            try:
                while True:
                    frame = self.server.get_frame()
                    if frame:
                        self.wfile.write(b'--frame\r\nContent-Type: image/jpeg\r\nContent-Length: ')
                        self.wfile.write(str(len(frame)).encode())
                        self.wfile.write(b'\r\n\r\n')
                        self.wfile.write(frame)
                        self.wfile.write(b'\r\n')
                        self.wfile.flush()
                    self.server.frame_event.wait(0.08)
                    self.server.frame_event.clear()
            except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                pass
        else:
            self.send_response(404)
            self.end_headers()


class CameraNode(Node):
    def __init__(self):
        super().__init__('camera_node')

        Gst.init(None)

        # Parameters (IMX219: 4:3 sensor, native 640x480)
        self.declare_parameter('width', 640)
        self.declare_parameter('height', 480)
        self.declare_parameter('fps', 15)
        self.declare_parameter('quality', 60)
        self.declare_parameter('flip_method', 2)
        self.declare_parameter('http_port', 8080)

        width = self.get_parameter('width').value
        height = self.get_parameter('height').value
        fps = self.get_parameter('fps').value
        quality = self.get_parameter('quality').value
        flip = self.get_parameter('flip_method').value
        http_port = self.get_parameter('http_port').value

        # Publisher (RELIABLE for rosbridge compatibility)
        self.pub = self.create_publisher(CompressedImage, '/camera/image_raw/compressed', 1)

        # Frame storage
        self.latest_frame = None
        self.frame_lock = threading.Lock()

        # GStreamer pipeline
        pipeline_str = (
            f'nvarguscamerasrc sensor-id=0 ! '
            f'video/x-raw(memory:NVMM),width={width},height={height},framerate={fps}/1,format=NV12 ! '
            f'nvvidconv flip-method={flip} ! '
            f'video/x-raw(memory:NVMM),format=NV12 ! '
            f'nvjpegenc quality={quality} ! '
            f'appsink name=sink emit-signals=true max-buffers=1 drop=true sync=false'
        )

        self.get_logger().info(f'Pipeline: {pipeline_str}')

        try:
            self.pipeline = Gst.parse_launch(pipeline_str)
        except GLib.Error as e:
            self.get_logger().error(f'Pipeline error: {e}')
            return

        self.sink = self.pipeline.get_by_name('sink')
        self.sink.connect('new-sample', self.on_new_sample)

        ret = self.pipeline.set_state(Gst.State.PLAYING)
        if ret == Gst.StateChangeReturn.FAILURE:
            self.get_logger().error('Failed to start pipeline')
            return

        self.frame_count = 0

        # GLib loop
        self.loop = GLib.MainLoop()
        self.loop_thread = threading.Thread(target=self.loop.run, daemon=True)
        self.loop_thread.start()

        # HTTP MJPEG server (port 8080)
        self.http_server = HTTPServer(('0.0.0.0', http_port), MJPEGHandler)
        self.http_server.frame_event = threading.Event()
        self.http_server.get_frame = self.get_frame
        self.http_thread = threading.Thread(target=self.http_server.serve_forever, daemon=True)
        self.http_thread.start()

        self.get_logger().info(f'Camera: {width}x{height}@{fps}fps, flip={flip}')
        self.get_logger().info(f'MJPEG stream: http://0.0.0.0:{http_port}/stream')

    def get_frame(self):
        with self.frame_lock:
            return self.latest_frame

    def on_new_sample(self, sink):
        sample = sink.emit('pull-sample')
        if sample:
            buf = sample.get_buffer()
            success, map_info = buf.map(Gst.MapFlags.READ)
            if success:
                frame_data = bytes(map_info.data)

                with self.frame_lock:
                    self.latest_frame = frame_data
                self.http_server.frame_event.set()

                msg = CompressedImage()
                msg.header.stamp = self.get_clock().now().to_msg()
                msg.header.frame_id = 'camera_link'
                msg.format = 'jpeg'
                msg.data = frame_data
                self.pub.publish(msg)

                buf.unmap(map_info)
                self.frame_count += 1

        return Gst.FlowReturn.OK

    def destroy_node(self):
        if hasattr(self, 'http_server'):
            self.http_server.shutdown()
        if hasattr(self, 'pipeline'):
            self.pipeline.set_state(Gst.State.NULL)
        if hasattr(self, 'loop'):
            self.loop.quit()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = CameraNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
