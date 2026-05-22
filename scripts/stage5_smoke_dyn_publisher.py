"""Stage-5.B.1c — smoke test the new /sim/dyn_obs subscriber.

Launches a small Python publisher that:
  1. Publishes a single drone odometry at the origin facing +X (one-shot).
  2. Publishes a single PoseArray with one ball at (5, 0, 1) radius 1.0.
  3. Subscribes to /depth_image and saves the next received frame as PNG.

Run order:
  T1) rosrun sensor_simulator sensor_simulator               (terminal A)
  T2) python scripts/stage5_smoke_dyn_publisher.py           (terminal B)

Expected result:
  /tmp/smoke_depth_with_ball.png shows the static scene depth WITH a
  visible spherical hole at the center-ish of the image (the ball is in
  front of the drone, occluding the static cloud).

Compare to /tmp/smoke_depth_no_ball.png (publish an empty PoseArray first
to clear), which should match the unchanged static-only render.

Sanity check: with the ball at (5, 0, 1) r=1.0, the ball front face is at
~4m and should appear as a depth-4m disc with ~0.4 radians angular size
(about 16% of the image width).
"""
import os
import sys
import time

import cv2
import numpy as np
import rospy
from geometry_msgs.msg import PoseArray, Pose
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Image
from cv_bridge import CvBridge


def main():
    rospy.init_node("stage5_dyn_smoke", anonymous=True)
    pub_odom = rospy.Publisher("/sim/odom", Odometry, queue_size=1, latch=True)
    pub_dyn  = rospy.Publisher("/sim/dyn_obs", PoseArray, queue_size=1, latch=True)
    bridge   = CvBridge()
    state = {"latest": None}

    def depth_cb(msg):
        state["latest"] = bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")

    rospy.Subscriber("/depth_image", Image, depth_cb, queue_size=1)
    time.sleep(0.5)

    # Drone at origin, identity quat = looking down +X
    odom = Odometry()
    odom.header.frame_id = "world"
    odom.pose.pose.position.x = 0.0
    odom.pose.pose.position.y = 0.0
    odom.pose.pose.position.z = 1.5
    odom.pose.pose.orientation.x = 0.0
    odom.pose.pose.orientation.y = 0.0
    odom.pose.pose.orientation.z = 0.0
    odom.pose.pose.orientation.w = 1.0
    pub_odom.publish(odom)

    # Empty dyn_obs first -> let the sim get into a steady no-ball state
    empty = PoseArray()
    empty.header.frame_id = "world"
    pub_dyn.publish(empty)
    print("[smoke] published empty dyn_obs; waiting 1.5 s for baseline depth ...")
    time.sleep(1.5)
    no_ball = state["latest"].copy() if state["latest"] is not None else None

    # Now publish one ball at (5, 0, 1.5) radius 1.0
    arr = PoseArray()
    arr.header.frame_id = "world"
    p = Pose()
    p.position.x = 5.0; p.position.y = 0.0; p.position.z = 1.5
    p.orientation.w = 1.0     # radius
    p.orientation.x = 0.0; p.orientation.y = 0.0; p.orientation.z = 0.0
    arr.poses.append(p)
    pub_dyn.publish(arr)
    print("[smoke] published 1 ball at (5,0,1.5) r=1.0; waiting 1.5 s ...")
    time.sleep(1.5)
    with_ball = state["latest"].copy() if state["latest"] is not None else None

    captured = {"no_ball": no_ball, "with_ball": with_ball}
    if captured["no_ball"] is None or captured["with_ball"] is None:
        print(f"[FAIL] missing frame: no_ball={captured['no_ball'] is not None}, "
              f"with_ball={captured['with_ball'] is not None}")
        print("   make sure rosrun sensor_simulator sensor_simulator is running in another terminal")
        return 1

    cv2.imwrite("/tmp/smoke_depth_no_ball.png",   captured["no_ball"])
    cv2.imwrite("/tmp/smoke_depth_with_ball.png", captured["with_ball"])
    diff = np.abs(captured["with_ball"].astype(np.float32) -
                  captured["no_ball"].astype(np.float32))
    n_changed = int((diff > 0.1).sum())
    print(f"[smoke] wrote /tmp/smoke_depth_{{no_ball,with_ball}}.png")
    print(f"[smoke] pixels that changed by > 0.1m: {n_changed} of {diff.size} "
          f"({100*n_changed/diff.size:.1f}%)")

    if n_changed < 50:
        print("[FAIL] very few pixels changed; the ball did not appear in depth")
        return 1
    print("[PASS] ball is visible in depth")
    return 0


if __name__ == "__main__":
    sys.exit(main())
