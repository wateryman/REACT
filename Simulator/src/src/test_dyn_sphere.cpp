// REACT stage-2 [🟧 D-3] smoke test for the new DynSphere ray-sphere path
// in sensor_simulator.cu. No ROS, no yaml, no scene generation -- just a
// minimal GridMap that returns "empty everywhere" (1-point cloud with a
// large occupy_threshold) so forward rays naturally hit max_depth, and
// the dynamic sphere is the only thing that can occlude.
//
// Three checks:
//   (1) n_dyn=0: center pixel == max_depth_dist  (regression for the
//       default-args overload; verifies the new code path is truly no-op)
//   (2) n_dyn=1, sphere @ world (5, 0, 0) r=0.5: center pixel ≈ 4.5 m
//       (camera looks +X with T_wc=identity; depth is cam-frame x of hit)
//   (3) n_dyn=1, far-corner pixel unchanged at max_depth_dist (sphere
//       doesn't leak outside its angular footprint)

#include <iostream>
#include <cmath>
#include <vector>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <opencv2/opencv.hpp>
#include "sensor_simulator.cuh"

using namespace raycast;

namespace {

int check(const char* name, float got, float want, float tol)
{
    bool ok = std::fabs(got - want) <= tol;
    std::cout << "  [" << (ok ? "PASS" : "FAIL") << "] " << name
              << ": got=" << got << " want=" << want << " tol=" << tol << std::endl;
    return ok ? 0 : 1;
}

}  // namespace

int main()
{
    // ---- 1. Minimal "empty" GridMap ----
    // One-point cloud + occupy_threshold=999 means every mapQuery returns
    // 0 (empty), so the static raycast walks until max_depth_dist.
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>());
    cloud->points.emplace_back(0.0f, 0.0f, 0.0f);
    cloud->width = 1;
    cloud->height = 1;
    GridMap grid_map(cloud, 0.1f /*resolution*/, 999 /*occupy_threshold*/);

    // ---- 2. Camera params: YOPO sim defaults (160x90, fx=fy=80, cx=80, cy=45) ----
    CameraParams cam;  // all defaults match REACT_TARGET_INTRINSICS

    // ---- 3. T_wc: drone at (0,0,2) looking +X (above the z<=0 "ground"
    //         occupancy rule baked into GridMap::mapQuery) ----
    const float cam_z = 2.0f;
    cudaMat::SE3<float> T_wc(1.0f, 0.0f, 0.0f, 0.0f /*quat wxyz*/,
                             0.0f, 0.0f, cam_z /*pos*/);

    // ---- (1) n_dyn = 0 -> center pixel should be max_depth_dist ----
    cv::Mat depth0;
    renderDepthImage(&grid_map, &cam, T_wc, depth0);  // uses default n_dyn=0
    int fails = 0;
    fails += check("(1) n_dyn=0 center depth",
                   depth0.at<float>(45, 80), cam.max_depth_dist, 1e-4f);

    // ---- (2) one sphere at world (5, 0, cam_z) r=0.5 -> center depth ~ 4.5
    //         (sphere at same world-z as the camera so the +X ray hits it
    //         dead-center; center pixel sees the front surface at 4.5 m) ----
    std::vector<DynSphere> host_sph = {{ make_float3(5.0f, 0.0f, cam_z), 0.5f }};
    DynSphere* d_dyn = nullptr;
    int n_dyn = 0;
    uploadDynamicSpheres(&d_dyn, &n_dyn, host_sph);

    cv::Mat depth1;
    renderDepthImage(&grid_map, &cam, T_wc, depth1, d_dyn, n_dyn);
    fails += check("(2) one sphere center depth",
                   depth1.at<float>(45, 80), 4.5f, 0.05f);

    // ---- (3) corner pixel still untouched ----
    fails += check("(3) one sphere corner (10,10) untouched",
                   depth1.at<float>(10, 10), cam.max_depth_dist, 1e-4f);

    // ---- cleanup ----
    freeDynamicSpheres(&d_dyn, &n_dyn);
    grid_map.freeGridMap();

    if (fails == 0)
    {
        std::cout << "\n[PASS] test_dyn_sphere: 3/3 checks ok\n";
        return 0;
    }
    std::cout << "\n[FAIL] test_dyn_sphere: " << fails << " checks failed\n";
    return 1;
}
