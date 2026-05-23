#include <pcl/io/ply_io.h>
#include <pcl/point_types.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <opencv2/opencv.hpp>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <yaml-cpp/yaml.h>
#include <iostream>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <random>
#include <sstream>
#include <string>
#include <vector>
#include "sensor_simulator.cuh"
#include "maps.hpp"

using namespace raycast;
namespace fs = std::filesystem;

void prepareSavePath(const std::string &path, bool print=false)
{
    if (fs::exists(path))
    {
        if (print)
            std::cout << "Directory exists. Removing: " << path << std::endl;
        fs::remove_all(path);
    }
    fs::create_directories(path);
    if (print)
        std::cout << "Created new dataset directory: " << path << std::endl;
}

void savePointCloudAsPLY(const pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud, const std::string &path)
{
    if (pcl::io::savePLYFileBinary(path, *cloud) == -1)
        std::cerr << "Failed to save ply file to " << path << std::endl;
}

void saveDepthAs16BitPNG(const cv::Mat &depth_float, float max_depth_dist, const std::string &filepath)
{
    cv::Mat depth_scaled;
    depth_scaled = depth_float / max_depth_dist; // 归一化0~1

    // clip [0,1]
    cv::threshold(depth_scaled, depth_scaled, 1.0, 1.0, cv::THRESH_TRUNC);
    cv::threshold(depth_scaled, depth_scaled, 0.0, 0.0, cv::THRESH_TOZERO);

    // 转成uint16
    depth_scaled.convertTo(depth_scaled, CV_16UC1, 65535.0);

    cv::imwrite(filepath, depth_scaled);
}

Eigen::Quaternionf RPY2Quat(float roll_deg, float pitch_deg, float yaw_deg)
{
    float roll = roll_deg * M_PI / 180.0f;
    float pitch = pitch_deg * M_PI / 180.0f;
    float yaw = yaw_deg * M_PI / 180.0f;
    Eigen::AngleAxisf rollAngle(roll, Eigen::Vector3f::UnitX());
    Eigen::AngleAxisf pitchAngle(pitch, Eigen::Vector3f::UnitY());
    Eigen::AngleAxisf yawAngle(yaw, Eigen::Vector3f::UnitZ());
    return yawAngle * pitchAngle * rollAngle;
}

void printProgressBar(int current, int total, int bar_width = 50)
{
    float progress = static_cast<float>(current) / total;
    int pos = static_cast<int>(bar_width * progress);

    std::cout << "\r[";
    for (int i = 0; i < bar_width; ++i)
    {
        if (i < pos)
            std::cout << "=";
        else if (i == pos)
            std::cout << ">";
        else
            std::cout << " ";
    }
    std::cout << "] " << int(progress * 100.0f) << "%";
    std::cout.flush();
}

// ============================================================================
// REACT stage-2 [🟧 D-3] dynamic-dataset baking helpers
// ============================================================================

// Minimal inline JSON formatting (avoids an extra apt-install nlohmann/json3).
// Only floats, fixed-key objects, and arrays are needed for our schema.
namespace jsn
{
    inline std::string f(float x)
    {
        std::ostringstream oss;
        oss << std::setprecision(7) << x;
        return oss.str();
    }
    inline std::string vec3(float x, float y, float z)
    {
        return "[" + f(x) + "," + f(y) + "," + f(z) + "]";
    }
}

struct DynamicBall
{
    Eigen::Vector3f pos;
    Eigen::Vector3f vel;
    float radius;

    // Constant-velocity step with axis-aligned bbox reflection.
    void step(float dt, const Eigen::Vector3f& bbox_lo, const Eigen::Vector3f& bbox_hi)
    {
        pos += vel * dt;
        for (int k = 0; k < 3; ++k)
        {
            if (pos[k] < bbox_lo[k] || pos[k] > bbox_hi[k])
            {
                vel[k] *= -1.0f;
                pos[k] = std::clamp(pos[k], bbox_lo[k], bbox_hi[k]);
            }
        }
    }
};

static std::vector<DynamicBall> spawn_balls(std::mt19937& rng, const YAML::Node& dcfg,
                                             const Eigen::Vector3f& bbox_lo,
                                             const Eigen::Vector3f& bbox_hi,
                                             const Eigen::Vector3f& drone_start_pos = Eigen::Vector3f::Zero(),
                                             const Eigen::Quaternionf& drone_start_quat = Eigen::Quaternionf::Identity())
{
    std::uniform_int_distribution<int> count_dist(dcfg["count_min"].as<int>(), dcfg["count_max"].as<int>());
    std::uniform_real_distribution<float> speed_dist(dcfg["speed_min"].as<float>(), dcfg["speed_max"].as<float>());
    std::uniform_real_distribution<float> radius_dist(dcfg["radius_min"].as<float>(), dcfg["radius_max"].as<float>());
    std::uniform_real_distribution<float> u01(0.0f, 1.0f);
    std::normal_distribution<float> normal01(0.0f, 1.0f);

    // 🟦 stage-3.7 v4: camera-aware spawn. When spawn_in_front is true, balls are
    // generated inside a forward cone of the drone's initial body frame instead
    // of uniformly in the world bbox.  This raises the FOV-presence rate from
    // the ~22 % observed in v3 to ~60 %, giving the model more dynamic-obstacle
    // training signal per sequence.  See REACT_MATH_Derivations/01 §4 last
    // bullet ("camera-aware ball spawn") for motivation.
    bool spawn_in_front = dcfg["spawn_in_front"] ? dcfg["spawn_in_front"].as<bool>() : false;
    float fov_yaw_deg   = dcfg["fov_yaw_deg"]   ? dcfg["fov_yaw_deg"].as<float>()   : 40.0f;
    float fov_pitch_deg = dcfg["fov_pitch_deg"] ? dcfg["fov_pitch_deg"].as<float>() : 30.0f;
    float fov_dist_min  = dcfg["fov_dist_min"]  ? dcfg["fov_dist_min"].as<float>()  : 3.0f;
    float fov_dist_max  = dcfg["fov_dist_max"]  ? dcfg["fov_dist_max"].as<float>()  : 12.0f;
    float fov_yaw_rad   = fov_yaw_deg   * static_cast<float>(M_PI) / 180.0f;
    float fov_pitch_rad = fov_pitch_deg * static_cast<float>(M_PI) / 180.0f;

    int n = count_dist(rng);
    std::vector<DynamicBall> balls;
    balls.reserve(n);
    for (int i = 0; i < n; ++i)
    {
        DynamicBall b;
        if (spawn_in_front)
        {
            // Sample (distance, yaw_offset, pitch_offset) in the camera-aware
            // forward cone, then transform from body to world via drone_quat.
            float d   = fov_dist_min + u01(rng) * (fov_dist_max - fov_dist_min);
            float th  = (-1.0f + 2.0f * u01(rng)) * fov_yaw_rad   / 2.0f;
            float phi = (-1.0f + 2.0f * u01(rng)) * fov_pitch_rad / 2.0f;
            Eigen::Vector3f p_body(d * std::cos(phi) * std::cos(th),
                                   d * std::cos(phi) * std::sin(th),
                                   d * std::sin(phi));
            Eigen::Vector3f p_world = drone_start_quat * p_body + drone_start_pos;
            // Clamp to bbox so ball-step's bbox reflection stays sane.
            p_world.x() = std::max(bbox_lo.x(), std::min(bbox_hi.x(), p_world.x()));
            p_world.y() = std::max(bbox_lo.y(), std::min(bbox_hi.y(), p_world.y()));
            p_world.z() = std::max(bbox_lo.z(), std::min(bbox_hi.z(), p_world.z()));
            b.pos = p_world;
        }
        else
        {
            // Legacy v1/v2/v3 spawn: uniform in bbox.
            b.pos.x() = bbox_lo.x() + u01(rng) * (bbox_hi.x() - bbox_lo.x());
            b.pos.y() = bbox_lo.y() + u01(rng) * (bbox_hi.y() - bbox_lo.y());
            b.pos.z() = bbox_lo.z() + u01(rng) * (bbox_hi.z() - bbox_lo.z());
        }
        // Random direction: gaussian unit vector with vertical damping
        Eigen::Vector3f dir(normal01(rng), normal01(rng), 0.2f * normal01(rng));
        dir.normalize();
        b.vel = speed_dist(rng) * dir;
        b.radius = radius_dist(rng);
        balls.push_back(b);
    }
    return balls;
}

// Generates K consecutive drone poses at constant body-frame +X velocity.
// Each frame is checked against the static kdtree for safe_dist; if any frame
// fails the check, the whole trajectory is resampled (up to max_retry).
struct DroneFrame
{
    Eigen::Vector3f pos;
    Eigen::Quaternionf quat_wc;   // world->camera (includes pitch_bc baked in)
    Eigen::Vector3f vel_world;
};

static std::vector<DroneFrame> sample_drone_traj_k_frames(
    std::mt19937& rng, int K, float dt,
    const YAML::Node& tcfg, const YAML::Node& main_cfg,
    pcl::KdTreeFLANN<pcl::PointXYZ>& kdtree, float safe_dist,
    const Eigen::Quaternionf& quat_bc, int max_retry = 32)
{
    float x_range    = main_cfg["x_range"].as<float>();
    float y_range    = main_cfg["y_range"].as<float>();
    float z_min      = main_cfg["z_range"][0].as<float>();
    float z_max      = main_cfg["z_range"][1].as<float>();
    float roll_range = main_cfg["roll_range"].as<float>();
    float pitch_rng  = main_cfg["pitch_range"].as<float>();
    float vx_min     = tcfg["v_body_x_min"].as<float>();
    float vx_max     = tcfg["v_body_x_max"].as<float>();

    std::normal_distribution<float> normal01(0.0f, 1.0f);
    std::uniform_real_distribution<float> u01(0.0f, 1.0f);

    for (int attempt = 0; attempt < max_retry; ++attempt)
    {
        Eigen::Vector3f pos_0;
        pos_0.x() = -x_range / 2.0f + u01(rng) * x_range;
        pos_0.y() = -y_range / 2.0f + u01(rng) * y_range;
        pos_0.z() = z_min + u01(rng) * (z_max - z_min);
        float roll  = normal01(rng) * roll_range / 3.0f;
        float pitch = normal01(rng) * pitch_rng  / 3.0f;
        float yaw   = u01(rng) * 360.0f;
        Eigen::Quaternionf quat_wb = RPY2Quat(roll, pitch, yaw);
        Eigen::Quaternionf quat_wc = quat_wb * quat_bc;
        float vx = vx_min + u01(rng) * (vx_max - vx_min);
        Eigen::Vector3f v_body(vx, 0.0f, 0.0f);
        Eigen::Vector3f v_world = quat_wb * v_body;

        std::vector<DroneFrame> traj;
        traj.reserve(K);
        bool ok = true;
        for (int k = 0; k < K; ++k)
        {
            Eigen::Vector3f pos_k = pos_0 + v_world * (k * dt);
            pcl::PointXYZ probe(pos_k.x(), pos_k.y(), pos_k.z());
            std::vector<int> idx(1);
            std::vector<float> sqd(1);
            kdtree.nearestKSearch(probe, 1, idx, sqd);
            if (std::sqrt(sqd[0]) < safe_dist)
            {
                ok = false;
                break;
            }
            DroneFrame f;
            f.pos = pos_k;
            f.quat_wc = quat_wc;
            f.vel_world = v_world;
            traj.push_back(f);
        }
        if (ok) return traj;
    }
    return {};   // empty -> caller skips this seq
}

// Run one episode: spawn balls, sample drone traj, render K frames + write JSON.
// Returns true on success, false if drone trajectory could not be sampled.
static bool generate_one_sequence(GridMap& grid_map, CameraParams& camera,
                                  pcl::KdTreeFLANN<pcl::PointXYZ>& kdtree,
                                  float safe_dist, const Eigen::Quaternionf& quat_bc,
                                  std::mt19937& rng, int env_id, int seq_id, int K, float dt,
                                  const YAML::Node& dcfg, const YAML::Node& tcfg,
                                  const YAML::Node& main_cfg, const fs::path& seq_dir)
{
    Eigen::Vector3f bbox_lo(-dcfg["bbox_xy_half"].as<float>(),
                            -dcfg["bbox_xy_half"].as<float>(),
                             dcfg["bbox_z_lo"].as<float>());
    Eigen::Vector3f bbox_hi( dcfg["bbox_xy_half"].as<float>(),
                             dcfg["bbox_xy_half"].as<float>(),
                             dcfg["bbox_z_hi"].as<float>());

    // 🟦 stage-3.7 v4: drone trajectory FIRST so camera-aware spawn knows
    // the drone's initial pose.  Legacy (spawn_in_front=false) path is
    // unaffected because spawn_balls ignores the drone_start args.
    auto traj  = sample_drone_traj_k_frames(rng, K, dt, tcfg, main_cfg, kdtree, safe_dist, quat_bc);
    if (traj.empty()) return false;
    auto balls = spawn_balls(rng, dcfg, bbox_lo, bbox_hi, traj[0].pos, traj[0].quat_wc);

    fs::create_directories(seq_dir);

    DynSphere* d_dyn = nullptr;
    int n_dyn = 0;

    // ---- per-frame loop: step balls, upload, render, log ----
    // dyn_obs.json: per-frame array of obstacle dicts
    // state.json:   per-frame drone dict (pos, quat_wc, vel_world)
    std::ostringstream dyn_obs_json;
    std::ostringstream state_json;
    dyn_obs_json << "[";
    state_json << "[";

    for (int k = 0; k < K; ++k)
    {
        // ball state at frame k is the state AFTER stepping from the prior frame.
        // For k=0 we log the spawn state without stepping; for k>=1 we step first.
        if (k > 0)
            for (auto& b : balls) b.step(dt, bbox_lo, bbox_hi);

        // upload to GPU
        std::vector<DynSphere> sph;
        sph.reserve(balls.size());
        for (const auto& b : balls)
            sph.push_back({ make_float3(b.pos.x(), b.pos.y(), b.pos.z()), b.radius });
        uploadDynamicSpheres(&d_dyn, &n_dyn, sph);

        // render
        const DroneFrame& df = traj[k];
        cudaMat::SE3<float> T_wc(df.quat_wc.w(), df.quat_wc.x(), df.quat_wc.y(), df.quat_wc.z(),
                                 df.pos.x(), df.pos.y(), df.pos.z());
        cv::Mat depth_image;
        renderDepthImage(&grid_map, &camera, T_wc, depth_image, d_dyn, n_dyn);

        // save depth: same normalized uint16 format as YOPO static dataset
        // (saveDepthAs16BitPNG: depth / max_depth_dist * 65535).
        std::string depth_path = (seq_dir / ("depth_t" + std::to_string(k) + ".png")).string();
        saveDepthAs16BitPNG(depth_image, camera.max_depth_dist, depth_path);

        // log dyn_obs (per-frame ball list)
        if (k > 0) dyn_obs_json << ",";
        dyn_obs_json << "[";
        for (size_t i = 0; i < balls.size(); ++i)
        {
            if (i > 0) dyn_obs_json << ",";
            dyn_obs_json << "{\"pos\":"    << jsn::vec3(balls[i].pos.x(),  balls[i].pos.y(),  balls[i].pos.z())
                         << ",\"vel\":"   << jsn::vec3(balls[i].vel.x(),  balls[i].vel.y(),  balls[i].vel.z())
                         << ",\"radius\":" << jsn::f(balls[i].radius)
                         << ",\"kind\":\"sphere\"}";
        }
        dyn_obs_json << "]";

        // log state (drone pose)
        if (k > 0) state_json << ",";
        state_json << "{\"pos\":"  << jsn::vec3(df.pos.x(), df.pos.y(), df.pos.z())
                   << ",\"quat_wc\":[" << jsn::f(df.quat_wc.w()) << "," << jsn::f(df.quat_wc.x())
                   << "," << jsn::f(df.quat_wc.y()) << "," << jsn::f(df.quat_wc.z()) << "]"
                   << ",\"vel_world\":" << jsn::vec3(df.vel_world.x(), df.vel_world.y(), df.vel_world.z())
                   << "}";
    }
    dyn_obs_json << "]";
    state_json << "]";

    std::ofstream(seq_dir / "dyn_obs.json") << dyn_obs_json.str();
    std::ofstream(seq_dir / "state.json")  << state_json.str();

    // meta.json -- intrinsics self-check + seq params
    std::ostringstream meta;
    meta << "{\"K\":" << K << ",\"dt\":" << jsn::f(dt)
         << ",\"env_id\":" << env_id << ",\"seq_id\":" << seq_id
         << ",\"intrinsics\":{\"fx\":" << jsn::f(camera.fx) << ",\"fy\":" << jsn::f(camera.fy)
         << ",\"cx\":" << jsn::f(camera.cx) << ",\"cy\":" << jsn::f(camera.cy)
         << ",\"W\":" << camera.image_width << ",\"H\":" << camera.image_height
         << ",\"max_depth_m\":" << jsn::f(camera.max_depth_dist) << "}"
         << ",\"depth_encoding\":\"uint16_normalized_by_max_depth\"}";
    std::ofstream(seq_dir / "meta.json") << meta.str();

    freeDynamicSpheres(&d_dyn, &n_dyn);
    return true;
}

// Orchestrator for `mode: dynamic`. Builds static map per env (same logic as
// static main loop), then for each env loops over n_seqs_per_env sequences.
static int run_dynamic_mode(const YAML::Node& cfg)
{
    // camera params (same as static path)
    CameraParams camera;
    camera.fx              = cfg["camera"]["fx"].as<float>();
    camera.fy              = cfg["camera"]["fy"].as<float>();
    camera.cx              = cfg["camera"]["cx"].as<float>();
    camera.cy              = cfg["camera"]["cy"].as<float>();
    camera.image_width     = cfg["camera"]["image_width"].as<int>();
    camera.image_height    = cfg["camera"]["image_height"].as<int>();
    camera.max_depth_dist  = cfg["camera"]["max_depth_dist"].as<float>();
    camera.normalize_depth = cfg["camera"]["normalize_depth"].as<bool>();
    float pitch = cfg["camera"]["pitch"].as<float>() * M_PI / 180.0f;
    Eigen::Quaternionf quat_bc(Eigen::AngleAxisf(pitch, Eigen::Vector3f::UnitY()));

    // map params
    float resolution = cfg["resolution"].as<float>();
    int occupy_threshold = cfg["occupy_threshold"].as<int>();
    int seed = cfg["seed"].as<int>();
    int sizeX = cfg["x_length"].as<int>();
    int sizeY = cfg["y_length"].as<int>();
    int sizeZ = cfg["z_length"].as<int>();
    double scale = 1.0 / resolution;
    sizeX = sizeX * scale;
    sizeY = sizeY * scale;
    sizeZ = sizeZ * scale;
    float safe_dist = cfg["safe_dist"].as<float>();
    float ply_res   = cfg["ply_res"].as<float>();

    // dynamic params
    int n_envs         = cfg["n_envs"].as<int>();
    int n_seqs_per_env = cfg["n_seqs_per_env"].as<int>();
    int K              = cfg["K"].as<int>();
    float dt           = cfg["dt"].as<float>();
    std::string out_root = cfg["out_root"].as<std::string>();
    const YAML::Node& dcfg = cfg["dyn_obs"];
    const YAML::Node& tcfg = cfg["drone_traj"];

    prepareSavePath(out_root, true);
    std::cout << "[REACT D-3] mode=dynamic, n_envs=" << n_envs
              << ", n_seqs_per_env=" << n_seqs_per_env
              << ", K=" << K << ", dt=" << dt << " s" << std::endl;

    int total_seqs = n_envs * n_seqs_per_env;
    int done = 0, failed = 0;
    for (int env_id = 0; env_id < n_envs; ++env_id)
    {
        // build static scene exactly like static main() does
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>());
        mocka::Maps::BasicInfo info;
        info.sizeX = sizeX;
        info.sizeY = sizeY;
        info.sizeZ = sizeZ;
        info.seed  = seed + env_id;
        info.scale = scale;
        info.cloud = cloud;
        mocka::Maps map;
        map.setParam(cfg);
        map.setInfo(info);
        map.generate(cfg["maze_type"].as<int>());

        GridMap grid_map(cloud, resolution, occupy_threshold);

        // filtered cloud for kdtree (same voxel size as static path)
        pcl::PointCloud<pcl::PointXYZ>::Ptr filtered_cloud(new pcl::PointCloud<pcl::PointXYZ>());
        pcl::VoxelGrid<pcl::PointXYZ> sor;
        sor.setInputCloud(cloud);
        sor.setLeafSize(ply_res, ply_res, ply_res);
        sor.filter(*filtered_cloud);
        pcl::KdTreeFLANN<pcl::PointXYZ> kdtree;
        kdtree.setInputCloud(filtered_cloud);

        // save the pointcloud beside env_X/  (so ESDF in stage-3 can reload)
        std::ostringstream env_name;
        env_name << "env_" << std::setfill('0') << std::setw(4) << env_id;
        fs::path env_dir = fs::path(out_root) / env_name.str();
        fs::create_directories(env_dir);
        savePointCloudAsPLY(filtered_cloud, (env_dir / "pointcloud.ply").string());

        std::mt19937 rng(seed + env_id * 1000);
        for (int s = 0; s < n_seqs_per_env; ++s)
        {
            std::ostringstream seq_name;
            seq_name << "seq_" << std::setfill('0') << std::setw(4) << s;
            fs::path seq_dir = env_dir / seq_name.str();
            bool ok = generate_one_sequence(grid_map, camera, kdtree, safe_dist, quat_bc,
                                            rng, env_id, s, K, dt, dcfg, tcfg, cfg, seq_dir);
            if (!ok) { ++failed; }
            ++done;
            printProgressBar(done, total_seqs);
        }
        grid_map.freeGridMap();
    }
    std::cout << "\n[REACT D-3] done: " << (total_seqs - failed) << "/" << total_seqs
              << " sequences (" << failed << " skipped after retries)" << std::endl;
    return 0;
}

int main(int argc, char **argv)
{
    // REACT stage-2 [🟧 D-3]: optional `--config <path>` argv flag overrides
    // the compile-time CONFIG_FILE_PATH macro. Default (no flag) preserves
    // the original behaviour byte-for-byte.
    std::string config_path = CONFIG_FILE_PATH;
    for (int i = 1; i + 1 < argc; ++i)
    {
        if (std::string(argv[i]) == "--config")
            config_path = argv[i + 1];
    }
    std::cout << "[REACT] config: " << config_path << std::endl;
    YAML::Node config = YAML::LoadFile(config_path);

    // REACT stage-2 [🟧 D-3]: dispatch on `mode`. Absent or "static" runs the
    // original single-frame flow below. "dynamic" runs the K-frame baker.
    std::string mode = config["mode"] ? config["mode"].as<std::string>() : "static";
    if (mode == "dynamic")
        return run_dynamic_mode(config);

    // 1. 相机参数
    CameraParams camera;
    camera.fx = config["camera"]["fx"].as<float>();
    camera.fy = config["camera"]["fy"].as<float>();
    camera.cx = config["camera"]["cx"].as<float>();
    camera.cy = config["camera"]["cy"].as<float>();
    camera.image_width = config["camera"]["image_width"].as<int>();
    camera.image_height = config["camera"]["image_height"].as<int>();
    camera.max_depth_dist = config["camera"]["max_depth_dist"].as<float>();
    camera.normalize_depth = config["camera"]["normalize_depth"].as<bool>();
    float pitch = config["camera"]["pitch"].as<float>() * M_PI / 180.0;
    Eigen::AngleAxisf angle_axis(pitch, Eigen::Vector3f::UnitY());
    Eigen::Quaternionf quat_bc(angle_axis);

    // 2. 地图参数
    float resolution = config["resolution"].as<float>();
    int occupy_threshold = config["occupy_threshold"].as<int>();
    int seed = config["seed"].as<int>();
    int sizeX = config["x_length"].as<int>();
    int sizeY = config["y_length"].as<int>();
    int sizeZ = config["z_length"].as<int>();
    double scale = 1 / resolution;
    sizeX *= scale;
    sizeY *= scale;
    sizeZ *= scale;

    // 3. 数据集参数
    std::string save_path = config["save_path"].as<std::string>();
    int env_num = config["env_num"].as<int>();
    int image_num = config["image_num"].as<int>();
    float roll_range = config["roll_range"].as<float>();
    float pitch_range = config["pitch_range"].as<float>();
    float x_range = config["x_range"].as<float>();
    float y_range = config["y_range"].as<float>();
    float z_min = config["z_range"][0].as<float>();
    float z_max = config["z_range"][1].as<float>();
    float safe_dist = config["safe_dist"].as<float>();
    float ply_res = config["ply_res"].as<float>();

    // 中心对齐，计算偏移量
    int dataset_num = env_num * image_num;
    float x_min = -x_range / 2.0f;
    float y_min = -y_range / 2.0f;

    std::cout << "地图范围 (m): "
              << "X: [" << -sizeX * resolution / 2.0 << ", " << sizeX * resolution / 2.0 << "], "
              << "Y: [" << -sizeY * resolution / 2.0 << ", " << sizeY * resolution / 2.0 << "], "
              << "Z: [" << 0 << ", " << sizeZ * resolution << "]" << std::endl;

    std::cout << "采集范围 (m): "
              << "X: [" << x_min << ", " << x_min + x_range << "], "
              << "Y: [" << y_min << ", " << y_min + y_range << "], "
              << "Z: [" << z_min << ", " << z_max << "]" << std::endl;

    std::cout << "角度范围 (度): "
              << "Roll: [" << -roll_range << ", " << roll_range << "], "
              << "Pitch: [" << -pitch_range << ", " << pitch_range << "], "
              << "Yaw: [0, 360]" << std::endl;

    // 收集所有数据
    std::default_random_engine generator(std::random_device{}());
    std::normal_distribution<float> normal_distribution(0.0f, 1.0f); // 均值0，标准差1
    std::uniform_real_distribution<float> uniform_uniform(0.0f, 1.0f);
    prepareSavePath(save_path, true);
    for (int map_i = 0; map_i < env_num; ++map_i)
    {
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>());
        mocka::Maps::BasicInfo info;
        info.sizeX = sizeX;
        info.sizeY = sizeY;
        info.sizeZ = sizeZ;
        info.seed = seed + map_i; // 每个环境使用不同的随机种子
        info.scale = scale;
        info.cloud = cloud;

        mocka::Maps map;
        map.setParam(config);
        map.setInfo(info);
        map.generate(config["maze_type"].as<int>());

        // 构建 GridMap
        GridMap grid_map(cloud, resolution, occupy_threshold);

        // 保存地图 (先滤波)
        pcl::PointCloud<pcl::PointXYZ>::Ptr filtered_cloud(new pcl::PointCloud<pcl::PointXYZ>());
        pcl::VoxelGrid<pcl::PointXYZ> sor;
        sor.setInputCloud(cloud);
        sor.setLeafSize(ply_res, ply_res, ply_res);
        sor.filter(*filtered_cloud);
        pcl::PointXYZ min_pt, max_pt;
        pcl::getMinMax3D(*filtered_cloud, min_pt, max_pt);

        std::string image_path = save_path + std::to_string(map_i) + "/";
        prepareSavePath(image_path);

        savePointCloudAsPLY(filtered_cloud, save_path + "pointcloud-" + std::to_string(map_i) + ".ply");

        pcl::KdTreeFLANN<pcl::PointXYZ> kdtree;
        kdtree.setInputCloud(filtered_cloud);

        // 收集当前环境的数据
        std::ofstream pose_file(save_path + "pose-" + std::to_string(map_i) + ".csv");
        pose_file << "px,py,pz,qw,qx,qy,qz\n";
        for (int image_i = 0; image_i < image_num; ++image_i)
        {
            Eigen::Vector3f pos;
            float dist;
            do{
                pos.x() = x_min + uniform_uniform(generator) * x_range;
                pos.y() = y_min + uniform_uniform(generator) * y_range;
                pos.z() = z_min + uniform_uniform(generator) * (z_max - z_min);
                pcl::PointXYZ searchPoint(pos.x(), pos.y(), pos.z());
                std::vector<int> pointIdxNKNSearch(1);
                std::vector<float> pointNKNSquaredDistance(1);
                int found_num = kdtree.nearestKSearch(searchPoint, 1, pointIdxNKNSearch, pointNKNSquaredDistance);
                dist = sqrt(pointNKNSquaredDistance[0]);
            } while (dist < safe_dist);

            float roll = normal_distribution(generator) * roll_range / 3.0f;   // 3 * sigmoid = range
            float pitch = normal_distribution(generator) * pitch_range / 3.0f; // 3 * sigmoid = range
            float yaw = uniform_uniform(generator) * 360.0f;

            Eigen::Quaternionf quat = RPY2Quat(roll, pitch, yaw);
            Eigen::Quaternionf quat_wc = quat * quat_bc;

            cudaMat::SE3<float> T_wc(quat_wc.w(), quat_wc.x(), quat_wc.y(), quat_wc.z(),
                                     pos.x(), pos.y(), pos.z());

            cv::Mat depth_image;
            renderDepthImage(&grid_map, &camera, T_wc, depth_image);

            std::string filename = image_path + "/img_" + std::to_string(image_i) + ".png";
            saveDepthAs16BitPNG(depth_image, camera.max_depth_dist, filename);

            pose_file << std::fixed << std::setprecision(6)
                      << pos.x() << "," << pos.y() << "," << pos.z() << ","
                      << quat_wc.w() << "," << quat_wc.x() << ","
                      << quat_wc.y() << "," << quat_wc.z() << "\n";

            printProgressBar(map_i * image_num + image_i + 1, dataset_num);
        }
        pose_file.close();
        grid_map.freeGridMap();
    }

    std::cout << "\nDataset generation completed!" << std::endl;

    return 0;
}
