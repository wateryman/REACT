#include "sensor_simulator.h"

cv::Mat SensorSimulator::renderDepthImage(){

    cv::Mat depth_image(image_height, image_width, CV_32FC1, cv::Scalar(std::numeric_limits<float>::max()));
    Eigen::Matrix3f R_wc = quat.toRotationMatrix();
    Eigen::Matrix3f R_cw = R_wc.inverse();

    // 🟦 stage-5.B: snapshot dyn_spheres_ in camera frame.  Done once per
    // call (not per pixel) so the per-pixel sphere test is a fast scan.
    std::vector<DynSphereCPU> spheres_cam;
    {
        std::lock_guard<std::mutex> g(dyn_mtx_);
        spheres_cam.reserve(dyn_spheres_.size());
        for (const auto &s : dyn_spheres_) {
            DynSphereCPU sc;
            sc.pos    = R_cw * (s.pos - pos);
            sc.radius = s.radius;
            spheres_cam.push_back(sc);
        }
    }

    auto start = std::chrono::high_resolution_clock::now();
#pragma omp parallel for
    for (int v = 0; v < image_height; ++v) {
        for (int u = 0; u < image_width; ++u) {
            // 计算射线方向（图像平面坐标系）
            float y = -(u - cx) / fx;
            float z = -(v - cy) / fy;
            float x = 1.0f;
            Eigen::Vector3f d(x, y, z);
            d.normalize();

            // 转换到世界坐标系下
            Eigen::Vector3f ray_direction = R_wc * d;  // 考虑相机旋转
            Eigen::Vector3f ray_origin = pos;         // 相机的位置

            float depth_static = std::numeric_limits<float>::max();
            // 使用Octree查找射线方向上最近的点
            std::vector<int> pointIdxVec;
            if (octree->getIntersectedVoxelIndices(ray_origin, ray_direction, pointIdxVec, 1)) {
                pcl::PointXYZ closest_point = cloud->points[pointIdxVec[0]];
                Eigen::Vector3f point_in_world = closest_point.getVector3fMap();
                Eigen::Vector3f closest_point_camera = R_cw * (point_in_world - pos);
                float distance = closest_point_camera(0);
                if (distance < 0) distance = 0;
                if (distance > max_depth_dist) distance = max_depth_dist;
                depth_static = distance;
            }

            // 🟦 stage-5.B: CPU ray-sphere intersection.  Mirrors the CUDA
            // ray_sphere_depth() in sensor_simulator.cu.  d is the unit ray
            // in CAMERA frame; sphere centres are stored in camera frame in
            // spheres_cam.  Returns +X distance (matching the static depth
            // convention).  Take min with the static depth.
            float depth_dyn = std::numeric_limits<float>::max();
            for (const auto &sc : spheres_cam) {
                // |t * d - c|^2 = r^2  =>  t^2 - 2 t (d.c) + (|c|^2 - r^2) = 0
                float b  = d.dot(sc.pos);
                float c2 = sc.pos.squaredNorm() - sc.radius * sc.radius;
                float disc = b * b - c2;
                if (disc < 0.0f) continue;                 // no hit
                float t = b - std::sqrt(disc);             // front face
                if (t <= 0.0f) continue;                   // behind / inside
                // d is unit; the depth-convention here is +X coord in camera
                // frame, i.e. t * d.x() (camera looks down +X by line 16
                // above).  Match the static-depth distance = closest_point_camera(0).
                float depth_t = t * d.x();
                if (depth_t < 0.0f) continue;
                if (depth_t < depth_dyn) depth_dyn = depth_t;
            }

            float depth = std::min(depth_static, depth_dyn);
            if (depth < std::numeric_limits<float>::max()) {
                if (depth > max_depth_dist) depth = max_depth_dist;
                if (normalize_depth) depth = depth / max_depth_dist;
                depth_image.at<float>(v, u) = depth;
            }
        }
    }
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    // std::cout << "生成图像耗时: " << elapsed.count() << " 秒" << std::endl; // 输出耗时

    // 将无效值设置为0
    for (int v = 0; v < image_height; ++v) {
        for (int u = 0; u < image_width; ++u) {
            if (depth_image.at<float>(v, u) == std::numeric_limits<float>::max())
                depth_image.at<float>(v, u) = max_depth_dist;
        }
    }

    return depth_image;
}

pcl::PointCloud<pcl::PointXYZ> SensorSimulator::renderLidarPointcloud() {
    Eigen::Matrix3f R_wc = quat.toRotationMatrix();
    Eigen::Matrix3f R_cw = R_wc.inverse();
    pcl::PointCloud<pcl::PointXYZ> lidar_points;
    float vertical_resolution = (vertical_angle_end - vertical_angle_start) / (vertical_lines - 1);
    std::vector<pcl::PointCloud<pcl::PointXYZ>> line_clouds(vertical_lines);

    auto start = std::chrono::high_resolution_clock::now();
#pragma omp parallel for
    for (int v = 0; v < vertical_lines; ++v) {
        float vertical_angle = vertical_angle_start + v * vertical_resolution;
        float sin_vert = std::sin(vertical_angle * M_PI / 180.0);
        float cos_vert = std::cos(vertical_angle * M_PI / 180.0);

        for (int h = 0; h < horizontal_num; ++h) {
            float horizontal_angle = h * horizontal_resolution;
            float sin_horz = std::sin(horizontal_angle * M_PI / 180.0);
            float cos_horz = std::cos(horizontal_angle * M_PI / 180.0);

            Eigen::Vector3f ray_direction(cos_vert * cos_horz, cos_vert * sin_horz, sin_vert);
            ray_direction = R_wc * ray_direction;
            Eigen::Vector3f ray_origin = pos;

            std::vector<int> pointIdxVec;
            if (octree->getIntersectedVoxelIndices(ray_origin, ray_direction, pointIdxVec, 1)) {
                pcl::PointXYZ point = cloud->points[pointIdxVec[0]];
                Eigen::Vector3f point_in_world = point.getVector3fMap();
                Eigen::Vector3f point_in_body = R_cw * (point_in_world - pos);
                if (max_lidar_dist > point_in_body.norm())
                    line_clouds[v].points.push_back(pcl::PointXYZ(point_in_body.x(), point_in_body.y(), point_in_body.z()));
            }
        }
    }
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    // std::cout << "生成雷达耗时: " << elapsed.count() << " 秒" << std::endl;

    for (int i = 0; i < vertical_lines; ++i) {
        lidar_points += line_clouds[i];
    }
    lidar_points.width = lidar_points.points.size();
    lidar_points.height = 1;
    lidar_points.is_dense = true;
    return lidar_points;
}

// TODO: 不能像python那样用一个向量一次性全算吗？
void SensorSimulator::expand_cloud(pcl::PointCloud<pcl::PointXYZ>::Ptr expanded_cloud, int direction) {
    auto start = std::chrono::high_resolution_clock::now();
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_temp(new pcl::PointCloud<pcl::PointXYZ>());
    *cloud_temp = *expanded_cloud;
    pcl::PointXYZ min_point, max_point;
    pcl::getMinMax3D(*expanded_cloud, min_point, max_point);

    float min_value = (direction == 0) ? min_point.x : min_point.y;
    float max_value = (direction == 0) ? max_point.x : max_point.y;

    // 镜像原始点云并添加到扩展点云中
    for (const auto& point : cloud_temp->points) {
        pcl::PointXYZ mirrored_point = point;
        if (direction == 0) {
            mirrored_point.x = 2 * min_value - point.x;  // 以 x 轴最小值为轴进行镜像
        } else {
            mirrored_point.y = 2 * min_value - point.y;  // 以 y 轴最小值为轴进行镜像
        }
        expanded_cloud->push_back(mirrored_point);
    }

    // 计算偏移量，保证方向上的最小值为0
    float offset = max_value - min_value;
    for (auto& point : expanded_cloud->points) {
        if (direction == 0) {
            point.x += offset;
        } else {
            point.y += offset;
        }
    }
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    // std::cout << "点云扩张一次耗时: " << elapsed.count() << " 秒" << std::endl; // 输出耗时
}


void SensorSimulator::timerDepthCallback(const ros::TimerEvent&) {
    if (!odom_init || !render_depth)
        return;
    cv::Mat depth_iamge = renderDepthImage();
    sensor_msgs::Image ros_image;
    cv_bridge::CvImage cv_image;
    cv_image.header.stamp = ros::Time::now();
    cv_image.encoding = sensor_msgs::image_encodings::TYPE_32FC1;
    cv_image.image = depth_iamge;
    cv_image.toImageMsg(ros_image);
    image_pub_.publish(ros_image);
}

void SensorSimulator::timerLidarCallback(const ros::TimerEvent&) {
    if (!odom_init || !render_lidar)
        return;
    pcl::PointCloud<pcl::PointXYZ> lidar_points = renderLidarPointcloud();
    sensor_msgs::PointCloud2 output;
    pcl::toROSMsg(lidar_points, output);
    output.header.stamp = ros::Time::now();
    output.header.frame_id = "odom";

    point_cloud_pub_.publish(output);
}

void SensorSimulator::odomCallback(const nav_msgs::Odometry::ConstPtr& msg) {
    quat.x() = msg->pose.pose.orientation.x;
    quat.y() = msg->pose.pose.orientation.y;
    quat.z() = msg->pose.pose.orientation.z;
    quat.w() = msg->pose.pose.orientation.w;

    pos.x() = msg->pose.pose.position.x;
    pos.y() = msg->pose.pose.position.y;
    pos.z() = msg->pose.pose.position.z;

    odom_init = true;
}

// 🟦 stage-5.B: receive a PoseArray where each Pose's position holds the
// ball world-frame xyz and orientation.w holds the radius (orientation.x/y/z
// reserved for future per-ball velocity).  An empty PoseArray clears the
// dynamic set, which is how scenarios get reset between runs.
void SensorSimulator::dynObsCallback(const geometry_msgs::PoseArray::ConstPtr& msg) {
    std::vector<DynSphereCPU> next;
    next.reserve(msg->poses.size());
    for (const auto &p : msg->poses) {
        DynSphereCPU s;
        s.pos    = Eigen::Vector3f(p.position.x, p.position.y, p.position.z);
        s.radius = static_cast<float>(p.orientation.w);
        if (s.radius > 0.0f) next.push_back(s);
    }
    std::lock_guard<std::mutex> g(dyn_mtx_);
    dyn_spheres_.swap(next);
}