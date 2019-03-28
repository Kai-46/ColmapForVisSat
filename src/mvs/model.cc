// Copyright (c) 2018, ETH Zurich and UNC Chapel Hill.
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
//     * Redistributions of source code must retain the above copyright
//       notice, this list of conditions and the following disclaimer.
//
//     * Redistributions in binary form must reproduce the above copyright
//       notice, this list of conditions and the following disclaimer in the
//       documentation and/or other materials provided with the distribution.
//
//     * Neither the name of ETH Zurich and UNC Chapel Hill nor the names of
//       its contributors may be used to endorse or promote products derived
//       from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDERS OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.
//
// Author: Johannes L. Schoenberger (jsch-at-demuc-dot-de)

#include "mvs/model.h"

#include "base/camera_models.h"
#include "base/pose.h"
#include "base/projection.h"
#include "base/reconstruction.h"
#include "base/triangulation.h"
#include "util/misc.h"

#include <map>
#include <fstream>
#include <iomanip>

#define PRECISION 17

namespace colmap {
namespace mvs {

void Model::Read(const std::string& path, const std::string& format) {
  auto format_lower_case = format;
  StringToLower(&format_lower_case);
  if (format_lower_case == "colmap") {
    ReadFromCOLMAP(path);
//  } else if (format_lower_case == "pmvs") {
//    ReadFromPMVS(path);
  } else {
    LOG(FATAL) << "Invalid input format";
  }
}

void Model::ReadFromCOLMAP(const std::string& path) {
  Reconstruction reconstruction;
  reconstruction.Read(JoinPaths(path, "sparse"));

  // std::cout << "line 64 at model.cc" << std::endl;

  // read-in the last row of the 4 by 4 matrices here
  std::map<std::string, double *> last_rows;
  std::ifstream infile;
  infile.open(JoinPaths(path, "last_rows.txt"));
  std::string image_name;
  double vec[4];
  while (infile >> image_name >> vec[0] >> vec[1] >> vec[2] >> vec[3]) {
	  double* vec_ptr = new double[4];
	  memcpy(vec_ptr, vec, 4*sizeof(double));
	  last_rows[image_name] = vec_ptr;
  }

  // std::cout << "line 78 at model.cc" << std::endl;

  images.reserve(reconstruction.NumRegImages());
  std::unordered_map<image_t, size_t> image_id_to_idx;
  for (size_t i = 0; i < reconstruction.NumRegImages(); ++i) {
    const auto image_id = reconstruction.RegImageIds()[i];
    const auto& image = reconstruction.Image(image_id);

    // check whether the image exists in last_rows
    // if (last_rows.find(image.Name()) == last_rows.end()) {
    //     std::cout << "ignoring image: " << image.Name() << ", because no last_row exists" << std::endl;
    //     continue;
    // }

    const auto& camera = reconstruction.Camera(image.CameraId());

    CHECK_EQ(camera.ModelId(), PinholeCameraModel::model_id);

    const std::string image_path = JoinPaths(path, "images", image.Name());
    const Eigen::Matrix<double, 3, 3, Eigen::RowMajor> K = camera.CalibrationMatrix();
    const Eigen::Matrix<double, 3, 3, Eigen::RowMajor> R = QuaternionToRotationMatrix(image.Qvec());
    const Eigen::Vector3d T = image.Tvec();

    images.emplace_back(image_path, camera.Width(), camera.Height(), K.data(),
                        R.data(), T.data());

    // images.push_back(Image(image_path, camera.Width(), camera.Height(), K.data(),
    //                     R.data(), T.data()));

    // std::cout << "line 97 at model.cc" << std::endl;

    // std::cout << "image_name: " << image.Name() << std::endl;
    // std::cout << "find image?: " << (last_rows.find(image.Name()) != last_rows.end()) << std::endl;

    // set last row
    //    check whether the image exists in last_rows
    if (last_rows.find(image.Name()) == last_rows.end()) {
        std::cout << "setting the last row for image: " << image.Name() << " to be (0, 0, 0, 1), because no last_row exists" << std::endl;
        double default_last_row[4] = {0., 0., 0., 1.};
        images.back().SetLastRow(default_last_row);
    } else {
        images.back().SetLastRow(last_rows.find(image.Name())->second);
    }

    // debug test rotation
    // debug
//    std::cout << image_path << std::endl;
//    for (int i =0; i<4; ++i) {
//    	images.back().Rotate90Multi_test(i);
//    }

    image_id_to_idx.emplace(image_id, i);
    image_names_.push_back(image.Name());
    image_name_to_idx_.emplace(image.Name(), i);
  }

  //std::cout << "line 107 at model.cc" << std::endl;
  // free memory
  for (std::map<std::string, double *>::iterator it=last_rows.begin(); it != last_rows.end(); ++it) {
	  delete [] it->second;
  }

  points.reserve(reconstruction.NumPoints3D());
  for (const auto& point3D : reconstruction.Points3D()) {
    Point point;
    point.x = point3D.second.X();
    point.y = point3D.second.Y();
    point.z = point3D.second.Z();
    point.track.reserve(point3D.second.Track().Length());
    for (const auto& track_el : point3D.second.Track().Elements()) {
      point.track.push_back(image_id_to_idx.at(track_el.image_id));
    }
    points.push_back(point);
  }

  // write image_name_to_idx to file
  std::ofstream	img_idx2name_file(JoinPaths(path, "img_idx2name.txt"), std::ios::trunc);
  for(int i=0; i < image_names_.size(); ++i) {
	  img_idx2name_file << i << " " << image_names_[i] << "\n";
  }
  img_idx2name_file.close();

  // write projection matrices and inverse projection matrices to files
  std::ofstream P_file(JoinPaths(path, "proj_mats.txt"), std::ios::trunc);
    // set fulll precision
  P_file << std::setprecision(PRECISION);

  std::ofstream inv_P_file(JoinPaths(path, "inv_proj_mats.txt"), std::ios::trunc);
    // set fulll precision
  inv_P_file << std::setprecision(PRECISION);

  for (const auto& image: images) {
	  std::string img_path = image.GetPath();
	  std::string img_name = img_path.substr(img_path.rfind('/')+1);
	  // get projection matrices
	  double P[16];
	  double inv_P[16];
	  image.GetPinvPDouble(P, inv_P);

	  P_file << img_name;
	  inv_P_file << img_name;
	  for (int i=0; i<16; ++i) {
		  P_file << " " << P[i];
		  inv_P_file << " " << inv_P[i];
	  }
	  P_file << '\n';
	  inv_P_file << '\n';
  }

  P_file.close();
  inv_P_file.close();
}

//void Model::ReadFromPMVS(const std::string& path) {
//  if (ReadFromBundlerPMVS(path)) {
//    return;
//  } else if (ReadFromRawPMVS(path)) {
//    return;
//  } else {
//    LOG(FATAL) << "Invalid PMVS format";
//  }
//}

int Model::GetImageIdx(const std::string& name) const {
  CHECK_GT(image_name_to_idx_.count(name), 0)
      << "Image with name `" << name << "` does not exist";
  return image_name_to_idx_.at(name);
}

std::string Model::GetImageName(const int image_idx) const {
  CHECK_GE(image_idx, 0);
  CHECK_LT(image_idx, image_names_.size());
  return image_names_.at(image_idx);
}

std::vector<std::vector<int>> Model::GetMaxOverlappingImages(
    const size_t num_images, const double min_triangulation_angle) const {
  std::vector<std::vector<int>> overlapping_images(images.size());

  const float min_triangulation_angle_rad = DegToRad(min_triangulation_angle);

  const auto shared_num_points = ComputeSharedPoints();

  const float kTriangulationAnglePercentile = 75;
  const auto triangulation_angles =
      ComputeTriangulationAngles(kTriangulationAnglePercentile);

  for (size_t image_idx = 0; image_idx < images.size(); ++image_idx) {
    const auto& shared_images = shared_num_points.at(image_idx);
    const auto& overlapping_triangulation_angles =
        triangulation_angles.at(image_idx);

    std::vector<std::pair<int, int>> ordered_images;
    ordered_images.reserve(shared_images.size());
    for (const auto& image : shared_images) {
      if (overlapping_triangulation_angles.at(image.first) >=
          min_triangulation_angle_rad) {
        ordered_images.emplace_back(image.first, image.second);
      }
    }

    const size_t eff_num_images = std::min(ordered_images.size(), num_images);
    if (eff_num_images < shared_images.size()) {
      std::partial_sort(ordered_images.begin(),
                        ordered_images.begin() + eff_num_images,
                        ordered_images.end(),
                        [](const std::pair<int, int> image1,
                           const std::pair<int, int> image2) {
                          return image1.second > image2.second;
                        });
    } else {
      std::sort(ordered_images.begin(), ordered_images.end(),
                [](const std::pair<int, int> image1,
                   const std::pair<int, int> image2) {
                  return image1.second > image2.second;
                });
    }

    overlapping_images[image_idx].reserve(eff_num_images);
    for (size_t i = 0; i < eff_num_images; ++i) {
      overlapping_images[image_idx].push_back(ordered_images[i].first);
    }
  }

  return overlapping_images;
}

//const std::vector<std::vector<int>>& Model::GetMaxOverlappingImagesFromPMVS()
//    const {
//  return pmvs_vis_dat_;
//}

// need to modify here
// depth is now defined as the inverse of the fourth component
std::vector<std::pair<float, float>> Model::ComputeDepthRanges() const {
  std::vector<std::vector<float>> depths(images.size());
  for (const auto& point : points) {
     for (const auto& image_idx : point.track) {
      const auto& image = images.at(image_idx);
      // make sure single-precision is enough for depth
      const float depth = image.GetDepth(point.x, point.y, point.z);
//      if (depth > 0) {
      depths[image_idx].push_back(depth);
//      }
    }
  }

  std::vector<std::pair<float, float>> depth_ranges(depths.size());
  for (size_t image_idx = 0; image_idx < depth_ranges.size(); ++image_idx) {
    auto& depth_range = depth_ranges[image_idx];

    auto& image_depths = depths[image_idx];

    if (image_depths.empty()) {
      // set to an absurd value
      depth_range.first = -1e20f;
      depth_range.second = -1e20f;
      continue;
    }

    std::sort(image_depths.begin(), image_depths.end());

//     const float kMinPercentile = 0.01f;
//     const float kMaxPercentile = 0.99f;
    
    // let's be more conservative
    const float kMinPercentile = 0.01f;
    const float kMaxPercentile = 0.99f;
    
    depth_range.first = image_depths[image_depths.size() * kMinPercentile];
    depth_range.second = image_depths[image_depths.size() * kMaxPercentile];

//     const float kStretchRatio = 0.25f;
//     depth_range.first *= (1.0f - kStretchRatio);
//     depth_range.second *= (1.0f + kStretchRatio);
      
    // one should stretch the portion of depth range span
//    const float kStretchRatio = 0.1;
//    float stretch = kStretchRatio * (depth_range.second - depth_range.first);
//    depth_range.first -= stretch;
//    depth_range.second += stretch;
    
    // compute inv_depth uncertainty
    //const float inv_depth_range = 1.0f / depth_range.first - 1.0f / depth_range.second;
    // const float kStretchRatio = 0.2;
    // const float stretch = kStretchRatio * inv_depth_range;

    const float lower_stretch = 10;	// 10 meters
    const float upper_stretch = 20;

    const float min_depth_new = depth_range.first - lower_stretch;
    const float max_depth_new = depth_range.second + upper_stretch;
    if (max_depth_new > min_depth_new) {
        depth_range.first = min_depth_new;
        depth_range.second = max_depth_new;
        std::cout << "image id: " << image_idx << ", depth range: "
        		<< depth_range.first << ", " << depth_range.second
				<< ", lower_stretch: " << lower_stretch << ", upper_stretch: " << upper_stretch << std::endl;
    } else {
    	std::cout << "image id: " << image_idx << ", depth range: "
        		<< depth_range.first << ", " << depth_range.second <<
    			", depth range cannot be stretched" << std::endl;
    }
  }

  return depth_ranges;
}

std::vector<std::map<int, int>> Model::ComputeSharedPoints() const {
  std::vector<std::map<int, int>> shared_points(images.size());
  for (const auto& point : points) {
    for (size_t i = 0; i < point.track.size(); ++i) {
      const int image_idx1 = point.track[i];
      for (size_t j = 0; j < i; ++j) {
        const int image_idx2 = point.track[j];
        if (image_idx1 != image_idx2) {
          shared_points.at(image_idx1)[image_idx2] += 1;
          shared_points.at(image_idx2)[image_idx1] += 1;
        }
      }
    }
  }
  return shared_points;
}

std::vector<std::map<int, float>> Model::ComputeTriangulationAngles(
    const float percentile) const {
  std::vector<Eigen::Vector3d> proj_centers(images.size());
  for (size_t image_idx = 0; image_idx < images.size(); ++image_idx) {
    const auto& image = images[image_idx];
    double C[3];
    image.GetCDouble(C);
    proj_centers[image_idx] = Eigen::Map<const Eigen::Vector3d>(C);
  }

  std::vector<std::map<int, std::vector<float>>> all_triangulation_angles(
      images.size());
  for (const auto& point : points) {
    for (size_t i = 0; i < point.track.size(); ++i) {
      const int image_idx1 = point.track[i];
      for (size_t j = 0; j < i; ++j) {
        const int image_idx2 = point.track[j];
        if (image_idx1 != image_idx2) {
          const float angle = CalculateTriangulationAngle(
              proj_centers.at(image_idx1), proj_centers.at(image_idx2),
              Eigen::Vector3d(point.x, point.y, point.z));
          all_triangulation_angles.at(image_idx1)[image_idx2].push_back(angle);
          all_triangulation_angles.at(image_idx2)[image_idx1].push_back(angle);
        }
      }
    }
  }

  std::vector<std::map<int, float>> triangulation_angles(images.size());
  for (size_t image_idx = 0; image_idx < all_triangulation_angles.size();
       ++image_idx) {
    const auto& overlapping_images = all_triangulation_angles[image_idx];
    for (const auto& image : overlapping_images) {
      triangulation_angles[image_idx].emplace(
          image.first, Percentile(image.second, percentile));
    }
  }

  return triangulation_angles;
}

//bool Model::ReadFromBundlerPMVS(const std::string& path) {
//  const std::string bundle_file_path = JoinPaths(path, "bundle.rd.out");
//
//  if (!ExistsFile(bundle_file_path)) {
//    return false;
//  }
//
//  std::ifstream file(bundle_file_path);
//  CHECK(file.is_open()) << bundle_file_path;
//
//  // Header line.
//  std::string header;
//  std::getline(file, header);
//
//  int num_images, num_points;
//  file >> num_images >> num_points;
//
//  images.reserve(num_images);
//  for (int image_idx = 0; image_idx < num_images; ++image_idx) {
//    const std::string image_name = StringPrintf("%08d.jpg", image_idx);
//    const std::string image_path = JoinPaths(path, "visualize", image_name);
//
//    float K[9] = {1.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 1.0f};
//    file >> K[0];
//    K[4] = K[0];
//
//    Bitmap bitmap;
//    CHECK(bitmap.Read(image_path));
//    K[2] = bitmap.Width() / 2.0f;
//    K[5] = bitmap.Height() / 2.0f;
//
//    float k1, k2;
//    file >> k1 >> k2;
//    CHECK_EQ(k1, 0.0f);
//    CHECK_EQ(k2, 0.0f);
//
//    float R[9];
//    for (size_t i = 0; i < 9; ++i) {
//      file >> R[i];
//    }
//    for (size_t i = 3; i < 9; ++i) {
//      R[i] = -R[i];
//    }
//
//    float T[3];
//    file >> T[0] >> T[1] >> T[2];
//    T[1] = -T[1];
//    T[2] = -T[2];
//
//    images.emplace_back(image_path, bitmap.Width(), bitmap.Height(), K, R, T);
//    // images.push_back(Image(image_path, bitmap.Width(), bitmap.Height(), K, R, T));
//    image_names_.push_back(image_name);
//    image_name_to_idx_.emplace(image_name, image_idx);
//  }
//
//  points.resize(num_points);
//  for (int point_id = 0; point_id < num_points; ++point_id) {
//    auto& point = points[point_id];
//
//    file >> point.x >> point.y >> point.z;
//
//    int color[3];
//    file >> color[0] >> color[1] >> color[2];
//
//    int track_len;
//    file >> track_len;
//    point.track.resize(track_len);
//
//    for (int i = 0; i < track_len; ++i) {
//      int feature_idx;
//      float imx, imy;
//      file >> point.track[i] >> feature_idx >> imx >> imy;
//      CHECK_LT(point.track[i], images.size());
//    }
//  }
//
//  return true;
//}
//
//bool Model::ReadFromRawPMVS(const std::string& path) {
//  const std::string vis_dat_path = JoinPaths(path, "vis.dat");
//  if (!ExistsFile(vis_dat_path)) {
//    return false;
//  }
//
//  for (int image_idx = 0;; ++image_idx) {
//    const std::string image_name = StringPrintf("%08d.jpg", image_idx);
//    const std::string image_path = JoinPaths(path, "visualize", image_name);
//
//    if (!ExistsFile(image_path)) {
//      break;
//    }
//
//    Bitmap bitmap;
//    CHECK(bitmap.Read(image_path));
//
//    const std::string proj_matrix_path =
//        JoinPaths(path, "txt", StringPrintf("%08d.txt", image_idx));
//
//    std::ifstream proj_matrix_file(proj_matrix_path);
//    CHECK(proj_matrix_file.is_open()) << proj_matrix_path;
//
//    std::string contour;
//    proj_matrix_file >> contour;
//    CHECK_EQ(contour, "CONTOUR");
//
//    Eigen::Matrix3x4d P;
//    for (int i = 0; i < 3; ++i) {
//      proj_matrix_file >> P(i, 0) >> P(i, 1) >> P(i, 2) >> P(i, 3);
//    }
//
//    Eigen::Matrix3d K;
//    Eigen::Matrix3d R;
//    Eigen::Vector3d T;
//    DecomposeProjectionMatrix(P, &K, &R, &T);
//
//    // The COLMAP patch match algorithm requires that there is no skew.
//    K(0, 1) = 0.0f;
//    K(1, 0) = 0.0f;
//    K(2, 0) = 0.0f;
//    K(2, 1) = 0.0f;
//    K(2, 2) = 1.0f;
//
//    const Eigen::Matrix<float, 3, 3, Eigen::RowMajor> K_float = K.cast<float>();
//    const Eigen::Matrix<float, 3, 3, Eigen::RowMajor> R_float = R.cast<float>();
//    const Eigen::Vector3f T_float = T.cast<float>();
//
//    images.emplace_back(image_path, bitmap.Width(), bitmap.Height(),
//                        K_float.data(), R_float.data(), T_float.data());
//    image_names_.push_back(image_name);
//    image_name_to_idx_.emplace(image_name, image_idx);
//  }
//
//  std::ifstream vis_dat_file(vis_dat_path);
//  CHECK(vis_dat_file.is_open()) << vis_dat_path;
//
//  std::string visdata;
//  vis_dat_file >> visdata;
//  CHECK_EQ(visdata, "VISDATA");
//
//  int num_images;
//  vis_dat_file >> num_images;
//  CHECK_GE(num_images, 0);
//  CHECK_EQ(num_images, images.size());
//
//  pmvs_vis_dat_.resize(num_images);
//  for (int i = 0; i < num_images; ++i) {
//    int image_idx;
//    vis_dat_file >> image_idx;
//    CHECK_GE(image_idx, 0);
//    CHECK_LT(image_idx, num_images);
//
//    int num_visible_images;
//    vis_dat_file >> num_visible_images;
//
//    auto& visible_image_idxs = pmvs_vis_dat_[image_idx];
//    visible_image_idxs.reserve(num_visible_images);
//
//    for (int j = 0; j < num_visible_images; ++j) {
//      int visible_image_idx;
//      vis_dat_file >> visible_image_idx;
//      CHECK_GE(visible_image_idx, 0);
//      CHECK_LT(visible_image_idx, num_images);
//      if (visible_image_idx != image_idx) {
//        visible_image_idxs.push_back(visible_image_idx);
//      }
//    }
//  }
//
//  return true;
//}

}  // namespace mvs
}  // namespace colmap
