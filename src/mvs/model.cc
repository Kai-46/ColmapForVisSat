// ===============================================================================================================
// Copyright (c) 2019, Cornell University. All rights reserved.
//
// Redistribution and use in source and binary forms, with or without modification, are permitted provided that
// the following conditions are met:
//
//     * Redistributions of source code must retain the above copyright otice, this list of conditions and
//       the following disclaimer.
//
//     * Redistributions in binary form must reproduce the above copyright notice, this list of conditions and
//       the following disclaimer in the documentation and/or other materials provided with the distribution.
//       
//     * Neither the name of Cornell University nor the names of its contributors may be used to endorse or
//       promote products derived from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED 
// WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
// A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDERS OR CONTRIBUTORS BE LIABLE
// FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED
// TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
// HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING 
// NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY
// OF SUCH DAMAGE.
//
// Author: Kai Zhang (kz298@cornell.edu)
//
// The research is based upon work supported by the Office of the Director of National Intelligence (ODNI),     
// Intelligence Advanced Research Projects Activity (IARPA), via DOI/IBC Contract Number D17PC00287.            
// The U.S. Government is authorized to reproduce and distribute copies of this work for Governmental purposes. 
// ===============================================================================================================
//
//
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
  } else {
    LOG(FATAL) << "Invalid input format";
  }
}

void Model::ReadFromCOLMAP(const std::string& path) {
  Reconstruction reconstruction;
  reconstruction.Read(JoinPaths(path, "sparse"));

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
  infile.close();

  images.reserve(reconstruction.NumRegImages());
  std::unordered_map<image_t, size_t> image_id_to_idx;
  for (size_t i = 0; i < reconstruction.NumRegImages(); ++i) {
    const auto image_id = reconstruction.RegImageIds()[i];
    const auto& image = reconstruction.Image(image_id);

    const auto& camera = reconstruction.Camera(image.CameraId());

//    CHECK_EQ(camera.ModelId(), PinholeCameraModel::model_id);
    CHECK(camera.ModelId() == PinholeCameraModel::model_id ||
        camera.ModelId() == PerspectiveCameraModel::model_id);

    const Eigen::Matrix<double, 3, 3, Eigen::RowMajor> K = camera.CalibrationMatrix();
    const Eigen::Matrix<double, 3, 3, Eigen::RowMajor> R = QuaternionToRotationMatrix(image.Qvec());
    const Eigen::Vector3d T = image.Tvec();

    const std::string image_path = JoinPaths(path, "images", image.Name());
    images.emplace_back(image_path, camera.Width(), camera.Height(), K.data(),
                        R.data(), T.data());

    // set last row
    // check whether the image exists in last_rows
    if (last_rows.find(image.Name()) == last_rows.end()) {
      std::cout << "setting the last row for image: " << image.Name() << " to be (0, 0, 0, 1), because no last_row exists" << std::endl;
      double default_last_row[4] = {0., 0., 0., 1.};
      images.back().SetLastRow(default_last_row);
    } else {
      images.back().SetLastRow(last_rows.find(image.Name())->second);
    }

    image_id_to_idx.emplace(image_id, i);
    image_names_.push_back(image.Name());
    image_name_to_idx_.emplace(image.Name(), i);
  }

  // free memory
  for (auto it=last_rows.begin(); it != last_rows.end(); ++it) {
    delete [] it->second;
  }

  // read depth ranges
  for (size_t i=0; i<image_names_.size(); ++i) {
    depth_ranges_.emplace_back(-1e20f, -1e20f);
  }

  infile.open(JoinPaths(path, "depth_ranges.txt"));
  while (infile >> image_name >> vec[0] >> vec[1]) {
    depth_ranges_[image_name_to_idx_[image_name]] = std::make_pair (vec[0],vec[1]);
  }
  infile.close();

  // parse sparse points
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


}  // namespace mvs
}  // namespace colmap
