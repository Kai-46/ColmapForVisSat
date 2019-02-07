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

#ifndef COLMAP_SRC_MVS_IMAGE_H_
#define COLMAP_SRC_MVS_IMAGE_H_

#include <cstdint>
#include <fstream>
#include <set>
#include <string>
#include <unordered_map>
#include <vector>

#include <Eigen/Core>

#include "util/bitmap.h"

namespace colmap {
namespace mvs {

// save high-precision matrices internally
// while output low-precision to be used in MVS
class Image {
 public:
//  Image();
  // if not passing P and inv_P, then they're automatically induced from K, R, t
  // in some cases, this might not be appropriate, as you might need to compute P and inv_P with high precision
  explicit Image(const std::string& path, const size_t width, const size_t height,
        const double K[9], const double R[9], const double T[3]);

  inline size_t GetWidth() const;
  inline size_t GetHeight() const;

  void SetBitmap(const Bitmap& bitmap);
  inline const Bitmap& GetBitmap() const;

  void SetLastRow(const double last_row[4]);
  inline const double *GetLastRow() const;

  void SetK(const double K[9]);

  inline const std::string& GetPath() const;

  void Rescale(const float factor);
  void Rescale(const float factor_x, const float factor_y);
  void Downsize(const size_t max_width, const size_t max_height);

  // high-precision output
  inline void GetCDouble(double C[3]);
  inline void GetKDouble(double K[9]);

  // low-precision output
  inline float GetDepth(double x, double y, double z);
  inline void GetK(float K[9]);
  inline void GetRT(float R[9],  float T[3]);
  inline void GetC(float C[3]);
  inline void GetPinvP(float P[16], float inv_P[16]);
  inline void Rotate90Multi(int cnt, float K[9], float R[9], float T[3], float P[16], float inv_P[16], float C[3]);
  inline void Original(float K[9], float R[9], float T[3], float P[16], float inv_P[16], float C[3]);
  inline void Rotate90(float K[9], float R[9], float T[3], float P[16], float inv_P[16], float C[3]);
  inline void Rotate180(float K[9], float R[9], float T[3], float P[16], float inv_P[16], float C[3]);
  inline void Rotate270(float K[9], float R[9], float T[3], float P[16], float inv_P[16], float C[3]);

 private:
  std::string path_;
  size_t width_;
  size_t height_;
  double K_[9];
  double R_[9];
  double T_[3];
  double last_row_[4] = {0., 0., 0., 1.};	// last row of the 4 by 4 projection matrix, default value
  Bitmap bitmap_;
};

// only useful to estimate homography
void ComputeRelativePose(const float R1[9], const float T1[3],
                         const float R2[9], const float T2[3], float R[9],
                         float T[3]) {
  const Eigen::Map<const Eigen::Matrix<float, 3, 3, Eigen::RowMajor>> R1_m(R1);
  const Eigen::Map<const Eigen::Matrix<float, 3, 3, Eigen::RowMajor>> R2_m(R2);
  const Eigen::Map<const Eigen::Matrix<float, 3, 1>> T1_m(T1);
  const Eigen::Map<const Eigen::Matrix<float, 3, 1>> T2_m(T2);
  Eigen::Map<Eigen::Matrix<float, 3, 3, Eigen::RowMajor>> R_m(R);
  Eigen::Map<Eigen::Vector3f> T_m(T);

  R_m = R2_m * R1_m.transpose();
  T_m = T2_m - R_m * T1_m;
}

}  // namespace mvs
}  // namespace colmap

#endif  // COLMAP_SRC_MVS_IMAGE_H_
