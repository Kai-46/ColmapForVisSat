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

#ifndef COLMAP_SRC_MVS_IMAGE_H_
#define COLMAP_SRC_MVS_IMAGE_H_

#include <cstdint>
#include <fstream>
#include <set>
#include <string>
#include <unordered_map>
#include <vector>

#include "util/bitmap.h"

namespace colmap {
namespace mvs {

// save high-precision matrices internally
// while output low-precision to be used in MVS
class Image {
 public:
//  Image();
  Image(const std::string& path, const size_t width, const size_t height,
        const double K[9], const double R[9], const double T[3]);

  size_t GetWidth() const;
  size_t GetHeight() const;

  void SetBitmap(const Bitmap& bitmap);
  const Bitmap& GetBitmap() const;

  void SetLastRow(const double last_row[4]);
  const double *GetLastRow() const;

  void SetK(const double K[9]) const;

  const std::string& GetPath() const;

  void Rescale(const float factor);
  void Rescale(const float factor_x, const float factor_y);
  void Downsize(const size_t max_width, const size_t max_height);

  // high-precision output
  void GetCDouble(double C[3]) const;
  void GetKDouble(double K[9]) const;
  void GetPinvPDouble(double P[16], double inv_P[16]) const;

  // low-precision output
  float GetDepth(double x, double y, double z) const;
  void GetK(float K[9]) const;
  void GetRT(float R[9],  float T[3]) const;
  void GetC(float C[3]) const;
  void GetPinvP(float P[16], float inv_P[16]) const;

//  void Rotate90Multi_test(int cnt) const;
  void Rotate90Multi(int cnt, float K[9], float inv_K[9], float R[9], float T[3], float P[16], float inv_P[16], float C[3]) const;
  void Original(float K[9], float inv_K[9], float R[9], float T[3], float P[16], float inv_P[16], float C[3]) const;
  void Rotate90(float K[9], float inv_K[9],float R[9], float T[3], float P[16], float inv_P[16], float C[3]) const;
  void Rotate180(float K[9], float inv_K[9], float R[9], float T[3], float P[16], float inv_P[16], float C[3]) const;
  void Rotate270(float K[9], float inv_K[9], float R[9], float T[3], float P[16], float inv_P[16], float C[3]) const;

 private:
  std::string path_;
  size_t width_;
  size_t height_;
  mutable double K_[9];
  double R_[9];
  double T_[3];
  double last_row_[4];	// last row of the 4 by 4 projection matrix
  Bitmap bitmap_;
};


}  // namespace mvs
}  // namespace colmap

#endif  // COLMAP_SRC_MVS_IMAGE_H_
