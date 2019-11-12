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

#ifndef COLMAP_SRC_MVS_PATCH_MATCH_CUDA_H_
#define COLMAP_SRC_MVS_PATCH_MATCH_CUDA_H_

#include <iostream>
#include <memory>
#include <vector>

#include <cuda_runtime.h>

#include "mvs/cuda_array_wrapper.h"
#include "mvs/depth_map.h"
#include "mvs/gpu_mat.h"
#include "mvs/gpu_mat_prng.h"
#include "mvs/gpu_mat_ref_image.h"
#include "mvs/image.h"
#include "mvs/normal_map.h"
#include "mvs/patch_match.h"

namespace colmap {
namespace mvs {

class PatchMatchCuda {
 public:
  PatchMatchCuda(const PatchMatchOptions& options,
                 const PatchMatch::Problem& problem);
  ~PatchMatchCuda();

  void Run();

  DepthMap GetDepthMap() const;
  NormalMap GetNormalMap() const;
  Mat<float> GetSelProbMap() const;
  std::vector<int> GetConsistentImageIdxs() const;

 private:
  template <int kWindowSize, int kWindowStep>
  void RunWithWindowSizeAndStep();

  void ComputeCudaConfig();

  void InitRefImage();
  void InitSourceImages();
  void InitTransforms();
  void InitWorkspaceMemory();

  // Rotate reference image by 90 degrees in counter-clockwise direction.
  void Rotate();

  const PatchMatchOptions options_;
  const PatchMatch::Problem problem_;

  // Dimensions for sweeping from top to bottom, i.e. one thread per column.
  dim3 sweep_block_size_;
  dim3 sweep_grid_size_;
  // Dimensions for element-wise operations, i.e. one thread per pixel.
  dim3 elem_wise_block_size_;
  dim3 elem_wise_grid_size_;

  // Original (not rotated) dimension of reference image.
  size_t ref_width_;
  size_t ref_height_;

  // Rotation of reference image in pi/2. This is equivalent to the number of
  // calls to `rotate` mod 4.
  int rotation_in_half_pi_;

  std::unique_ptr<CudaArrayWrapper<uint8_t>> ref_image_device_;
  std::unique_ptr<CudaArrayWrapper<uint8_t>> src_images_device_;
  std::unique_ptr<CudaArrayWrapper<float>> src_depth_maps_device_;

  // Source images
  //
  //    [S(1), S(2), S(3), ..., S(n)]
  //
  // where n is the number of source images and:
  //
  //    S(i) = [P(:), P^-1(:), C_i(:)]
  //
  // where i denotes the index of the source image; P, P^-1 denote its 4 by 4 projection, and inverse projection matrices;
  // C is its camera center

  std::unique_ptr<CudaArrayWrapper<float>> poses_device_;

  // Calibration matrix for rotated versions of reference image
  // as {K[0, 0], K[0, 1], K[0, 2], K[1, 0], K[1, 1], K[1, 2]} corresponding to _rotationInHalfPi.
  float ref_K_host_[4][6];
  float ref_inv_K_host_[4][6];

  // Extrinsics for rotated versions of reference image
  float ref_R_host_[4][9];
  float ref_T_host_[4][3];

  // Projection center of the reference image in scene coordinate frame
  // rotation does not affect the projection center
  float ref_C_host_[3];

  // max depth difference for adjacent pixels during depth propagation
  float max_dist_per_pixel_host_ = 1.0; // 1.0 meters

  // Projection matrix and inverse projection matrix for rotated versions of reference image
  float ref_P_host_[4][16];
  float ref_inv_P_host_[4][16];

  // Data for reference image.
  std::unique_ptr<GpuMatRefImage> ref_image_;
  std::unique_ptr<GpuMat<float>> depth_map_;
  std::unique_ptr<GpuMat<float>> normal_map_;
  std::unique_ptr<GpuMat<float>> sel_prob_map_;
  std::unique_ptr<GpuMat<float>> prev_sel_prob_map_;
  std::unique_ptr<GpuMat<float>> cost_map_;
  std::unique_ptr<GpuMatPRNG> rand_state_map_;
  std::unique_ptr<GpuMat<uint8_t>> consistency_mask_;

  // Shared memory is too small to hold local state for each thread,
  // so this is workspace memory in global memory.
  std::unique_ptr<GpuMat<float>> global_workspace_;
};

}  // namespace mvs
}  // namespace colmap

#endif  // COLMAP_SRC_MVS_PATCH_MATCH_CUDA_H_
