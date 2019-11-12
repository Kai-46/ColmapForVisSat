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

#include "mvs/workspace.h"

#include <numeric>

#include "util/misc.h"

namespace colmap {
namespace mvs {

Workspace::CachedImage::CachedImage() {}

Workspace::CachedImage::CachedImage(CachedImage&& other) {
  num_bytes = other.num_bytes;
  bitmap = std::move(other.bitmap);
  depth_map = std::move(other.depth_map);
  normal_map = std::move(other.normal_map);
}

Workspace::CachedImage& Workspace::CachedImage::operator=(CachedImage&& other) {
  if (this != &other) {
    num_bytes = other.num_bytes;
    bitmap = std::move(other.bitmap);
    depth_map = std::move(other.depth_map);
    normal_map = std::move(other.normal_map);
  }
  return *this;
}

size_t Workspace::CachedImage::NumBytes() const { return num_bytes; }

Workspace::Workspace(const Options& options)
    : options_(options),
      cache_(1024 * 1024 * 1024 * options_.cache_size,
             [](const int) { return CachedImage(); }) {
  StringToLower(&options_.input_type);
  model_.Read(options_.workspace_path, options_.workspace_format);
  if (options_.max_image_size > 0) {
    for (auto& image : model_.images) {
      image.Downsize(options_.max_image_size, options_.max_image_size);
    }
  }

  depth_map_path_ = EnsureTrailingSlash(
      JoinPaths(options_.workspace_path, options_.stereo_folder, "depth_maps"));
  normal_map_path_ = EnsureTrailingSlash(JoinPaths(
      options_.workspace_path, options_.stereo_folder, "normal_maps"));
}

void Workspace::ClearCache() { cache_.Clear(); }

const Workspace::Options& Workspace::GetOptions() const { return options_; }

const Model& Workspace::GetModel() const { return model_; }

const Bitmap& Workspace::GetBitmap(const int image_idx) {
  auto& cached_image = cache_.GetMutable(image_idx);
  if (!cached_image.bitmap) {
    cached_image.bitmap.reset(new Bitmap());
    cached_image.bitmap->Read(GetBitmapPath(image_idx), options_.image_as_rgb);
    if (options_.max_image_size > 0) {
      cached_image.bitmap->Rescale(model_.images.at(image_idx).GetWidth(),
                                   model_.images.at(image_idx).GetHeight());
    }
    cached_image.num_bytes += cached_image.bitmap->NumBytes();
    cache_.UpdateNumBytes(image_idx);
  }
  return *cached_image.bitmap;
}

const DepthMap& Workspace::GetDepthMap(const int image_idx) {
  auto& cached_image = cache_.GetMutable(image_idx);
  if (!cached_image.depth_map) {
    cached_image.depth_map.reset(new DepthMap());
    cached_image.depth_map->Read(GetDepthMapPath(image_idx));
    if (options_.max_image_size > 0) {
      cached_image.depth_map->Downsize(model_.images.at(image_idx).GetWidth(),
                                       model_.images.at(image_idx).GetHeight());
    }
    cached_image.num_bytes += cached_image.depth_map->GetNumBytes();
    cache_.UpdateNumBytes(image_idx);
  }
  return *cached_image.depth_map;
}

const NormalMap& Workspace::GetNormalMap(const int image_idx) {
  auto& cached_image = cache_.GetMutable(image_idx);
  if (!cached_image.normal_map) {
    cached_image.normal_map.reset(new NormalMap());
    cached_image.normal_map->Read(GetNormalMapPath(image_idx));
    if (options_.max_image_size > 0) {
      cached_image.normal_map->Downsize(
          model_.images.at(image_idx).GetWidth(),
          model_.images.at(image_idx).GetHeight());
    }
    cached_image.num_bytes += cached_image.normal_map->GetNumBytes();
    cache_.UpdateNumBytes(image_idx);
  }
  return *cached_image.normal_map;
}

std::string Workspace::GetBitmapPath(const int image_idx) const {
  return model_.images.at(image_idx).GetPath();
}

std::string Workspace::GetDepthMapPath(const int image_idx) const {
  return depth_map_path_ + GetFileName(image_idx);
}

std::string Workspace::GetNormalMapPath(const int image_idx) const {
  return normal_map_path_ + GetFileName(image_idx);
}

bool Workspace::HasBitmap(const int image_idx) const {
  return ExistsFile(GetBitmapPath(image_idx));
}

bool Workspace::HasDepthMap(const int image_idx) const {
  return ExistsFile(GetDepthMapPath(image_idx));
}

bool Workspace::HasNormalMap(const int image_idx) const {
  return ExistsFile(GetNormalMapPath(image_idx));
}

std::string Workspace::GetFileName(const int image_idx) const {
  const auto& image_name = model_.GetImageName(image_idx);
  return StringPrintf("%s.%s.bin", image_name.c_str(),
                      options_.input_type.c_str());
}


}  // namespace mvs
}  // namespace colmap
