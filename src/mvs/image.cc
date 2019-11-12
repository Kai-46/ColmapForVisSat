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

#include "mvs/image.h"

#include <Eigen/Core>

#include "base/projection.h"
#include "util/logging.h"

namespace colmap {
namespace mvs {

// helper function
inline void Compute4by4ProjectionMatrix(const double K[9],
		const double R[9],
		const double T[9],
		const double last_row[4],
		double P[16], double inv_P[16]) {
	  // 3 by 4 projection matrix
	  Eigen::Matrix<double, 3, 4, Eigen::RowMajor> P_3by4;
	  P_3by4.leftCols<3>() = Eigen::Map<const Eigen::Matrix<double, 3, 3, Eigen::RowMajor>>(R);
	  P_3by4.rightCols<1>() = Eigen::Map<const Eigen::Matrix<double, 3, 1>>(T);
	  P_3by4 = Eigen::Map<const Eigen::Matrix<double, 3, 3, Eigen::RowMajor>>(K) * P_3by4;

	  // 4 by 4 projection matrix
	  Eigen::Matrix<double, 4, 4, Eigen::RowMajor> P_4by4;
	  P_4by4.block<3, 4>(0, 0) = P_3by4;
	  P_4by4.row(3) = Eigen::Map<const Eigen::Matrix<double, 1, 4>>(last_row);

	  // try to make the projection matrix more numerically stable
	  // scale all the numbers to lie in [0, 10]
	  P_4by4 *= (10.0 / P_4by4.maxCoeff());

	  memcpy(P, P_4by4.data(), 16 * sizeof(double));

	  Eigen::Matrix<double, 4, 4, Eigen::RowMajor> inv_P_4by4 = P_4by4.inverse();

	  inv_P_4by4 *= (10.0 / inv_P_4by4.maxCoeff());

	  memcpy(inv_P, inv_P_4by4.data(), 16 * sizeof(double));
}

// helper function
inline void ComputeProjectionCenter(const double R[9], const double T[9],
									double C[3]) {
	  // 3 by 4 projection matrix
	  Eigen::Map<const Eigen::Matrix<double, 3, 3, Eigen::RowMajor>> R_mat(R);
	  Eigen::Map<const Eigen::Matrix<double, 3, 1>> T_mat(T);
	  const Eigen::Matrix<double, 3, 1> C_mat = -R_mat.transpose() * T_mat;

	  memcpy(C, C_mat.data(), 3 * sizeof(double));
}

// helper function
inline void DoubleArrToFloatArr(const double* double_arr, float* float_arr, int cnt) {
	for (int i=0; i<cnt; ++i) {
		float_arr[i] = (float) double_arr[i];
	}
}


Image::Image(const std::string& path, const size_t width, const size_t height,
             const double K[9], const double R[9], const double T[3])
    : path_(path), width_(width), height_(height) {
  memcpy(K_, K, 9 * sizeof(double));
  memcpy(R_, R, 9 * sizeof(double));
  memcpy(T_, T, 3 * sizeof(double));
}

void Image::SetBitmap(const Bitmap& bitmap) {
  bitmap_ = bitmap;
  CHECK_EQ(width_, bitmap_.Width());
  CHECK_EQ(height_, bitmap_.Height());
}


void Image::SetK(const double K[9]) const {
	memcpy(K_, K, 9 * sizeof(double));
}

size_t Image::GetWidth() const { return width_; }

size_t Image::GetHeight() const { return height_; }

const Bitmap& Image::GetBitmap() const { return bitmap_; }

const std::string& Image::GetPath() const { return path_; }


void Image::SetLastRow(const double last_row[4]) {
  memcpy(last_row_, last_row, 4 * sizeof(double));
}

float Image::GetDepth(double x, double y, double z) const {
	  // 3 by 4 projection matrix
	  Eigen::Matrix<double, 3, 4, Eigen::RowMajor> P_3by4;
	  P_3by4.leftCols<3>() = Eigen::Map<const Eigen::Matrix<double, 3, 3, Eigen::RowMajor>>(R_);
	  P_3by4.rightCols<1>() = Eigen::Map<const Eigen::Matrix<double, 3, 1>>(T_);
	  P_3by4 = Eigen::Map<const Eigen::Matrix<double, 3, 3, Eigen::RowMajor>>(K_) * P_3by4;

	  // 4 by 4 projection matrix
	  Eigen::Matrix<double, 4, 4, Eigen::RowMajor> P_4by4;
	  P_4by4.block<3, 4>(0, 0) = P_3by4;
	  P_4by4.row(3) = Eigen::Map<const Eigen::Matrix<double, 1, 4>>(last_row_);

	  const Eigen::Matrix<double, 4, 1> X(x, y, z, 1.0);

	  Eigen::Matrix<double, 4, 1> result = P_4by4 * X;
	  // depth is the fourth component
	  double depth = result[3] / result[2];

	  // low precision output
	  return (float) depth;
}

const double* Image::GetLastRow() const {
	return last_row_;
}

void Image::GetK(float K[9]) const {
	DoubleArrToFloatArr(K_, K, 9);
}

void Image::GetRT(float R[9],  float T[3]) const {
	DoubleArrToFloatArr(R_, R, 9);
	DoubleArrToFloatArr(T_, T, 3);
}

void Image::GetC(float C[3]) const {
	double C_double[3];
	ComputeProjectionCenter(R_, T_, C_double);
	DoubleArrToFloatArr(C_double, C, 3);
}

void Image::GetCDouble(double C[3]) const {
	ComputeProjectionCenter(R_, T_, C);
}


void Image::GetKDouble(double K[9]) const {
	memcpy(K, K_, 9*sizeof(double));
}

void Image::GetPinvP(float P[16], float inv_P[16]) const {
	// compute projection matrix
	double P_double[16];
	double inv_P_double[16];
	Compute4by4ProjectionMatrix(K_, R_, T_, last_row_, P_double, inv_P_double);

	// convert to low-precision
	DoubleArrToFloatArr(P_double, P, 16);
	DoubleArrToFloatArr(inv_P_double, inv_P, 16);
}

void Image::GetPinvPDouble(double P[16], double inv_P[16]) const {
	// compute projection matrix
	Compute4by4ProjectionMatrix(K_, R_, T_, last_row_, P, inv_P);
}

// low-precision output
void Image::Original(float K[9], float inv_K[9], float R[9], float T[3], float P[16], float inv_P[16], float C[3]) const{
	// compute projection matrix
	double P_double[16];
	double inv_P_double[16];
	Compute4by4ProjectionMatrix(K_, R_, T_, last_row_, P_double, inv_P_double);

	double C_double[3];
	ComputeProjectionCenter(R_, T_, C_double);

	// compute inverse K
	const Eigen::Map<const Eigen::Matrix<double, 3, 3, Eigen::RowMajor>> K_mat(K_);
	Eigen::Matrix<double, 3, 3, Eigen::RowMajor> K_mat_inv = K_mat.inverse();
	const double *K_mat_inv_data = K_mat_inv.data();

	// convert to low-precision
	DoubleArrToFloatArr(K_, K, 9);
	DoubleArrToFloatArr(K_mat_inv_data, inv_K, 9);
	DoubleArrToFloatArr(R_, R, 9);
	DoubleArrToFloatArr(T_, T, 3);
	DoubleArrToFloatArr(P_double, P, 16);
	DoubleArrToFloatArr(inv_P_double, inv_P, 16);
	DoubleArrToFloatArr(C_double, C, 3);
}

inline void MatrixPrint(float *mat, int row_cnt, int col_cnt) {
	for (int i=0; i<row_cnt; ++i) {
		for (int j=0; j<col_cnt; ++j) {
			std::cout << mat[i*col_cnt+j] << ", ";
		}
	}
}

//void Image::Rotate90Multi_test(int cnt) const {
//	float K[9];
//	float R[9];
//	float T[3];
//	float P[16];
//	float inv_P[16];
//	float C[3];
//
//	Rotate90Multi(cnt, K, R, T, P, inv_P, C);
//
//	float K_float[9];
//	DoubleArrToFloatArr(K_, K_float, 9);
//	std::cout << "\nrot=0, K_: ";
//	MatrixPrint(K_float, 3, 3);
//	std::cout << "\nwidth, height: " << width_ << ", " << height_;
//	std::cout << "\nrot=" << cnt << ", K: ";
//	MatrixPrint(K, 3, 3);
//
//	float R_float[9];
//	DoubleArrToFloatArr(R_, R_float, 9);
//	std::cout << "\nrot=0, R_: ";
//	MatrixPrint(R_float, 3, 3);
//	std::cout << "\nrot=" << cnt << ", R: ";
//    MatrixPrint(R, 3, 3);
//
//	float T_float[3];
//	DoubleArrToFloatArr(T_, T_float, 3);
//	std::cout << "\nrot=0, T_: ";
//	MatrixPrint(T_float, 3, 1);
//	std::cout << "\nrot=" << cnt << ", T: ";
//	MatrixPrint(T, 3, 1);
//
//	float last_row_float[4];
//	DoubleArrToFloatArr(last_row_, last_row_float, 4);
//	std::cout << "\nlast_row_: ";
//	MatrixPrint(last_row_float, 1, 4);
//	std::cout << "\nrot=" << cnt << ", P: ";
//	MatrixPrint(P, 4, 4);
//	std::cout << "\nrot=" << cnt << ", inv_P: ";
//	MatrixPrint(inv_P, 4, 4);
//
//	std::cout << std::endl;
//}

void Image::Rotate90Multi(int cnt, float K[9], float inv_K[9], float R[9], float T[3], float P[16], float inv_P[16], float C[3]) const {
	switch (cnt % 4) {
	case 0:
		Original(K, inv_K, R, T, P, inv_P, C);
		break;
	case 1:
		Rotate90(K, inv_K, R, T, P, inv_P, C);
		break;
	case 2:
		Rotate180(K, inv_K, R, T, P, inv_P, C);
		break;
	case 3:
		Rotate270(K, inv_K, R, T, P, inv_P, C);
		break;
	default:
		break;
	}
}

// rotate the camera coordinate frame by 90 degrees
// a 3d point (X, Y, Z) in camera coordinate frame is rotated to (Y, -X, Z)
// the corresponing pixel (u, v) is rotated to (v, w-1-u)
// [u; v; 1] = K[X; Y; Z]
// needs to modify the intrinsics so that the new mapping holds, i.e.,
// [v; w-1-u; 1] = K'[Y; -X; Z]
// first, [X; Y; Z]= [0, -1, 0; 1, 0, 0; 0, 0, 1][Y; -X; Z]
// then, [v; w-1-u; 1] = [0, 1, 0; -1, 0, w-1; 0, 0, 1][u; v; 1]
void Image::Rotate90(float K[9], float inv_K[9], float R[9], float T[3], float P[16], float inv_P[16], float C[3]) const {
    // modify intrinsics
    const Eigen::Map<const Eigen::Matrix<double, 3, 3, Eigen::RowMajor>> K_old(K_);
    Eigen::Matrix<double, 3, 3, Eigen::RowMajor> A;
    A << 0, 1, 0, -1, 0, width_-1, 0, 0, 1;
    Eigen::Matrix<double, 3, 3, Eigen::RowMajor> B;
    B << 0, -1, 0, 1, 0, 0, 0, 0, 1;

    Eigen::Matrix<double, 3, 3, Eigen::RowMajor> K_new = A * K_old * B;
    Eigen::Matrix<double, 3, 3, Eigen::RowMajor> K_new_inv = K_new.inverse();

	// modify extrinsics
	Eigen::Matrix<double, 3, 3, Eigen::RowMajor> rot;
	rot << 0.0, 1.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 1.0;
	const Eigen::Map<const Eigen::Matrix<double, 3, 3, Eigen::RowMajor>> R_old(R_);
	const Eigen::Map<const Eigen::Matrix<double, 3, 1>> T_old(T_);

	const Eigen::Matrix<double, 3, 3, Eigen::RowMajor> R_new = rot * R_old;
	const Eigen::Matrix<double, 3, 1> T_new = rot * T_old;

	const double *K_new_data = K_new.data();
	const double *K_new_inv_data = K_new_inv.data();
	const double *R_new_data = R_new.data();
	const double *T_new_data = T_new.data();

	// compute projection matrix
	double P_new_double[16];
	double inv_P_new_double[16];
	Compute4by4ProjectionMatrix(K_new_data, R_new_data, T_new_data, last_row_, P_new_double, inv_P_new_double);

	// compute projection center
	double C_new_double[3];
	ComputeProjectionCenter(R_new_data, T_new_data, C_new_double);

	// convert to low-precision
	DoubleArrToFloatArr(K_new_data, K, 9);
	DoubleArrToFloatArr(K_new_inv_data, inv_K, 9);
	DoubleArrToFloatArr(R_new_data, R, 9);
	DoubleArrToFloatArr(T_new_data, T, 3);
	DoubleArrToFloatArr(P_new_double, P, 16);
	DoubleArrToFloatArr(inv_P_new_double, inv_P, 16);
	DoubleArrToFloatArr(C_new_double, C, 3);
}

// a 3d point (X, Y, Z) in camera coordinate frame is rotated to (-X, -Y, Z)
// the corresponing pixel (u, v) is rotated to (w-1-u, h-1-v)
// [u; v; 1] = K[X; Y; Z]
// needs to modify the intrinsics so that the new mapping holds, i.e.,
// [w-1-u; h-1-v ; 1] = K'[-X; -Y; Z]
// first, [X; Y; Z]= [-1, 0, 0; 0, -1, 0; 0, 0, 1][-X; -Y; Z]
// then, [w-1-u; h-1-v; 1] = [-1, 0, w-1; 0, -1, h-1; 0, 0, 1][u; v; 1]
void Image::Rotate180(float K[9], float inv_K[9], float R[9], float T[3], float P[16], float inv_P[16], float C[3]) const {
    // modify intrinsics
    const Eigen::Map<const Eigen::Matrix<double, 3, 3, Eigen::RowMajor>> K_old(K_);
    Eigen::Matrix<double, 3, 3, Eigen::RowMajor> A;
    A << -1, 0, width_-1, 0, -1, height_-1, 0, 0, 1;
    Eigen::Matrix<double, 3, 3, Eigen::RowMajor> B;
    B << -1, 0, 0, 0, -1, 0, 0, 0, 1;

    Eigen::Matrix<double, 3, 3, Eigen::RowMajor> K_new = A * K_old * B;
    Eigen::Matrix<double, 3, 3, Eigen::RowMajor> K_new_inv = K_new.inverse();

	// modify extrinsics
	Eigen::Matrix<double, 3, 3, Eigen::RowMajor> rot;
	rot << 0.0, 1.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 1.0;
	rot = rot * rot;
	const Eigen::Map<const Eigen::Matrix<double, 3, 3, Eigen::RowMajor>> R_old(R_);
	const Eigen::Map<const Eigen::Matrix<double, 3, 1>> T_old(T_);

	const Eigen::Matrix<double, 3, 3, Eigen::RowMajor> R_new = rot * R_old;
	const Eigen::Matrix<double, 3, 1> T_new = rot * T_old;

	const double *K_new_data = K_new.data();
	const double *K_new_inv_data = K_new_inv.data();
	const double *R_new_data = R_new.data();
	const double *T_new_data = T_new.data();

	// compute projection matrix
	double P_new_double[16];
	double inv_P_new_double[16];
	Compute4by4ProjectionMatrix(K_new_data, R_new_data, T_new_data, last_row_, P_new_double, inv_P_new_double);

	// compute projection center
	double C_new_double[3];
	ComputeProjectionCenter(R_new_data, T_new_data, C_new_double);

	// convert to low-precision
	DoubleArrToFloatArr(K_new_data, K, 9);
	DoubleArrToFloatArr(K_new_inv_data, inv_K, 9);
	DoubleArrToFloatArr(R_new_data, R, 9);
	DoubleArrToFloatArr(T_new_data, T, 3);
	DoubleArrToFloatArr(P_new_double, P, 16);
	DoubleArrToFloatArr(inv_P_new_double, inv_P, 16);
	DoubleArrToFloatArr(C_new_double, C, 3);
}

// a 3d point (X, Y, Z) in camera coordinate frame is rotated to (-Y, X, Z)
// the corresponing pixel (u, v) is rotated to (h-1-v, u)
// needs to modify the intrinsics so that the new mapping holds, i.e.,
// [h-1-v; u ; 1] = K'[-Y; X; Z]
// first, [X; Y; Z]= [0, 1, 0; -1, 0, 0; 0, 0, 1][-Y; X; Z]
// then, [h-1-v; u; 1] = [0, -1, h-1; 1, 0, 0; 0, 0, 1][u; v; 1]
void Image::Rotate270(float K[9], float inv_K[9], float R[9], float T[3], float P[16], float inv_P[16], float C[3]) const {
    // modify intrinsics
    const Eigen::Map<const Eigen::Matrix<double, 3, 3, Eigen::RowMajor>> K_old(K_);
    Eigen::Matrix<double, 3, 3, Eigen::RowMajor> A;
    A << 0, -1, height_-1, 1, 0, 0, 0, 0, 1;
    Eigen::Matrix<double, 3, 3, Eigen::RowMajor> B;
    B << 0, 1, 0, -1, 0, 0, 0, 0, 1;

    Eigen::Matrix<double, 3, 3, Eigen::RowMajor> K_new = A * K_old * B;
    Eigen::Matrix<double, 3, 3, Eigen::RowMajor> K_new_inv = K_new.inverse();

    // modify extrinsics
	Eigen::Matrix<double, 3, 3, Eigen::RowMajor> rot;
	rot << 0.0, 1.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 1.0;
	rot = rot * rot * rot;
	const Eigen::Map<const Eigen::Matrix<double, 3, 3, Eigen::RowMajor>> R_old(R_);
	const Eigen::Map<const Eigen::Matrix<double, 3, 1>> T_old(T_);

	const Eigen::Matrix<double, 3, 3, Eigen::RowMajor> R_new = rot * R_old;
	const Eigen::Matrix<double, 3, 1> T_new = rot * T_old;

	const double *K_new_data = K_new.data();
	const double *K_new_inv_data = K_new_inv.data();
	const double *R_new_data = R_new.data();
	const double *T_new_data = T_new.data();

	// compute projection matrix
	double P_new_double[16];
	double inv_P_new_double[16];
	Compute4by4ProjectionMatrix(K_new_data, R_new_data, T_new_data, last_row_, P_new_double, inv_P_new_double);

	// compute projection center
	double C_new_double[3];
	ComputeProjectionCenter(R_new_data, T_new_data, C_new_double);

	// convert to low-precision
	DoubleArrToFloatArr(K_new_data, K, 9);
	DoubleArrToFloatArr(K_new_inv_data, inv_K, 9);
	DoubleArrToFloatArr(R_new_data, R, 9);
	DoubleArrToFloatArr(T_new_data, T, 3);
	DoubleArrToFloatArr(P_new_double, P, 16);
	DoubleArrToFloatArr(inv_P_new_double, inv_P, 16);
	DoubleArrToFloatArr(C_new_double, C, 3);
}


void Image::Rescale(const float factor) { Rescale(factor, factor); }

void Image::Rescale(const float factor_x, const float factor_y) {
  const size_t new_width = std::round(width_ * factor_x);
  const size_t new_height = std::round(height_ * factor_y);

  if (bitmap_.Data() != nullptr) {
    bitmap_.Rescale(new_width, new_height);
  }

  const double scale_x = new_width / static_cast<float>(width_);
  const double scale_y = new_height / static_cast<float>(height_);
  K_[0] *= scale_x;
  K_[1] *= scale_x;
  K_[2] *= scale_x;
  K_[4] *= scale_y;
  K_[5] *= scale_y;

  width_ = new_width;
  height_ = new_height;
}


void Image::Downsize(const size_t max_width, const size_t max_height) {
  if (width_ <= max_width && height_ <= max_height) {
    return;
  }
  const float factor_x = static_cast<float>(max_width) / width_;
  const float factor_y = static_cast<float>(max_height) / height_;
  Rescale(std::min(factor_x, factor_y));
}


}  // namespace mvs
}  // namespace colmap
