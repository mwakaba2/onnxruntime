// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/common/status.h"
#include "core/framework/tensor.h"
#include "core/framework/data_types_internal.h"
#include "core/framework/data_transfer_manager.h"
#include "core/framework/sparse_tensor.h"
#include "core/framework/sparse_cooformat_rep.h"
#include "core/framework/sparse_csrcformat_rep.h"

#include "math_cpuonly.h"

#include <Eigen/SparseCore>

namespace onnxruntime {
namespace sparse_utils {

// Determines if this is a type specific zero
using IsZeroFunc = bool (*)(const void*);
// Copy element
using CopyElementFunc = void (*)(void* dest, const void* src, int64_t dest_index, int64_t src_index);

template <typename T>
inline bool IsZero(const void* p) {
  return (static_cast<T>(0) == *reinterpret_cast<const T*>(p));
}

template <typename T>
inline void CopyElement(void* dst, const void* src, int64_t dst_index, int64_t src_index) {
  reinterpret_cast<T*>(dst)[dst_index] = reinterpret_cast<const T*>(src)[src_index];
}

// StorageIndex must be int64_t, since our Csr/Csc formats currently require int64_t type
// for indexing, but this may change
template <class T>
using SparseMatrixRowMajor = Eigen::SparseMatrix<T, Eigen::RowMajor, int64_t>;

template <typename In>
struct TypeMap {
  using Out = In;
};

template <>
struct TypeMap<MLFloat16> {
  using Out = Eigen::half;
};

template <typename T>
struct ToCsrSparseConvert {
  Status operator()(const DataTransferManager& data_manager, const Tensor& src_cpu,
                    const AllocatorPtr& allocator, SparseTensor& dst) const {
    const auto* input_data = src_cpu.Data<T>();
    const auto& dense_shape = src_cpu.Shape();
    // We do not support a stack of matrices here
    ORT_RETURN_IF_NOT(dense_shape.NumDimensions() == 2, "Currently support two dim tensors");
    const auto M = dense_shape.GetDims()[0];
    const auto N = dense_shape.GetDims()[1];

    ConstEigenMatrixMapRowMajor<TypeMap<T>::Out> dense_map(reinterpret_cast<const TypeMap<T>::Out*>(input_data), M, N);
    // Quick way to convert.
    SparseMatrixRowMajor<TypeMap<T>::Out> sparse_matrix = dense_map.sparseView();
    sparse_matrix.makeCompressed();
    static_assert(sizeof(T) == sizeof(typename SparseMatrixRowMajor<T>::Scalar), "Expecting data type parity");
    static_assert(sizeof(int64_t) == sizeof(typename SparseMatrixRowMajor<T>::StorageIndex), "Expecting index type parity");
    static_assert(std::is_signed<int64_t>::value == std::is_signed<typename SparseMatrixRowMajor<T>::StorageIndex>::value,
                  "Indices must be both (un)signed");

    const auto nnz = sparse_matrix.nonZeros();

    TensorShape values_shape{nnz};
    TensorShape inner_shape{nnz};
    TensorShape outer_shape{M + 1};
    const OrtMemoryInfo& cpu_info = src_cpu.Location();
    Tensor values(src_cpu.DataType(), values_shape, sparse_matrix.valuePtr(), cpu_info);
    Tensor inner_indices(DataTypeImpl::GetType<int64_t>(), inner_shape, sparse_matrix.innerIndexPtr(), cpu_info);
    Tensor outer_indices(DataTypeImpl::GetType<int64_t>(), outer_shape, sparse_matrix.outerIndexPtr(), cpu_info);

    SparseTensor sparse_tensor(src_cpu.DataType(), dense_shape, nnz, allocator);
    SparseCsrcFormatRep* rep = nullptr;
    auto builder = sparse_tensor.RepBuilder<SparseCsrcBuilder>();
    ORT_RETURN_IF_ERROR(builder.GetOrCreate(SparseCsrcFormatRep::kRowMajor, inner_shape, outer_shape, rep));
    if (nnz > 0) {
      ORT_RETURN_IF_ERROR(data_manager.CopyTensor(values, sparse_tensor.MutableValues()));
      ORT_RETURN_IF_ERROR(data_manager.CopyTensor(inner_indices, rep->MutableInner()));
    }
    ORT_RETURN_IF_ERROR(data_manager.CopyTensor(outer_indices, rep->MutableOuter()));
    dst = std::move(sparse_tensor);
    return Status::OK();
  }
};

common::Status DenseTensorToSparseCsr(const DataTransferManager& data_manager, const Tensor& src,
                                      const AllocatorPtr& cpu_allocator, const AllocatorPtr& allocator,
                                      SparseTensor& dst) {
  const auto num_dims = src.Shape().NumDimensions();
  if (num_dims > 2) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Currently do not support dims higher than 2 dimensions");
  }

  // Eigen currently does not have BFloat16 support but it may be coming.
  utils::MLTypeCallDispatcher<int8_t, uint8_t, int16_t, uint16_t, int32_t, uint32_t,
                              int64_t, uint64_t, double, float, MLFloat16>
      t_disp(src.GetElementType());

  Status status;
  if (src.Location().device != cpu_allocator->Info().device) {
    Tensor src_cpu(src.DataType(), src.Shape(), cpu_allocator);
    ORT_RETURN_IF_ERROR(data_manager.CopyTensor(src, src_cpu));
    status = t_disp.InvokeRet<common::Status, ToCsrSparseConvert>(data_manager, src_cpu, allocator, dst);
  } else {
    status = t_disp.InvokeRet<common::Status, ToCsrSparseConvert>(data_manager, src, allocator, dst);
  }

  return status;
}

template <typename T>
struct ConvertCsrToDense {
  Status operator()(const DataTransferManager& data_manager, const SparseTensor& cpu_tensor,
                    const AllocatorPtr& cpu_allocator, const AllocatorPtr& dst_allocator, Tensor& dst) {
    const auto& dense_shape = cpu_tensor.Shape();
    const auto M = dense_shape.GetDims()[0];
    const auto N = dense_shape.GetDims()[1];
    const auto nnz = cpu_tensor.NumValues();

    const SparseCsrcFormatRep* rep = cpu_tensor.GetRep<SparseCsrcFormatRep>();
    ConstSparseMatrixMap<TypeMap<T>::Out> sparse_map(M, N, nnz,
                                                     rep->Outer().Data<int64_t>(),
                                                     rep->Inner().Data<int64_t>(),
                                                     reinterpret_cast<const TypeMap<T>::Out*>(cpu_tensor.Values().Data<T>()));

    // Convert to a dense tensor
    const AllocatorPtr& conversion_allocator = (cpu_tensor.Location().device == dst_allocator->Info().device) ? dst_allocator : cpu_allocator;
    Tensor cpu_result(cpu_tensor.DataType(), dense_shape, conversion_allocator);
    EigenMatrixMapRowMajor<TypeMap<T>::Out> result_map(reinterpret_cast<TypeMap<T>::Out*>(cpu_result.MutableData<T>()),
                                                       M, N);
    result_map = sparse_map;

    if (cpu_tensor.Location().device == dst_allocator->Info().device) {
      dst = std::move(cpu_result);
    } else {
      Tensor dst_result(cpu_tensor.DataType(), dense_shape, dst_allocator);
      ORT_RETURN_IF_ERROR(data_manager.CopyTensor(cpu_result, dst_result));
      dst = std::move(dst_result);
    }

    return Status::OK();
  }
};

common::Status SparseCsrToDenseTensor(const DataTransferManager& data_manager, const SparseTensor& src,
                                      const AllocatorPtr& cpu_allocator, const AllocatorPtr& dst_allocator,
                                      Tensor& dst) {
  if (!IsSet(src.FormatFlags(), SparseFormatFlags::kCsrc) ||
      src.GetRep<SparseCsrcFormatRep>()->Major() != SparseCsrcFormatRep::kRowMajor) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Input must be of CRS format");
  }

  if (src.Shape().NumDimensions() != 2) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Support 2-D matrices only");
  }

  const SparseCsrcFormatRep* rep = src.GetRep<SparseCsrcFormatRep>();
  const auto inner_num = rep->Inner().Shape().Size();
  const auto outer_num = rep->Outer().Shape().Size();
  ORT_ENFORCE(inner_num == src.NumValues(), "Expecting inner indecies to be same as nnz. Got: ", inner_num);
  ORT_ENFORCE(outer_num == (src.Shape().GetDims()[0] + 1), "Outer indecies must be M + 1. Got: ", outer_num);

  utils::MLTypeCallDispatcher<int8_t, uint8_t, int16_t, uint16_t, int32_t, uint32_t,
                              int64_t, uint64_t, double, float, MLFloat16>
      t_disp(src.GetElementType());

  Status status;
  if (src.Location().device != cpu_allocator->Info().device) {
    SparseTensor src_cpu(src.DataType(), src.Shape(), src.NumValues(), cpu_allocator);
    ORT_RETURN_IF_ERROR(src.Copy(data_manager, 0, src_cpu));
    status = t_disp.InvokeRet<Status, ConvertCsrToDense>(data_manager, src_cpu, cpu_allocator, dst_allocator, dst);
  } else {
    status = t_disp.InvokeRet<Status, ConvertCsrToDense>(data_manager, src, cpu_allocator, dst_allocator, dst);
  }

  return status;
}

Status DenseTensorToSparseCoo(const DataTransferManager& data_manager, const Tensor& src,
                              const AllocatorPtr& cpu_allocator,
                              const AllocatorPtr& dst_allocator, bool linear_index, SparseTensor& dst) {
  const auto& src_dims = src.Shape().GetDims();
  if (src_dims.size() != 2) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "Currently do not support dims higher than 2 dimensions: ", src_dims.size());
  }

  const auto dense_elements = src.Shape().Size();
  const auto element_size = src.DataType()->AsPrimitiveDataType()->Size();
  gsl::span<const gsl::byte> src_span;
  Tensor src_cpu;
  if (src.Location().device != cpu_allocator->Info().device) {
    Tensor t(src.DataType(), src.Shape(), cpu_allocator);
    ORT_RETURN_IF_ERROR(data_manager.CopyTensor(src, t));
    src_cpu = std::move(t);
    src_span = gsl::make_span(reinterpret_cast<const gsl::byte*>(src_cpu.DataRaw()), src_cpu.SizeInBytes());
  } else {
    src_span = gsl::make_span(reinterpret_cast<const gsl::byte*>(src.DataRaw()), src.SizeInBytes());
  }

  IsZeroFunc is_zero = nullptr;
  CopyElementFunc copy_func = nullptr;
  size_t nnz = 0;
  switch (element_size) {
    case sizeof(uint8_t):
      nnz = std::count_if(src_span.cbegin(), src_span.cend(), [](auto v) { return v != gsl::byte{0}; });
      is_zero = IsZero<uint8_t>;
      copy_func = CopyElement<uint8_t>;
      break;
    case sizeof(uint16_t): {
      // MFFloat16 and BFloat16 are handled fine
      auto span16 = src_span.as_span<const uint16_t>();
      nnz = std::count_if(span16.cbegin(), span16.cend(), [](auto v) { return v != 0; });
      is_zero = IsZero<uint16_t>;
      copy_func = CopyElement<uint16_t>;
    } break;
    case sizeof(uint32_t): {
      auto span32 = src_span.as_span<const uint32_t>();
      nnz = std::count_if(span32.cbegin(), span32.cend(), [](auto v) { return v != 0; });
      is_zero = IsZero<uint32_t>;
      copy_func = CopyElement<uint32_t>;
    } break;
    case sizeof(uint64_t): {
      auto span64 = src_span.as_span<const uint64_t>();
      nnz = std::count_if(span64.cbegin(), span64.cend(), [](auto v) { return v != 0; });
      is_zero = IsZero<uint64_t>;
      copy_func = CopyElement<uint64_t>;
    } break;
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Unsupported element size: ", element_size);
  }

  const AllocatorPtr& conversion_allocator = (cpu_allocator->Info().device == dst_allocator->Info().device) ? dst_allocator : cpu_allocator;
  SparseTensor cpu_result(src.DataType(), src.Shape(), nnz, conversion_allocator);
  SparseCooFormatRep* rep;
  ORT_RETURN_IF_ERROR(cpu_result.RepBuilder<SparseCooBuilder>().GetOrCreate(linear_index, rep));
  if (nnz > 0) {
    auto advance = [element_size](const gsl::byte* start, size_t elements) {
      return (start + elements * element_size);
    };

    ORT_RETURN_IF_NOT(cpu_result.NumValues() == nnz, "Wrong values size");

    const auto* input_data = src_span.data();
    void* cpu_output = cpu_result.MutableValues().MutableDataRaw();
    int64_t output_index = 0;
    int64_t* indices_out = rep->MutableIndices().MutableData<int64_t>();
    const auto cols = src_dims[1];
    int64_t row = 0;
    int64_t col = 0;
    for (int64_t i = 0; i < static_cast<int64_t>(dense_elements); ++i, ++col) {
      if (col >= cols) {
        col = 0;
        ++row;
      }
      if (!is_zero(input_data)) {
        if (linear_index) {
          *indices_out++ = i;
        } else {
          *indices_out++ = row;
          *indices_out++ = col;
        }
        copy_func(cpu_output, input_data, output_index++, 0);
      }
      input_data = advance(input_data, 1U);
    }
  }

  // Check if we need to copy
  if (conversion_allocator->Info().device == dst_allocator->Info().device) {
    dst = std::move(cpu_result);
  } else {
    SparseTensor t(src.DataType(), src.Shape(), nnz, dst_allocator);
    ORT_RETURN_IF_ERROR(data_manager.CopyTensor(cpu_result, t));
    dst = std::move(t);
  }

  return Status::OK();
}

Status SparseCooToDenseTensor(const DataTransferManager& data_manager, const SparseTensor& src,
                              const AllocatorPtr& cpu_allocator, const AllocatorPtr& dst_allocator, Tensor& dst) {
  const auto& src_dims = src.Shape().GetDims();
  if (src_dims.size() != 2) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "Currently do not support dims higher than 2 dimensions: ", src_dims.size());
  }

  const AllocatorPtr& conversion_allocator = (cpu_allocator->Info().device == dst_allocator->Info().device) ? dst_allocator : cpu_allocator;
  Tensor cpu_result(src.DataType(), src.Shape(), conversion_allocator);
  memset(cpu_result.MutableDataRaw(), 0, cpu_result.SizeInBytes());

  if (src.NumValues() > 0) {
    const void* values = nullptr;
    const int64_t* indices = nullptr;
    const auto num_values = src.Values().Shape().Size();
    const auto num_indices = src.GetRep<SparseCooFormatRep>()->Indices().Shape().Size();
    ORT_RETURN_IF_NOT((num_values == num_indices || 2 * num_values == num_indices), "Invalid indices number");
    SparseTensor src_cpu;
    if (src.Location().device != cpu_allocator->Info().device) {
      SparseTensor t(src.DataType(), src.Shape(), src.NumValues(), cpu_allocator);
      ORT_RETURN_IF_ERROR(data_manager.CopyTensor(src, t));
      src_cpu = std::move(t);
      values = src_cpu.Values().DataRaw();
      const auto* rep = src_cpu.GetRep<SparseCooFormatRep>();
      indices = rep->Indices().Data<int64_t>();
    } else {
      values = src.Values().DataRaw();
      const auto* rep = src.GetRep<SparseCooFormatRep>();
      indices = rep->Indices().Data<int64_t>();
    }
    const auto element_size = src.DataType()->AsPrimitiveDataType()->Size();
    void* output = cpu_result.MutableDataRaw();
    auto advance = [element_size](const gsl::byte* start, size_t elements) {
      return (start + elements * element_size);
    };

    CopyElementFunc copy_func = nullptr;
    switch (element_size) {
      case sizeof(uint8_t):
        copy_func = CopyElement<uint8_t>;
        break;
      case sizeof(uint16_t): {
        copy_func = CopyElement<uint16_t>;
      } break;
      case sizeof(uint32_t): {
        copy_func = CopyElement<uint32_t>;
      } break;
      case sizeof(uint64_t): {
        copy_func = CopyElement<uint64_t>;
      } break;
        assert(false);
        return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Unsupported element size: ", element_size);
    }

    const auto dense_size = src.Shape().Size();
    // Linear index
    if (num_indices == num_values) {
      for (int64_t src_idx = 0; src_idx < num_values; ++src_idx) {
        auto dst_idx = indices[src_idx];
        ORT_RETURN_IF_NOT(dst_idx < dense_size, "Invalid index: ", dst_idx, " > dense_size: ", dense_size);
        copy_func(output, values, dst_idx, src_idx);
      }
    } else {
      const auto cols = src_dims[1];
      for (int64_t src_idx = 0; src_idx < num_values; ++src_idx) {
        auto tuple_idx = src_idx * 2;
        auto dst_idx = indices[tuple_idx] * cols + indices[tuple_idx + 1];
        ORT_RETURN_IF_NOT(dst_idx < dense_size, "Invalid index: ", dst_idx, " > dense_size: ", dense_size);
        copy_func(output, values, dst_idx, src_idx);
      }
    }
  }

  if (conversion_allocator->Info().device == dst_allocator->Info().device) {
    dst = std::move(cpu_result);
  } else {
    Tensor t(src.DataType(), src.Shape(), dst_allocator);
    ORT_RETURN_IF_ERROR(data_manager.CopyTensor(cpu_result, t));
    dst = std::move(t);
  }

  return Status::OK();
}

}  // namespace sparse_utils
}  // namespace onnxruntime