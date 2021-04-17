// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/framework/sparse_tensor.h"

namespace onnxruntime {
/// <summary>
/// This is a representation of Coo format that is generic.
/// </summary>
class SparseCooFormatRep : public SparseRep {
 public:
  SparseCooFormatRep(const TensorShape& ind_shape, const AllocatorPtr& allocator)
      : indices_(DataTypeImpl::GetType<int64_t>(),
                 ind_shape,
                 allocator) {
  }

  SparseCooFormatRep(const TensorShape& ind_shape, const OrtMemoryInfo& info, void* indices_data)
      : indices_(DataTypeImpl::GetType<int64_t>(),
                 ind_shape,
                 indices_data,
                 info,
                 0) {}

  ~SparseCooFormatRep() override;

  const Tensor& Indices() const noexcept {
    return indices_;
  }

  Tensor& MutableIndices() noexcept {
    return indices_;
  }

  Status Copy(const DataTransferManager& data_transfer_manager, const AllocatorPtr& allocator,
              int exec_q_id, std::unique_ptr<SparseRep>& dst_rep) const override;

  Status Copy(const IDataTransfer& data_transfer, const AllocatorPtr& allocator,
              int exec_q_id, std::unique_ptr<SparseRep>& dst_rep) const override;

 private:
  Tensor indices_;  // may be 1-D or 2-D.
};

/// <summary>
/// Class returns rep specific interface to properly construct a given
/// sparse format
/// </summary>
class SparseCooBuilder {
  AllocatorPtr allocator_;
  SparseTensor* sp_;
  std::unique_ptr<SparseRep>* rep_;

 public:
  SparseCooBuilder(AllocatorPtr allocator, SparseTensor& sp, std::unique_ptr<SparseRep>& rep) noexcept
      : allocator_(std::move(allocator)),
        sp_(&sp),
        rep_(&rep) {}

  /// <summary>
  /// Creates an owned representation using SparseTensor allocator
  /// and dense shape dimensions.
  /// </summary>
  /// <param name="linearized">true if indices have 1-D linearized format and have a size of NumValues() or
  /// 2-D which is a coordinate format. The latter would have a length of 2 * NumValues()</param>
  /// <returns>Existing or created representation</returns>
  Status GetOrCreate(bool linearized, SparseCooFormatRep*&);

  /// <summary>
  /// Create a COO representation that would not own the data. Use for inputs/outputs
  /// The builder is going to use the same OrtMemoryInfo as for values
  /// </summary>
  /// <param name="indices_shape">1-D or 2-D indices shape</param>
  /// <param name="indices_data">pointer to indices data</param>
  /// <param name="result">resulting representation</param>
  /// <returns>Status</returns>
  Status GetOrCreate(const TensorShape& indices_shape, void* indices_data, SparseCooFormatRep*& result);
};

template <>
inline const SparseCooFormatRep* SparseTensor::GetRep<SparseCooFormatRep>() const {
  ORT_ENFORCE(IsSet(format_flags_, SparseFormatFlags::kCoo), "Expecting COO format");
  return static_cast<const SparseCooFormatRep*>(rep_.get());
}

template <>
inline SparseCooBuilder SparseTensor::RepBuilder<SparseCooBuilder>() {
  if (!rep_) {
    format_flags_ = Set(format_flags_, SparseFormatFlags::kCoo);
  } else {
    ORT_ENFORCE(IsSet(format_flags_, SparseFormatFlags::kCoo), "Expecting COO format set");
  }
  return SparseCooBuilder(allocator_, *this, rep_);
}

}  // namespace onnxruntime
