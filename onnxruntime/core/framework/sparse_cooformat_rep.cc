// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/framework/sparse_cooformat_rep.h"
#include "core/framework/data_transfer_manager.h"

namespace onnxruntime {
SparseCooFormatRep ::~SparseCooFormatRep() = default;

Status SparseCooFormatRep::Copy(const DataTransferManager& data_transfer_manager, const AllocatorPtr& allocator,
                               int exec_q_id, std::unique_ptr<SparseRep>& dst_rep) const {
  auto rep_copy = std::make_unique<SparseCooFormatRep>(indices_.Shape(), allocator);
  ORT_RETURN_IF_ERROR(data_transfer_manager.CopyTensor(indices_, rep_copy->indices_, exec_q_id));
  dst_rep = std::move(rep_copy);
  return Status::OK();
}

Status SparseCooFormatRep::Copy(const IDataTransfer& data_transfer, const AllocatorPtr& allocator,
                               int exec_q_id, std::unique_ptr<SparseRep>& dst_rep) const {
  auto rep_copy = std::make_unique<SparseCooFormatRep>(indices_.Shape(), allocator);
  ORT_RETURN_IF_ERROR(data_transfer.CopyTensor(indices_, rep_copy->indices_, exec_q_id));
  dst_rep = std::move(rep_copy);
  return Status::OK();
}

Status SparseCooBuilder::GetOrCreate(bool linearized, SparseCooFormatRep*& result) {
  ORT_RETURN_IF_NOT(allocator_ != nullptr, "Must have an allocator set with Sparse Tensor instance");

  if (rep_->get()) {
    result =  static_cast<SparseCooFormatRep*>(rep_->get());
    return Status::OK();
  }

  const auto nnz = sp_->Values().Shape().Size();
  if (linearized) {
    result = new SparseCooFormatRep({nnz}, allocator_);
    rep_->reset(result);
  } else {
    result = new SparseCooFormatRep({nnz, 2}, allocator_);
    rep_->reset(result);
  }
  return Status::OK();
}

Status SparseCooBuilder::GetOrCreate(const TensorShape& indices_shape, void* indices_data, SparseCooFormatRep*& result) {
  ORT_ENFORCE(allocator_ == nullptr, "Should not have an allocator set");

  if (rep_->get()) {
    result = static_cast<SparseCooFormatRep*>(rep_->get());
    return Status::OK();
  }

  const auto num_dim = indices_shape.NumDimensions();
  ORT_RETURN_IF_NOT(num_dim == 1 || num_dim == 2, "Require indices shape to be 1-D or 2-D");
  if (num_dim == 1) {
    ORT_RETURN_IF_NOT(indices_shape.Size() == sp_->NumValues(), "Sparse COO 1-D indices must have the size of NNZ");
  } else {
    ORT_RETURN_IF_NOT(indices_shape.Size() == sp_->NumValues() * 2, "Sparse COO 2-D indices must have the size of 2 * NNZ");
  }

  result = new SparseCooFormatRep(indices_shape, sp_->Location(), indices_data);
  rep_->reset(result);
  return Status::OK();
}

}  // namespace onnxruntime