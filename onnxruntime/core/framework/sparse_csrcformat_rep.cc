// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/framework/sparse_csrcformat_rep.h"
#include "core/framework/data_transfer_manager.h"

namespace onnxruntime {

SparseCsrcFormatRep::~SparseCsrcFormatRep() = default;

Status SparseCsrcFormatRep::Copy(const DataTransferManager& data_transfer_manager,
                                 const AllocatorPtr& allocator,
                                 int exec_q_id, std::unique_ptr<SparseRep>& dst_rep) const {
  auto rep_copy = std::make_unique<SparseCsrcFormatRep>(Major(), inner_indecies_.Shape(), outer_indecies_.Shape(), allocator);
  ORT_RETURN_IF_ERROR(data_transfer_manager.CopyTensor(inner_indecies_, rep_copy->MutableInner(), exec_q_id));
  ORT_RETURN_IF_ERROR(data_transfer_manager.CopyTensor(outer_indecies_, rep_copy->MutableOuter(), exec_q_id));
  dst_rep = std::move(rep_copy);
  return Status::OK();
}

Status SparseCsrcFormatRep::Copy(const IDataTransfer& data_transfer, const AllocatorPtr& allocator,
                                 int exec_q_id, std::unique_ptr<SparseRep>& dst_rep) const {
  auto rep_copy = std::make_unique<SparseCsrcFormatRep>(Major(), inner_indecies_.Shape(), outer_indecies_.Shape(), allocator);
  ORT_RETURN_IF_ERROR(data_transfer.CopyTensor(inner_indecies_, rep_copy->MutableInner(), exec_q_id));
  ORT_RETURN_IF_ERROR(data_transfer.CopyTensor(outer_indecies_, rep_copy->MutableOuter(), exec_q_id));
  dst_rep = std::move(rep_copy);
  return Status::OK();
}

Status SparseCsrcBuilder::GetOrCreate(SparseCsrcFormatRep::Order major,
                                      const TensorShape& inner, const TensorShape& outer,
                                      SparseCsrcFormatRep*& result) {
  ORT_RETURN_IF_NOT(allocator_ != nullptr, "Must have an allocator set with Sparse Tensor instance");
  if (rep_->get()) {
    result = static_cast<SparseCsrcFormatRep*>(rep_->get());
    return Status::OK();
  }

  result = new SparseCsrcFormatRep(major, inner, outer, allocator_);
  rep_->reset(result);
  return Status::OK();
}

Status SparseCsrcBuilder::GetOrCreate(SparseCsrcFormatRep::Order major, const TensorShape& inner, const TensorShape& outer,
                                      int64_t* inner_data, int64_t* outer_data, SparseCsrcFormatRep*& result) {
  ORT_RETURN_IF_NOT(allocator_ == nullptr, "Must have NOT an allocator set with Sparse Tensor instance");
  if (rep_->get()) {
    result = static_cast<SparseCsrcFormatRep*>(rep_->get());
    return Status::OK();
  }

  result = new SparseCsrcFormatRep(major, inner, outer, inner_data, outer_data, sp_->Location());
  rep_->reset(result);
  return Status::OK();
}

}  // namespace onnxruntime