//onesided parameter have to be added

#pragma once

#include "rpu_forward_backward_pass.h"
#include "rpu_pulsed_device.h"
#include "rpu_simple_device.h"
#include "rpu_vector_device.h"
#include "rpu_weight_updater.h"
#include <sstream>
#include <stdio.h>

namespace RPU {

template <typename T> class OneSidedTransferRPUDevice;

template <typename T> struct OneSidedTransferRPUDeviceParameter : VectorRPUDeviceMetaParameter<T> {

T gamma= (T)0.0;

T transfer_every = (T)1.0;
bool units_in_mbatch = false;
int n_reads_per_transfer = 1;
T with_reset_prob = (T)0.0;
bool no_self_transfer = true;
bool random_selection = false;
T fast_lr = 0.0;
T transfer_lr = (T)1.0;
std::vector<T> transfer_lr_vec;
bool scale_transfer_lr = true;
bool transfer_columns = true; // or rows
int _in_size = 0;
int _out_size = 0;

std::vector<T> transfer_every_vec; // TODO

IOMetaParamter<T> transfer_io;
PulsedUpdateMetaParameter<T> transfer_up;

OneSidedTransferRPUDeviceMetaParameter(){};
OneSidedTransferRPUDeviceMetaParameter(const OneSidedRPUDeviceMetaParameterBase<T> &dp, int n_devices)
    : VectorRPUDeviceMetaParameter<T>(dp, n_devices){};

OneSidedTransferRPUDeviceMetaParameter(
    const OneSidedRPUDeviceMetaParameterBase<T> &dp_fast,
    const OneSidedRPUDeviceMetaParameterBase<T> &dp_rest,
    int n_total_devices);

virtual void initializeWithSize(int x_size, int d_size);
void initialize() override{/* do nothing */};
inline bool fullyHidden() const { return (!gamma && this->gamma_vec.back() == 1.0); };

inline int getInSize() const { return _in_size; };
inline int getOutSize() const { return _out_size; };


std::string getName() const override {
    std::ostringstream ss;
    ss << "OneSidedTransfer(" << this->vec_par.size() << ")";
    if (this->vec_par.size() > 1) {
      ss << ": " << this->vec_par[0]->getName() << " -> " << this->vec_par[1]->getName();
      ;
    }
    return ss.str();
  };

OneSidedTransferRPUDevice<T> *createDevice(int x_size, int d_size, RealWorldRNG<T> *rng) override {
    return new OneSidedTransferRPUDevice<T>(x_size, d_size, *this, rng);
  };

OneSidedTransferRPUDeviceMetaParameter<T> *clone() const override {
    return new OneSidedTransferRPUDeviceMetaParameter<T>(*this);
  };

DeviceUpdateType implements() const override { return DeviceUpdateType::Transfer; };
void printToStream(std::stringstream &ss) const override;



virtual T getTransferLR(int to_device_idx, int from_device_idx, T current_lr) const;



}

//// start changing
template <typename T> class OneSidedTransferRPUDevice : public VectorRPUDevice<T> {

public:
  // constructor / destructor
  OneSidedTransferRPUDevice(){};
  OneSidedTransferRPUDevice(int x_size, int d_size);
  OneSidedTransferRPUDevice(
      int x_size, int d_size, const OneSidedTransferRPUDeviceMetaParameter<T> &par, RealWorldRNG<T> *rng);
  ~OneSidedTransferRPUDevice();

  OneSidedTransferRPUDevice(const OneSidedTransferRPUDevice<T> &);
  OneSidedTransferRPUDevice<T> &operator=(const OneSidedTransferRPUDevice<T> &);
  OneSidedTransferRPUDevice(OneSidedTransferRPUDevice<T> &&);
  OneSidedTransferRPUDevice<T> &operator=(OneSidedTransferRPUDevice<T> &&);

  friend void swap(OneSidedTransferRPUDevice<T> &a, OneSidedTransferRPUDevice<T> &b) noexcept {
    using std::swap;//TODO : have to modify onesided
    swap(static_cast<VectorRPUDevice<T> &>(a), static_cast<VectorRPUDevice<T> &>(b));

    swap(a.transfer_pwu_, b.transfer_pwu_);
    swap(a.transfer_fb_pass_, b.transfer_fb_pass_);
    swap(a.transfer_vecs_, b.transfer_vecs_);
    swap(a.transfer_every_, b.transfer_every_);
    swap(a.current_slice_indices_, b.current_slice_indices_);
    swap(a.fully_hidden_, b.fully_hidden_);
    swap(a.last_weight_, b.last_weight_);
  }

  OneSidedTransferRPUDeviceMetaParameter<T> &getPar() const override {
    return static_cast<OneSidedTransferRPUDeviceMetaParameter<T> &>(SimpleRPUDevice<T>::getPar());
  };

  OneSidedTransferRPUDevice<T> *clone() const override { return new OneSidedTransferRPUDevice<T>(*this); };
  bool onSetWeights(T **weights) override;

  void decayWeights(T **weights, bool bias_no_decay) override;
  void decayWeights(T **weights, T alpha, bool bias_no_decay) override;
  void driftWeights(T **weights, T time_since_last_call, RNG<T> &rng) override;
  void diffuseWeights(T **weights, RNG<T> &rng) override;
  void clipWeights(T **weights, T clip) override;
  void
  resetCols(T **weights, int start_col, int n_cols, T reset_prob, RealWorldRNG<T> &rng) override;

  void setDeviceParameter(T **out_weights, const std::vector<T *> &data_ptrs) override;
  void setHiddenUpdateIdx(int idx) override{};
  void initUpdateCycle(
      T **weights, const PulsedUpdateMetaParameter<T> &up, T current_lr, int m_batch_info) override;
  void finishUpdateCycle(//TODO check variables
      T **weights, const PulsedUpdateMetaParameter<T> &up, T current_lr, int m_batch_info) override;
  T getPulseCountLearningRate(T learning_rate) override;

  virtual int getTransferEvery(int from_device_idx, int m_batch) const;
  virtual void setTransferVecs(const T *transfer_vecs = nullptr);
  virtual void transfer(int to_device_idx, int from_device_idx, T current_lr);
  virtual void readAndUpdate(
      int to_device_idx,
      int from_device_idx,
      const T lr,
      const T *x_input,
      const int n_vec,
      const T reset_prob,
      const int i_col);
  virtual const T *getTransferVecs() const { return &transfer_vecs_[0]; };
  virtual void writeVector(
      int device_idx, const T *in_vec, const T *out_vec, const T lr, const int m_batch_info);
  virtual void readVector(int device_idx, const T *in_vec, T *out_vec, T alpha);
//TODO: add refresh things
  void doSparseUpdate(
      T **weights, int i, const int *x_signed_indices, int x_count, int d_sign, RNG<T> *rng)
      override;

  void doDenseUpdate(T **weights, int *coincidences, RNG<T> *rng) override;

protected:
  void populate(const OneSidedTransferRPUDeviceMetaParameter<T> &par, RealWorldRNG<T> *rng);
  void reduceToWeights(T **weights) const override;
  T **getDeviceWeights(int device_idx) const;
  int resetCounters(bool force = false) override;

  std::unique_ptr<ForwardBackwardPassIOManaged<T>> transfer_fb_pass_ = nullptr;
  std::unique_ptr<PulsedRPUWeightUpdater<T>> transfer_pwu_ = nullptr;
  std::vector<T> transfer_vecs_;
  std::vector<T> transfer_every_;
  std::vector<int> current_slice_indices_;
  bool fully_hidden_ = false;
  T **last_weight_ = nullptr;

  // no need to swap/copy.
  std::vector<T> transfer_tmp_;
};


}