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

template <typename T> struct OneSidedTransferRPUDeviceMetaParameter : VectorRPUDeviceMetaParameter<T> {

T gamma= (T)0.0;

T transfer_every = (T)1.0;
bool units_in_mbatch = false;//have to check
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

std::vector<T> transfer_every_vec; // IBM TODO

IOMetaParameter<T> transfer_io;
PulsedUpdateMetaParameter<T> transfer_up;



int refresh_every = 0; // refresh every x updates (ie counting single vector updates)
IOMetaParameter<T> refresh_io;           // the IO for reading out during refresh
PulsedUpdateMetaParameter<T> refresh_up; // UP parameters for refresh
T refresh_upper_thres = 0.75;
T refresh_lower_thres = 0.25;
bool copy_inverted = false; // whether to use copy inverted for second device

OneSidedTransferRPUDeviceMetaParameter(){};
OneSidedTransferRPUDeviceMetaParameter(const PulsedRPUDeviceMetaParameterBase<T> &dp, int n_devices)
    : VectorRPUDeviceMetaParameter<T>(dp, n_devices){};

OneSidedTransferRPUDeviceMetaParameter(
    const PulsedRPUDeviceMetaParameterBase<T> &dp_fast,
    const PulsedRPUDeviceMetaParameterBase<T> &dp_rest,
    int n_total_devices);

virtual void initializeWithSize(int x_size, int d_size);
void initialize() override{/* do nothing */};
inline bool fullyHidden() const { return (!gamma && this->gamma_vec.back() == 1.0); };

inline int getInSize() const { return _in_size; };
inline int getOutSize() const { return _out_size; };


std::string getName() const override {
    std::ostringstream ss;
    ss << "OneSided_Transfer(" << this->vec_par.size() << ")";
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
  DeviceUpdateType implements() const override { return DeviceUpdateType::OneSided; };
  void printToStream(std::stringstream &ss) const override;

  T calcWeightGranularity() const override {
    T weight_granularity = 0.0;
    if (this->vec_par.size() > 0) {
      // only take that from first (fast) device
      weight_granularity = this->vec_par[0]->calcWeightGranularity();
    }
    return weight_granularity;
  }

  virtual T getTransferLR(int to_device_idx, int from_device_idx, T current_lr) const;

};


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
    using std::swap;
    swap(static_cast<VectorRPUDevice<T> &>(a), static_cast<VectorRPUDevice<T> &>(b));

    swap(a.transfer_pwu_, b.transfer_pwu_);
    swap(a.transfer_fb_pass_, b.transfer_fb_pass_);
    swap(a.transfer_vecs_, b.transfer_vecs_);
    swap(a.transfer_every_, b.transfer_every_);
    swap(a.current_slice_indices_, b.current_slice_indices_);
    swap(a.fully_hidden_, b.fully_hidden_);
    swap(a.last_weight_, b.last_weight_);
    
    swap(a.g_plus_, b.g_plus_);
    swap(a.g_minus_, b.g_minus_);
    swap(a.a_indices_, b.a_indices_);
    swap(a.b_indices_, b.b_indices_);
    swap(a.refresh_counter_, b.refresh_counter_);
    swap(a.refresh_fb_pass_, b.refresh_fb_pass_);
    swap(a.refresh_pwu_, b.refresh_pwu_);
    swap(a.refresh_vecs_, b.refresh_vecs_);
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
  void invert();
  void setDeviceParameter(T **out_weights, const std::vector<T *> &data_ptrs) override;
  void setHiddenUpdateIdx(int idx) override{};
  void initUpdateCycle(
      T **weights, const PulsedUpdateMetaParameter<T> &up, T current_lr, int m_batch_info) override;
  void finishUpdateCycle(
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

  void doSparseUpdate(
      T **weights, int i, const int *x_signed_indices, int x_count, int d_sign, RNG<T> *rng)
      override;

  void doDenseUpdate(T **weights, int *coincidences, RNG<T> *rng) override;


  virtual T **getPosWeights(int device_idx) { return this->getWeightVec()[g_plus_[device_idx]]; };
  virtual T **getNegWeights(int device_idx) { return this->getWeightVec()[g_minus_[device_idx]]; };

  inline uint64_t getRefreshCount() const { return refresh_counter_; };


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


 std::unique_ptr<ForwardBackwardPassIOManaged<T>> refresh_fb_pass_ = nullptr;
  std::unique_ptr<PulsedRPUWeightUpdater<T>> refresh_pwu_ = nullptr;
  std::vector<T> refresh_vecs_;

  inline bool
  refreshCriterion(T &wp, T &wm, T &w_max, T &w_min, T &upper_thres, T &lower_thres) const;

  private:
  bool isInverted() const;
  int refreshWeights();
  void setRefreshVecs();

  std::vector<int> g_plus_;
  std::vector<int> g_minus_ ;
  uint64_t refresh_counter_ = 0;

  std::vector<int> a_indices_;
  std::vector<int> b_indices_;

  // temporary: no need to copy
  std::vector<T> refresh_p_tmp_;
  std::vector<T> refresh_m_tmp_;
  std::vector<T> refresh_p_vec_;
  std::vector<T> refresh_m_vec_;
  std::vector<int> coincidences_p_;
  std::vector<int> coincidences_m_;
};




}
