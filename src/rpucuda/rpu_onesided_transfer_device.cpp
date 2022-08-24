/**
 * (C) Copyright 2020, 2021, 2022 IBM. All Rights Reserved.
 *
 * This code is licensed under the Apache License, Version 2.0. You may
 * obtain a copy of this license in the LICENSE.txt file in the root directory
 * of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
 *
 * Any modifications or derivative works of this code must retain this
 * copyright notice, and modified files need to carry a notice indicating
 * that they have been altered from the originals.
 
 ** Transfer device which is consistent of multiple onesided devices is developed 
 */


#include "rpu_onesided_transfer_device.h"
#include "math_util.h"
#include "utility_functions.h"
#include <algorithm>
#include <memory>
#include <sstream>


namespace RPU{


/* OneSidedTransferRPUDeviceMetaParameter*/
template <typename T>
OneSidedTransferRPUDeviceMetaParameter<T>::OneSidedTransferRPUDeviceMetaParameter(
     const PulsedRPUDeviceMetaParameterBase<T> &dp_fast,
    const PulsedRPUDeviceMetaParameterBase<T> &dp_rest,
    int n_total_devices) {
  this->vec_par.clear();
  if (n_total_devices < 2) {
    RPU_FATAL("More or equal than 2 devices expected.");
  }
  this->appendVecPar(dp_fast);// OT : g+ side parameter
  this->appendVecPar(dp_fast);// OT : g- side parameter
  for (int i = 1; i < n_total_devices; i++) {
    this->appendVecPar(dp_rest);//OT : G+ side parameter
    this->appendVecPar(dp_rest);//OT : G- side parameter
  }
};


// TO DO : modify print both ones
template <typename T>
void OnesidedTransferRPUDeviceMetaParameter<T>::printToStream(std::stringstream &ss) const {
  ss << this->getName() << std::endl;
  // gamma
  ss << "\tgamma:\t\t\t";
  if (this->_par_initialized)
    for (size_t k = 0; k < this->gamma_vec.size(); k++) {
      ss << this->gamma_vec[k] << " ";
    }
  else {
    ss << gamma;
  }
  ss << std::endl;

  // every
  ss << "\ttransfer_every [init]: \t";
  if (this->_par_initialized)
    for (size_t k = 0; k < transfer_every_vec.size(); k++) {
      ss << transfer_every_vec[k] << " ";
    }
  else {
    ss << transfer_every;
  }
  if (units_in_mbatch) {
    ss << " [in mbatches]";
  }
  ss << std::endl;

  // lr
  if (fast_lr > 0) {
    ss << "\tfast_lr:\t\t";
    ss << fast_lr;
    ss << std::endl;
  }

  ss << "\ttransfer_lr: \t\t";
  if (this->_par_initialized)
    for (size_t k = 0; k < transfer_lr_vec.size(); k++) {
      ss << transfer_lr_vec[k] << " ";
    }
  else {
    ss << transfer_lr;
  }
  if (scale_transfer_lr) {
    ss << "\t[scaled with current LR]";
  }
  ss << std::endl;

  ss << "\tn_reads_per_transfer: \t" << n_reads_per_transfer;
  if (transfer_columns) {
    ss << "\t[reading columns]";
  } else {
    ss << "\t[reading rows]";
  }
  ss << std::endl;

  if (with_reset_prob) {
    ss << "\t[with reset p=" << with_reset_prob << "]";
  }
  if (random_selection) {
    ss << "\t[random selection]";
  }
  ss << std::endl;

  ss << "   Transfer IO: \n";
  transfer_io.printToStream(ss);
  ss << "   Transfer Update Parameter: \n";
  transfer_up.printToStream(ss);

  for (size_t k = 0; k < this->vec_par.size(); k++) {
    ss << "   Device Parameter " << k << ": " << this->vec_par[k]->getName() << std::endl;
    ss << "   ";
    this->vec_par[k]->printToStream(ss);
  }


};


template <typename T>
void OneSidedTransferRPUDeviceMetaParameter<T>::initializeWithSize(int x_size, int d_size) {
  // check for _par_initialized ? Maybe just force?

  VectorRPUDeviceMetaParameter<T>::initialize();

  size_t n_devices = this->vec_par.size();

  if (n_devices < 2) {
    RPU_FATAL("Need at least 2 devices");
  }

  if (transfer_columns) {
    _in_size = x_size;
    _out_size = d_size;
  } else {
    _in_size = d_size;
    _out_size = x_size;
  }

  this->update_policy = VectorDeviceUpdatePolicy::SingleFixed;// TO DO: check how update_policy affect in update cycle
  this->first_update_idx = 0; // only first is updated
  this->same_context = true;

  // Only the first device might be different from the rest,
  // because we use only 2 pulsed weight updater
  auto impl = this->vec_par[1]->implements();
  for (size_t i = 2; i < n_devices; i++) {
    if (impl != this->vec_par[i]->implements()) {
      RPU_FATAL("Only the first device can be a different RPU device. ");
    }
  }

  // weightening of devices to get final weights // have to check for g plus and g minus
  if (this->gamma_vec.size() > 0) {
    if (this->gamma_vec.size() != n_devices) {
      RPU_FATAL("If gamma_vec is set manually expect the same size as number of devices.");
    }
    T g = 0;
    for (size_t i = 0; i < n_devices - 1; i++) {
      g += this->gamma_vec[i];
    }
    if (this->gamma_vec[n_devices - 1] == 0) {
      RPU_FATAL("Expect that last device has some constribution to the network weights. [otherwise "
                "why transfer?]");
    }
    gamma = g;
  }
  if (this->gamma_vec.size() == 0) {
    this->gamma_vec.resize(n_devices);
    for (size_t i = 0; i < n_devices; i++) {
        this->gamma_vec[n_devices - i - 1] = pow(-1,i%2)*pow(gamma, (T)i/(T)2);//OT : considered G+ G-
    }
  }

  if (transfer_lr_vec.size() == 0) {
    transfer_lr_vec.resize(n_devices);
    std::fill(transfer_lr_vec.begin(), transfer_lr_vec.end(), transfer_lr);
  }
  if (transfer_lr_vec.size() != n_devices) {
    RPU_FATAL("Expect transfer_lr_vec of size n_devices.");
  }

  if (n_reads_per_transfer > _in_size) {
    // should not be needed anyway
    n_reads_per_transfer = _in_size;
    RPU_WARNING("too many transfers in one shot. Using full transfer instead.");
  }

  // IBM TODO: make an default value, where the value of the transfer depends on  x_size

  if (transfer_every_vec.size() == 0) {
    T n = transfer_every;
    for (size_t i = 0; i < n_devices; i=i+2) { // OT : 2 devices are one set.
      transfer_every_vec.push_back(n);
      n *= (T)_in_size / n_reads_per_transfer;
    }
    if (no_self_transfer) {
      transfer_every_vec[n_devices - 1] = 0;
      transfer_every_vec[n_devices - 2] = 0;
    }
  }

  if (transfer_every_vec.size() != n_devices) {
    RPU_FATAL("Expect transfer_every_vec to be of length n_devices");
  }

  if (with_reset_prob > 0 && !transfer_columns) {
    RPU_FATAL("Reset prob is only implemented for column-transfer so far.");
  }

  // IO
  if (transfer_columns) {
    transfer_io.initializeForForward();
  } else {
    transfer_io.initializeForBackward();
  }

  // we turn BM off.
  if (transfer_io.bound_management != BoundManagementType::None) {
    RPU_WARNING("Transfer bound management turned off.");
    transfer_io.bound_management = BoundManagementType::None;
  }

  // up
  transfer_up.initialize();
}

template <typename T>
T OnesidedTransferRPUDeviceMetaParameter<T>::getTransferLR(
    int to_device_idx, int from_device_idx, T current_lr) const {

  T lr = transfer_lr_vec[from_device_idx];
  if (scale_transfer_lr) {
    lr *= current_lr;
  }
  return lr;
}

template struct OneSidedTransferRPUDeviceMetaParameter<float>;
#ifdef RPU_USE_DOUBLE
template struct OneSidedTransferRPUDeviceMetaParameter<double>;
#endif

// dtor
template <typename T> OneSidedTransferRPUDevice<T>::~OneSidedTransferRPUDevice() {}

// ctor
template <typename T>
OneSidedTransferRPUDevice<T>::OneSidedTransferRPUDevice(int x_sz, int d_sz) : VectorRPUDevice<T>(x_sz, d_sz) {}


template <typename T>
OneSidedTransferRPUDevice<T>::OneSidedTransferRPUDevice(
    int x_sz, int d_sz, const OneSidedTransferRPUDeviceMetaParameter<T> &par, RealWorldRNG<T> *rng)
    : OneSidedTransferRPUDevice<T>(x_sz, d_sz) {
  populate(par, rng);
}


// copy construcutor
template <typename T>
OneSidedTransferRPUDevice<T>::OneSidedTransferRPUDevice(const OneSidedTransferRPUDevice<T> &other)
    : VectorRPUDevice<T>(other) {

//onesided param copy
  g_plus_ = other.g_plus_;
  g_minus_ = other.g_minus_;
  a_indices_ = other.a_indices_;
  b_indices_ = other.b_indices_;
 

  refresh_fb_pass_ = make_unique<ForwardBackwardPassIOManaged<T>>(*other.refresh_fb_pass_);
  refresh_pwu_ = make_unique<PulsedRPUWeightUpdater<T>>(*other.refresh_pwu_);
  refresh_counter_ = other.refresh_counter_;
  refresh_vecs_ = other.refresh_vecs_;


//transfer param copy


  transfer_fb_pass_ = make_unique<ForwardBackwardPassIOManaged<T>>(*other.transfer_fb_pass_);
  transfer_pwu_ = make_unique<PulsedRPUWeightUpdater<T>>(*other.transfer_pwu_);

  current_slice_indices_ = other.current_slice_indices_;
  transfer_vecs_ = other.transfer_vecs_;
  transfer_every_ = other.transfer_every_;
  fully_hidden_ = other.fully_hidden_;
  last_weight_ = other.last_weight_;


}



// copy assignment
template <typename T>
OneSidedTransferRPUDevice<T> &OneSidedTransferRPUDevice<T>::operator=(const OneSidedTransferRPUDevice<T> &other) {

  OneSidedTransferRPUDevice<T> tmp(other);
  swap(*this, tmp);
  return *this;
}

// move constructor
template <typename T> OneSidedTransferRPUDevice<T>::OneSidedTransferRPUDevice(OneSidedTransferRPUDevice<T> &&other) {
  *this = std::move(other);
}

// move assignment
template <typename T>
OneSidedTransferRPUDevice<T> &OneSidedTransferRPUDevice<T>::operator=(OneSidedTransferRPUDevice<T> &&other) {
  VectorRPUDevice<T>::operator=(std::move(other));
 
 //onesided part
  g_plus_ = other.g_plus_;
  g_minus_ = other.g_minus_;
  a_indices_ = std::move(other.a_indices_);
  b_indices_ = std::move(other.b_indices_);


  refresh_fb_pass_ = std::move(other.refresh_fb_pass_);
  refresh_pwu_ = std::move(other.refresh_pwu_);
  refresh_counter_ = other.refresh_counter_;
  refresh_vecs_ = std::move(other.refresh_vecs_);
// transfer part
  current_slice_indices_ = std::move(other.current_slice_indices_);
  transfer_vecs_ = std::move(other.transfer_vecs_);
  transfer_every_ = std::move(other.transfer_every_);
  transfer_fb_pass_ = std::move(other.transfer_fb_pass_);
  transfer_pwu_ = std::move(other.transfer_pwu_);
  last_weight_ = std::move(other.last_weight_);
  fully_hidden_ = other.fully_hidden_;
  return *this;
}


//Onesided Functions


template <typename T> void OneSidedTransferRPUDevice<T>::setRefreshVecs() {
  refresh_vecs_.resize(this->x_size_ * this->x_size_); //!!  square matrix
  std::fill(refresh_vecs_.begin(), refresh_vecs_.end(), (T)0.0);

  // initialize refresh vectors with unit vectors. This might be overridden
  for (size_t i = 0; i < refresh_vecs_.size(); i += this->x_size_ + 1) {
    refresh_vecs_[i] = 1.0;
  }
}

//Transfer Functions


template <typename T> void TransferRPUDevice<T>::setTransferVecs(const T *transfer_vecs) {
  T in_size = getPar().getInSize();

  transfer_vecs_.resize(in_size * in_size); //!!  square matrix
  std::fill(transfer_vecs_.begin(), transfer_vecs_.end(), (T)0.0);

  if (transfer_vecs == nullptr) {
    // initialize transfer vectors with unit vectors. This might be overridden
    for (size_t i = 0; i < transfer_vecs_.size(); i += in_size + 1) {
      transfer_vecs_[i] = 1.0;
    }
  } else {
    // Caution: No size check!
    for (size_t i = 0; i < transfer_vecs_.size(); i++) {
      transfer_vecs_[i] = transfer_vecs[i];
    }
  }
}

//Both

template <typename T> int OneSidedTransferRPUDevice<T>::resetCounters(bool force) {
  refresh_counter_ = 0;
 
 //transfer part
  current_slice_indices_.resize(this->n_devices_/2); // OT : current slice size is half because 2 devices are one set.
  std::fill(current_slice_indices_.begin(), current_slice_indices_.end(), (int)0);
 
  return VectorRPUDevice<T>::resetCounters(force);
}



template <typename T>
void OneSidedTransferRPUDevice<T>::populate(
    const OneSidedTransferRPUDeviceMetaParameter<T> &p, RealWorldRNG<T> *rng) {

  VectorRPUDevice<T>::populate(p, rng);
  auto &par = getPar();
   par.initializeWithSize(this->x_size_, this->d_size_);
 if (par.copy_inverted) {
  for(int i=0;i<this->n_devices_/2;i++)
    this->rpu_device_vec_[i*2+1]->copyInvertDeviceParameter(&*this->rpu_device_vec_[i*2]);
  }

 

  auto shared_rng = std::make_shared<RNG<T>>(0); // we just take a new one here (seeds...)
  transfer_fb_pass_ =
      RPU::make_unique<ForwardBackwardPassIOManaged<T>>(this->x_size_, this->d_size_, shared_rng);

  transfer_fb_pass_->setIOPar(par.transfer_io, par.transfer_io);

  transfer_pwu_ =
      RPU::make_unique<PulsedRPUWeightUpdater<T>>(this->x_size_, this->d_size_, shared_rng);
  transfer_pwu_->setUpPar(par.transfer_up);

  for(int k=0;k<this->n_devices_/2;k++)
{  g_plus_[k] = 2*k+1;
  g_minus_[k] = 2*k;
}
  this->reduce_weightening_.resize(this->n_devices_);
  for(int k=0;k<this->n_devices_;k++)
  {
  this->reduce_weightening_[g_plus_[k]] = par.gamma_vec[k];
  this->reduce_weightening_[g_minus_[k]] = -par.gamma_vec[k];
  }



  resetCounters(); // "state" pars
  setTransferVecs();
  transfer_every_ = par.transfer_every_vec; // already checked for length
  fully_hidden_ = getPar().fullyHidden(); 



 this->setRefreshVecs();
  auto shared_rng = std::make_shared<RNG<T>>(0); // we just take a new one here (seeds...)
  refresh_fb_pass_ =
      RPU::make_unique<ForwardBackwardPassIOManaged<T>>(this->x_size_, this->d_size_, shared_rng);
  refresh_fb_pass_->setIOPar(par.refresh_io, par.refresh_io);

  refresh_pwu_ =
      RPU::make_unique<PulsedRPUWeightUpdater<T>>(this->x_size_, this->d_size_, shared_rng);
  refresh_pwu_->setUpPar(par.refresh_up);



    }

template <typename T> bool OneSidedTransferRPUDevice<T>::isInverted() const { return g_plus_[0] == 0; }

template <typename T> inline void OneSidedRPUDevice<T>::invert() { //OT TO DO: invert nth device by add parameter 
  std::swap(g_plus_[0], g_minus_[0]);
  this->reduce_weightening_[g_plus_[0]] = 1;
  this->reduce_weightening_[g_minus_[0]] = -1;
}



template <typename T>
void OneSidedTransferRPUDevice<T>::initUpdateCycle(
    T **weights, const PulsedUpdateMetaParameter<T> &up, T current_lr, int m_batch_info) {

  VectorRPUDevice<T>::initUpdateCycle(weights, up, current_lr, m_batch_info);

  if (a_indices_.size() < (size_t)up.desired_BL) {
    a_indices_.resize(up.desired_BL);
    b_indices_.resize(up.desired_BL);
  }
}


template <typename T>
void OneSidedTransferRPUDevice<T>::finishUpdateCycle(
    T **weights, const PulsedUpdateMetaParameter<T> &up, T current_lr, int m_batch_info) {

  VectorRPUDevice<T>::finishUpdateCycle(weights, up, current_lr, m_batch_info);

  const auto &par = getPar();
  if (par.refresh_every > 0) {
    int refresh_every = par.refresh_every;
    if (par.units_in_mbatch) {
      refresh_every *= m_batch_info;
    }
    int refresh_count = 0;
    if (this->current_update_idx_ % refresh_every == 0) {
      refresh_count += refreshWeights();
    }
    if (refresh_count > 0) {
      this->reduceToWeights(weights);
    }
    refresh_counter_ += refresh_count;
  }


//transfer part
// directly onto last weight if fully hidden. No reduce needed
  last_weight_ = fully_hidden_ ? weights : nullptr;

  // we transfer the device here to cope with the sparse update below.
  for (int j = 0; j < this->n_devices/2_; j++) {
    int every = getTransferEvery(j, m_batch_info);
    if (every > 0 && this->current_update_idx_ % every == 0) {
      // last is self-update (does nothing per default, but could implement refresh in child)
      transfer(MIN(j + 1, this->n_devices_ - 1), j, current_lr);
    }
  }





}

//Updates
//onesided based
template <typename T>
void OneSidedTransferRPUDevice<T>::doSparseUpdate(
    T **weights, int d_index, const int *x_signed_indices, int x_count, int d_sign, RNG<T> *rng) {

  int a_count = 0;
  int b_count = 0;

  for (int jj = 0; jj < x_count; jj++) {
    int j_signed = x_signed_indices[jj];
    int sign = (j_signed < 0) ? -d_sign : d_sign;

    if (sign > 0) { // a per default g-
      a_indices_[a_count++] =
          (j_signed > 0)
              ? j_signed
              : -j_signed; // always one sided update (to positive side, see also -1 below)

    } else { // b per default g+
      b_indices_[b_count++] = (j_signed > 0) ? j_signed : -j_signed;
    }
  }

  if (a_count > 0) {
    this->rpu_device_vec_[g_minus_[0]]->doSparseUpdate(
        this->weights_vec_[g_minus_[0]], d_index, a_indices_.data(), a_count, -1, rng);
  }
  if (b_count > 0) {
    this->rpu_device_vec_[g_plus_[0]]->doSparseUpdate(
        this->weights_vec_[g_plus_[0]], d_index, b_indices_.data(), b_count, -1, rng);
  }
  // update the changed weight indices // note that this is very
  // repetitive since the same indices might be present all the
  // time. However, should be cached at Onesided but not in transfer onesided
  // 
  //for (int jj = 0; jj < x_count; jj++) {
   // int j_signed = x_signed_indices[jj];
   // int j = (j_signed < 0) ? -j_signed - 1 : j_signed - 1;
   // weights[d_index][j] =
   //     this->weights_vec_[g_plus_[0]][d_index][j] - this->weights_vec_[g_minus_[0]][d_index][j];
  //}
}


template <typename T>
void OneSidedRPUDevice<T>::doDenseUpdate(T **weights, int *coincidences, RNG<T> *rng) {

  coincidences_p_.resize(this->size_);
  coincidences_m_.resize(this->size_);

  PRAGMA_SIMD
  for (int i = 0; i < this->size_; i++) {
    int c = coincidences[i];

    coincidences_p_[i] = c < 0 ? c : 0;
    coincidences_m_[i] = c > 0 ? -c : 0;
  }

  this->rpu_device_vec_[g_plus_[0]]->doDenseUpdate(
      this->weights_vec_[g_plus_[0]], coincidences_p_.data(), rng);
  this->rpu_device_vec_[g_minus_[0]]->doDenseUpdate(
      this->weights_vec_[g_minus_[0]], coincidences_m_.data(), rng);

  //  this might be better called in finish update cycle and only once per mini-batch?
  this->reduceToWeights(weights);
}


//Transfer


template <typename T> T **OneSidedTransferRPUDevice<T>::getDeviceWeights(int device_idx) const {

  if (fully_hidden_ && device_idx == this->n_devices_ - 1) {
    return last_weight_;
  } else {
    return this->weights_vec_[device_idx];
  }
}

template <typename T>
int OneSidedTransferRPUDevice<T>::getTransferEvery(int from_device_idx, int m_batch) const {

  if (getPar().units_in_mbatch) {
    return MAX(ceil(transfer_every_[from_device_idx] * m_batch), 0);
  } else {
    return MAX(round(transfer_every_[from_device_idx]), 0);
  }
}

template <typename T>
void OneSidedTransferRPUDevice<T>::readVector(int device_idx, const T *in_vec, T *out_vec, T alpha) {// OT: read difference between g+,g-
  T **W_plus = getDeviceWeights(g_plus_[device_idx]);
  T **W_minus = getDeviceWeights(g_minus_[device_idx]);
  if (getPar().transfer_columns) {
    transfer_fb_pass_->forwardVector(W_plus, in_vec, 1, out_vec, 1, alpha, false);
    transfer_fb_pass_->forwardVector(W_minus, in_vec,1,out_vec,1,-alpha,false);
  } else {
    transfer_fb_pass_->backwardVector(W_plus, in_vec, 1, out_vec, 1, alpha);
    trabsfer_fb_pass_->backwardVector(W_minus,in_vec,1,out_vec,1,-alpha);
  }
}

template <typename T>
void OneSidedTransferRPUDevice<T>::writeVector(
    int device_idx, const T *in_vec, const T *out_vec, const T lr, const int m_batch_info) {

  T **W_plus = getDeviceWeights(g_plus_[device_idx]);
  T **W_minus = getDeviceWeights(g_minus_[device_idx]);
  T **W=nullptr; //TO DO // HAVE TO CHECK
  if (getPar().transfer_columns) {
    // in_vec is x_input
    transfer_pwu_->updateVectorWithDevice(
        W, in_vec, 1, out_vec, 1, lr, m_batch_info, &*this->rpu_device_vec_[device_idx]);
         for(int i=0;i<this->x_size;i++){
  for(int j=0;j<this->d_size;j++)
{
if(W[i][j]>0)
{
W_plus[i][j]+=W[i][j];
}
else
{
W_minus[i][j]-=W[i][j];
}
}
  }
  } else {
    // in_vec is d_input
    transfer_pwu_->updateVectorWithDevice(
        W, out_vec, 1, in_vec, 1, lr, m_batch_info, &*this->rpu_device_vec_[device_idx]);
         for(int i=0;i<this->x_size;i++){
  for(int j=0;j<this->d_size;j++)
{
if(W[j][i]>0)
{
W_plus[j][i]+=W[j][i];
}
else
{
W_minus[j][i]-=W[j][i];
}
}
  }
  }
 
}

template <typename T>
void OneSidedTransferRPUDevice<T>::readAndUpdate( //TODO :have to update G+ if value is + else update G-
    int to_device_idx,
    int from_device_idx,
    const T lr,
    const T *vec, // these are the selected transfer vecs
    const int n_vec,
    const T reset_prob,
    const int i_slice) {

  if (lr == 0.0) {
    return;
  }

  if (to_device_idx == from_device_idx) {
    // self update not supported per default
    return;
  }
  const auto &par = getPar();

  int in_size = par.getInSize();
  int out_size = par.getOutSize();

  transfer_tmp_.resize(out_size);

  // forward or backward / update
  for (int i = 0; i < n_vec; i++) {
    const T *v = vec + i * in_size;

    readVector(from_device_idx, v, transfer_tmp_.data(), -1.0); // scale -1 for pos update

    if (this->rw_rng_.sampleUniform() < reset_prob && par.transfer_columns) {
      // potentially reset here (because of possible same device to-from):
      // NOTE that with_reset_prob is COL-wise prob (elem device prob is 1)
      T **W_from = getDeviceWeights(from_device_idx);
      this->rpu_device_vec_[from_device_idx]->resetCols(W_from, i_slice, n_vec, 1, this->rw_rng_);
    }

    // update according to device
    writeVector(to_device_idx, v, transfer_tmp_.data(), lr, n_vec);
  }
}

template <typename T>
void OneSidedTransferRPUDevice<T>::transfer(int to_device_idx, int from_device_idx, T current_lr) {

  int i_slice = current_slice_indices_[from_device_idx];
  const auto &par = getPar();

  int in_size = par.getInSize();

  if (par.random_selection) {
    i_slice = MAX(MIN(floor(this->rw_rng_.sampleUniform() * in_size), in_size - 1), 0);
  }

  // transfer_vecs_ is always in_size-major (that is trans==false)
  T *tvec = &transfer_vecs_[0] + i_slice * in_size;
  int n_rest = in_size - i_slice;

  T lr = par.getTransferLR(to_device_idx, from_device_idx, current_lr);

  int n_transfer = MIN(par.n_reads_per_transfer, in_size);

  if (n_rest < n_transfer) {
    // rest

    readAndUpdate(to_device_idx, from_device_idx, lr, tvec, n_rest, par.with_reset_prob, i_slice);
    // from beginning
    readAndUpdate(
        to_device_idx, from_device_idx, lr, &transfer_vecs_[0], n_transfer - n_rest,
        par.with_reset_prob, 0);

  } else {
    readAndUpdate(
        to_device_idx, from_device_idx, lr, tvec, n_transfer, par.with_reset_prob, i_slice);
  }

  current_slice_indices_[from_device_idx] = (i_slice + n_transfer) % in_size;
}









}
