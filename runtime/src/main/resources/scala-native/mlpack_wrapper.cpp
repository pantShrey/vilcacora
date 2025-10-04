#ifdef MLPACK_Wrapper
#include <mlpack/core.hpp>
#include <mlpack/methods/ann/layer/convolution.hpp>
#include <mlpack/methods/ann/layer/max_pooling.hpp>
#include <armadillo>
#include <cstring>

// ---------------- type aliases ----------------
using FConv = mlpack::ConvolutionType<
    mlpack::NaiveConvolution<mlpack::ValidConvolution>,
    mlpack::NaiveConvolution<mlpack::FullConvolution>,
    mlpack::NaiveConvolution<mlpack::ValidConvolution>,
    arma::fmat>;

using DConv       = mlpack::Convolution;
using FMaxPooling = mlpack::MaxPoolingType<arma::fmat>;
using DMaxPooling = mlpack::MaxPooling;

// ---------------- handle structs --------------
struct ConvHandleF {
  FConv*         layer;
  arma::fmat     inView;
  arma::fmat     outView;
};

struct ConvHandleD {
  DConv*         layer;
  arma::mat      inView;
  arma::mat      outView;
};

struct PoolHandleF {
  FMaxPooling*   layer;
  arma::fmat     inView;
  arma::fmat     outView;
};

struct PoolHandleD {
  DMaxPooling*   layer;
  arma::mat      inView;
  arma::mat      outView;
};

struct SoftmaxF { arma::fmat inView, outView; };
struct SoftmaxD { arma::mat  inView, outView; };

extern "C" {

// =======================================================
//                 FLOAT 32-bit CONVOLUTION
// =======================================================
ConvHandleF* initialise_conv_f(
    size_t  outMaps, size_t kH, size_t kW,
    size_t  sH, size_t sW, int autoPad, int useBias,
    size_t  inH, size_t inW, size_t inC,
    const float* wPtr, const float* bPtr,
    float* inputPtr,  float* outputPtr) {

  auto* layer = new FConv(outMaps, kW, kH, sW, sH, 0, 0,
                          (autoPad ? "same" : "valid"), useBias);
  layer->InputDimensions() = {inH, inW, inC};
  layer->ComputeOutputDimensions();

  arma::fcube wCube(const_cast<float*>(wPtr),
                    kW, kH, outMaps * inC, false, false);
  layer->Weight() = wCube;

  if (useBias) {
    arma::fmat bMat(const_cast<float*>(bPtr), outMaps, 1, false, false);
    layer->Bias() = bMat;
  }

  const auto& od = layer->OutputDimensions();
  size_t outElems = od[0] * od[1] * od[2];

  arma::fmat inView (inputPtr,  inH * inW * inC, 1, false, false);
  arma::fmat outView(outputPtr, outElems,          1, false, false);

  return new ConvHandleF{ layer, std::move(inView), std::move(outView) };
}

void execute_conv_f(ConvHandleF* h) {
  h->layer->Forward(h->inView, h->outView);
}

void cleanup_conv_f(ConvHandleF* h) {
  delete h->layer;
  delete h;
}

// =======================================================
//                 FLOAT 64-bit CONVOLUTION
// =======================================================
ConvHandleD* initialise_conv_d(
    size_t  outMaps, size_t kH, size_t kW,
    size_t  sH, size_t sW, int autoPad, int useBias,
    size_t  inH, size_t inW, size_t inC,
    const double* wPtr, const double* bPtr,
    double* inputPtr,  double* outputPtr) {

  auto* layer = new DConv(outMaps, kW, kH, sW, sH, 0, 0,
                          (autoPad ? "same" : "valid"), useBias);
  layer->InputDimensions() = {inH, inW, inC};
  layer->ComputeOutputDimensions();

  arma::cube wCube(const_cast<double*>(wPtr),
                   kW, kH, outMaps * inC, false, false);
  layer->Weight() = wCube;

  if (useBias) {
    arma::mat bMat(const_cast<double*>(bPtr), outMaps, 1, false, false);
    layer->Bias() = bMat;
  }

  const auto& od = layer->OutputDimensions();
  size_t outElems = od[0] * od[1] * od[2];

  arma::mat inView (inputPtr,  inH * inW * inC, 1, false, false);
  arma::mat outView(outputPtr, outElems,         1, false, false);

  return new ConvHandleD{ layer, std::move(inView), std::move(outView) };
}

void execute_conv_d(ConvHandleD* h) {
  h->layer->Forward(h->inView, h->outView);
}

void cleanup_conv_d(ConvHandleD* h) {
  delete h->layer;
  delete h;
}

// =======================================================
//                     FLOAT 32-bit MAXPOOL
// =======================================================
PoolHandleF* initialise_pool_f(
    size_t kH, size_t kW,
    size_t sH, size_t sW,
    size_t inH, size_t inW, size_t inC,
    float* inputPtr,  float* outputPtr) {

  auto* layer = new FMaxPooling(kW, kH, sW, sH);
  layer->InputDimensions() = {inH, inW, inC};
  layer->ComputeOutputDimensions();

  const auto& od = layer->OutputDimensions();
  size_t outElems = od[0] * od[1] * od[2];

  arma::fmat inView (inputPtr,  inH * inW * inC, 1, false, false);
  arma::fmat outView(outputPtr, outElems,         1, false, false);

  return new PoolHandleF{ layer, std::move(inView), std::move(outView) };
}

void execute_pool_f(PoolHandleF* h) {
  h->layer->Forward(h->inView, h->outView);
}

void cleanup_pool_f(PoolHandleF* h) {
  delete h->layer;
  delete h;
}

// =======================================================
//                     FLOAT 64-bit MAXPOOL
// =======================================================
PoolHandleD* initialise_pool_d(
    size_t kH, size_t kW,
    size_t sH, size_t sW,
    size_t inH, size_t inW, size_t inC,
    double* inputPtr,  double* outputPtr) {

  auto* layer = new DMaxPooling(kW, kH, sW, sH);
  layer->InputDimensions() = {inH, inW, inC};
  layer->ComputeOutputDimensions();

  const auto& od = layer->OutputDimensions();
  size_t outElems = od[0] * od[1] * od[2];

  arma::mat inView (inputPtr,  inH * inW * inC, 1, false, false);
  arma::mat outView(outputPtr, outElems,         1, false, false);

  return new PoolHandleD{ layer, std::move(inView), std::move(outView) };
}

void execute_pool_d(PoolHandleD* h) {
  h->layer->Forward(h->inView, h->outView);
}

void cleanup_pool_d(PoolHandleD* h) {
  delete h->layer;
  delete h;
}

// =======================================================
//                          SOFTMAX
// =======================================================
void F_perform_softmax_direct(
    const float* input_ptr, size_t input_size,
    float* output_ptr) // Scala pre-allocates same size as input
{
    // Direct computation - no intermediate allocations
    float max_val = *std::max_element(input_ptr, input_ptr + input_size);
    
    float sum = 0.0f;
    for (size_t i = 0; i < input_size; ++i) {
        output_ptr[i] = std::exp(input_ptr[i] - max_val);
        sum += output_ptr[i];
    }
    
    for (size_t i = 0; i < input_size; ++i) {
        output_ptr[i] /= sum;
    }
}
void perform_softmax_direct(
    const double* input_ptr, size_t input_size,
    double* output_ptr) // Scala pre-allocates same size as input
{
    // Direct computation - no intermediate allocations
    double max_val = *std::max_element(input_ptr, input_ptr + input_size);
    
    double sum = 0.0;
    for (size_t i = 0; i < input_size; ++i) {
        output_ptr[i] = std::exp(input_ptr[i] - max_val);
        sum += output_ptr[i];
    }
    
    for (size_t i = 0; i < input_size; ++i) {
        output_ptr[i] /= sum;
    }

} 

} // extern "C"
#endif