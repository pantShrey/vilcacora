#ifdef MLPACK_Wrapper
#include <mlpack/core.hpp>
#include <mlpack/methods/ann/layer/convolution.hpp>
#include <mlpack/methods/ann/layer/max_pooling.hpp>
#include <armadillo>
#include <cstring>

// ---------------- type aliases ----------------
using FConv       = mlpack::Convolution<arma::fmat>;
using DConv       = mlpack::Convolution<arma::mat>;
using FMaxPooling = mlpack::MaxPooling<arma::fmat>;
using DMaxPooling = mlpack::MaxPooling<arma::mat>;

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
        const float* input_ptr, 
        size_t outer_size, 
        size_t inner_size,
        float* output_ptr) 
    {
        for (size_t o = 0; o < outer_size; ++o) {
            size_t offset = o * inner_size;
            const float* in_row = input_ptr + offset;
            float* out_row = output_ptr + offset;

            // Find max for numerical stability (calculated per row)
            float max_val = *std::max_element(in_row, in_row + inner_size);
            
            float sum = 0.0f;
            for (size_t i = 0; i < inner_size; ++i) {
                out_row[i] = std::exp(in_row[i] - max_val);
                sum += out_row[i];
            }
            
            // Normalize the row
            for (size_t i = 0; i < inner_size; ++i) {
                out_row[i] /= sum;
            }
        }
    }

    void perform_softmax_direct(
        const double* input_ptr, 
        size_t outer_size, 
        size_t inner_size,
        double* output_ptr) 
    {
        for (size_t o = 0; o < outer_size; ++o) {
            size_t offset = o * inner_size;
            const double* in_row = input_ptr + offset;
            double* out_row = output_ptr + offset;

            double max_val = *std::max_element(in_row, in_row + inner_size);
            
            double sum = 0.0;
            for (size_t i = 0; i < inner_size; ++i) {
                out_row[i] = std::exp(in_row[i] - max_val);
                sum += out_row[i];
            }
            
            for (size_t i = 0; i < inner_size; ++i) {
                out_row[i] /= sum;
            }
        }
    }

} // extern "C"
#endif