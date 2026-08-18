#include "cnn/policy_value_resnet.h"

#include <algorithm>
#include <cstdint>
#include <cmath>
#include <fstream>

namespace deeplearning {
namespace {

template <typename Parameterized>
std::size_t ConvParameterCount(const Parameterized &layer) {
  return layer.weight().size() + layer.bias().size();
}

std::size_t NormParameterCount(const BatchNorm2D &layer) {
  return layer.scale().size() + layer.bias().size();
}

std::size_t LinearParameterCount(const FloatLinear &layer) {
  return layer.weight().size() + layer.bias().size();
}

template <typename Value>
bool WriteValue(std::ofstream &output, const Value &value) {
  output.write(reinterpret_cast<const char *>(&value), sizeof(Value));
  return output.good();
}

template <typename Value>
bool ReadValue(std::ifstream &input, Value &value) {
  input.read(reinterpret_cast<char *>(&value), sizeof(Value));
  return input.good();
}

bool WriteVector(std::ofstream &output, const std::vector<float> &values) {
  const std::uint64_t size = values.size();
  if (!WriteValue(output, size)) {
    return false;
  }
  output.write(reinterpret_cast<const char *>(values.data()),
               static_cast<std::streamsize>(size * sizeof(float)));
  return output.good();
}

bool ReadVector(std::ifstream &input, std::size_t expected_size,
                std::vector<float> &values) {
  std::uint64_t size = 0;
  if (!ReadValue(input, size) || size != expected_size) {
    return false;
  }
  values.resize(static_cast<std::size_t>(size));
  input.read(reinterpret_cast<char *>(values.data()),
             static_cast<std::streamsize>(size * sizeof(float)));
  return input.good();
}

bool WriteConvolution(std::ofstream &output, const BatchedConv2D &layer) {
  return WriteVector(output, layer.weight()) &&
         WriteVector(output, layer.bias());
}

bool ReadConvolution(std::ifstream &input, BatchedConv2D &layer) {
  std::vector<float> weight;
  std::vector<float> bias;
  return ReadVector(input, layer.weight().size(), weight) &&
         ReadVector(input, layer.bias().size(), bias) &&
         layer.set_weight(weight) == BatchedConv2D::SUCCESS &&
         layer.set_bias(bias) == BatchedConv2D::SUCCESS;
}

bool WriteNormalization(std::ofstream &output, const BatchNorm2D &layer) {
  return WriteVector(output, layer.scale()) &&
         WriteVector(output, layer.bias()) &&
         WriteVector(output, layer.running_mean()) &&
         WriteVector(output, layer.running_variance());
}

bool ReadNormalization(std::ifstream &input, BatchNorm2D &layer) {
  std::vector<float> scale;
  std::vector<float> bias;
  std::vector<float> mean;
  std::vector<float> variance;
  return ReadVector(input, layer.scale().size(), scale) &&
         ReadVector(input, layer.bias().size(), bias) &&
         ReadVector(input, layer.running_mean().size(), mean) &&
         ReadVector(input, layer.running_variance().size(), variance) &&
         layer.set_scale(scale) == BatchNorm2D::SUCCESS &&
         layer.set_bias(bias) == BatchNorm2D::SUCCESS &&
         layer.set_running_mean(mean) == BatchNorm2D::SUCCESS &&
         layer.set_running_variance(variance) == BatchNorm2D::SUCCESS;
}

bool WriteLinear(std::ofstream &output, const FloatLinear &layer) {
  return WriteVector(output, layer.weight()) &&
         WriteVector(output, layer.bias());
}

bool ReadLinear(std::ifstream &input, FloatLinear &layer) {
  std::vector<float> weight;
  std::vector<float> bias;
  return ReadVector(input, layer.weight().size(), weight) &&
         ReadVector(input, layer.bias().size(), bias) &&
         layer.set_weight(weight) == FloatLinear::SUCCESS &&
         layer.set_bias(bias) == FloatLinear::SUCCESS;
}

bool SameStructure(const PolicyValueResNet::Config &left,
                   const PolicyValueResNet::Config &right) {
  return left.input_channels_ == right.input_channels_ &&
         left.board_height_ == right.board_height_ &&
         left.board_width_ == right.board_width_ &&
         left.trunk_channels_ == right.trunk_channels_ &&
         left.residual_block_num_ == right.residual_block_num_ &&
         left.policy_channels_ == right.policy_channels_ &&
         left.policy_size_ == right.policy_size_ &&
         left.value_channels_ == right.value_channels_ &&
         left.value_hidden_dim_ == right.value_hidden_dim_;
}

} // namespace

PolicyValueResNet::RC PolicyValueResNet::Init(const Config &config) {
  if (is_init_) {
    err_msg_ = "[PolicyValueResNet::Init] already initialized";
    return ALREADY_INIT;
  }
  if (config.input_channels_ <= 0 || config.board_height_ <= 0 ||
      config.board_width_ <= 0 || config.trunk_channels_ <= 0 ||
      config.residual_block_num_ < 0 || config.policy_channels_ <= 0 ||
      config.policy_size_ <= 0 || config.value_channels_ <= 0 ||
      config.value_hidden_dim_ <= 0 || config.thread_num_ <= 0) {
    err_msg_ = "[PolicyValueResNet::Init] invalid config";
    return INVALID_DATA;
  }
  config_ = config;

  BatchedConv2D::Config convolution;
  convolution.input_channels_ = config.input_channels_;
  convolution.output_channels_ = config.trunk_channels_;
  convolution.kernel_height_ = 3;
  convolution.kernel_width_ = 3;
  convolution.padding_ = 1;
  convolution.use_bias_ = false;
  convolution.thread_num_ = config.thread_num_;
  convolution.rand_seed_ = config.rand_seed_;
  if (stem_conv_.Init(convolution) != BatchedConv2D::SUCCESS) {
    err_msg_ = stem_conv_.err_msg();
    return INVALID_DATA;
  }

  BatchNorm2D::Config normalization;
  normalization.channels_ = config.trunk_channels_;
  normalization.epsilon_ = config.batch_norm_epsilon_;
  normalization.momentum_ = config.batch_norm_momentum_;
  normalization.thread_num_ = config.thread_num_;
  if (stem_norm_.Init(normalization) != BatchNorm2D::SUCCESS) {
    err_msg_ = stem_norm_.err_msg();
    return INVALID_DATA;
  }

  blocks_.reserve(config.residual_block_num_);
  for (int index = 0; index < config.residual_block_num_; index++) {
    ResidualBlock2D block;
    ResidualBlock2D::Config block_config;
    block_config.channels_ = config.trunk_channels_;
    block_config.height_ = config.board_height_;
    block_config.width_ = config.board_width_;
    block_config.thread_num_ = config.thread_num_;
    block_config.rand_seed_ = config.rand_seed_ + 100 + index * 2;
    block_config.batch_norm_epsilon_ = config.batch_norm_epsilon_;
    block_config.batch_norm_momentum_ = config.batch_norm_momentum_;
    if (block.Init(block_config) != ResidualBlock2D::SUCCESS) {
      err_msg_ = block.err_msg();
      return INVALID_DATA;
    }
    blocks_.push_back(std::move(block));
  }

  convolution.input_channels_ = config.trunk_channels_;
  convolution.output_channels_ = config.policy_channels_;
  convolution.kernel_height_ = 1;
  convolution.kernel_width_ = 1;
  convolution.padding_ = 0;
  convolution.rand_seed_ = config.rand_seed_ + 1000;
  if (policy_conv_.Init(convolution) != BatchedConv2D::SUCCESS) {
    err_msg_ = policy_conv_.err_msg();
    return INVALID_DATA;
  }
  normalization.channels_ = config.policy_channels_;
  if (policy_norm_.Init(normalization) != BatchNorm2D::SUCCESS) {
    err_msg_ = policy_norm_.err_msg();
    return INVALID_DATA;
  }
  FloatLinear::Config linear;
  linear.input_dim_ =
      config.policy_channels_ * config.board_height_ * config.board_width_;
  linear.output_dim_ = config.policy_size_;
  linear.thread_num_ = config.thread_num_;
  linear.rand_seed_ = config.rand_seed_ + 1001;
  if (policy_linear_.Init(linear) != FloatLinear::SUCCESS) {
    err_msg_ = policy_linear_.err_msg();
    return INVALID_DATA;
  }

  convolution.output_channels_ = config.value_channels_;
  convolution.rand_seed_ = config.rand_seed_ + 2000;
  if (value_conv_.Init(convolution) != BatchedConv2D::SUCCESS) {
    err_msg_ = value_conv_.err_msg();
    return INVALID_DATA;
  }
  normalization.channels_ = config.value_channels_;
  if (value_norm_.Init(normalization) != BatchNorm2D::SUCCESS) {
    err_msg_ = value_norm_.err_msg();
    return INVALID_DATA;
  }
  linear.input_dim_ =
      config.value_channels_ * config.board_height_ * config.board_width_;
  linear.output_dim_ = config.value_hidden_dim_;
  linear.rand_seed_ = config.rand_seed_ + 2001;
  if (value_hidden_.Init(linear) != FloatLinear::SUCCESS) {
    err_msg_ = value_hidden_.err_msg();
    return INVALID_DATA;
  }
  linear.input_dim_ = config.value_hidden_dim_;
  linear.output_dim_ = 1;
  linear.rand_seed_ = config.rand_seed_ + 2002;
  if (value_output_.Init(linear) != FloatLinear::SUCCESS) {
    err_msg_ = value_output_.err_msg();
    return INVALID_DATA;
  }
  is_init_ = true;
  return SUCCESS;
}

bool PolicyValueResNet::ValidateInput(const FloatTensor4D &input,
                                      const char *function) {
  if (!is_init_) {
    err_msg_ = std::string("[PolicyValueResNet::") + function +
               "] not initialized";
    return false;
  }
  if (input.batch() <= 0 || input.channels() != config_.input_channels_ ||
      input.height() != config_.board_height_ ||
      input.width() != config_.board_width_) {
    err_msg_ = std::string("[PolicyValueResNet::") + function +
               "] invalid input shape";
    return false;
  }
  return true;
}

void PolicyValueResNet::ApplyRelu(FloatTensor4D &tensor) {
  for (float &value : tensor.values()) {
    value = std::max(0.0f, value);
  }
}

std::vector<float>
PolicyValueResNet::FlattenNCHW(const FloatTensor4D &tensor) {
  return tensor.values();
}

FloatTensor4D PolicyValueResNet::UnflattenNCHW(
    const std::vector<float> &values, int batch, int channels, int height,
    int width) {
  FloatTensor4D tensor(batch, channels, height, width);
  tensor.values() = values;
  return tensor;
}

PolicyValueResNet::RC PolicyValueResNet::Forward(
    const FloatTensor4D &input, Output &output, bool training,
    bool use_running_statistics) {
  if (!ValidateInput(input, "Forward")) {
    return is_init_ ? INVALID_DATA : NOT_INIT;
  }
  FloatTensor4D trunk;
  FloatTensor4D normalized;
  if (training) {
    if (stem_conv_.Forward(input, trunk, true) != BatchedConv2D::SUCCESS) {
      err_msg_ = stem_conv_.err_msg();
      return INVALID_DATA;
    }
    if (stem_norm_.Forward(trunk, normalized, true,
                           use_running_statistics) !=
        BatchNorm2D::SUCCESS) {
      err_msg_ = stem_norm_.err_msg();
      return INVALID_DATA;
    }
  } else {
    std::vector<float> scale, bias;
    stem_norm_.InferenceAffine(scale, bias);
    if (stem_conv_.ForwardAffine(input, normalized, scale, bias) !=
        BatchedConv2D::SUCCESS) {
      err_msg_ = stem_conv_.err_msg();
      return INVALID_DATA;
    }
  }
  ApplyRelu(normalized);
  if (training) {
    stem_relu_cache_ = normalized;
  }
  trunk = std::move(normalized);
  for (ResidualBlock2D &block : blocks_) {
    FloatTensor4D next;
    if (block.Forward(trunk, next, training, use_running_statistics) !=
        ResidualBlock2D::SUCCESS) {
      err_msg_ = block.err_msg();
      return INVALID_DATA;
    }
    trunk = std::move(next);
  }

  FloatTensor4D policy_tensor;
  FloatTensor4D policy_normalized;
  if (training) {
    if (policy_conv_.Forward(trunk, policy_tensor, true) !=
        BatchedConv2D::SUCCESS) {
      err_msg_ = policy_conv_.err_msg();
      return INVALID_DATA;
    }
    if (policy_norm_.Forward(policy_tensor, policy_normalized, true,
                             use_running_statistics) !=
        BatchNorm2D::SUCCESS) {
      err_msg_ = policy_norm_.err_msg();
      return INVALID_DATA;
    }
  } else {
    std::vector<float> scale, bias;
    policy_norm_.InferenceAffine(scale, bias);
    if (policy_conv_.ForwardAffine(trunk, policy_normalized, scale, bias) !=
        BatchedConv2D::SUCCESS) {
      err_msg_ = policy_conv_.err_msg();
      return INVALID_DATA;
    }
  }
  ApplyRelu(policy_normalized);
  if (training) {
    policy_relu_cache_ = policy_normalized;
  }
  if (policy_linear_.Forward(FlattenNCHW(policy_normalized), input.batch(),
                             output.policy_logits_, training) !=
      FloatLinear::SUCCESS) {
    err_msg_ = policy_linear_.err_msg();
    return INVALID_DATA;
  }

  FloatTensor4D value_tensor;
  FloatTensor4D value_normalized;
  if (training) {
    if (value_conv_.Forward(trunk, value_tensor, true) !=
        BatchedConv2D::SUCCESS) {
      err_msg_ = value_conv_.err_msg();
      return INVALID_DATA;
    }
    if (value_norm_.Forward(value_tensor, value_normalized, true,
                            use_running_statistics) !=
        BatchNorm2D::SUCCESS) {
      err_msg_ = value_norm_.err_msg();
      return INVALID_DATA;
    }
  } else {
    std::vector<float> scale, bias;
    value_norm_.InferenceAffine(scale, bias);
    if (value_conv_.ForwardAffine(trunk, value_normalized, scale, bias) !=
        BatchedConv2D::SUCCESS) {
      err_msg_ = value_conv_.err_msg();
      return INVALID_DATA;
    }
  }
  ApplyRelu(value_normalized);
  if (training) {
    value_relu_cache_ = value_normalized;
  }
  std::vector<float> hidden;
  if (value_hidden_.Forward(FlattenNCHW(value_normalized), input.batch(),
                            hidden, training) != FloatLinear::SUCCESS) {
    err_msg_ = value_hidden_.err_msg();
    return INVALID_DATA;
  }
  for (float &value : hidden) {
    value = std::max(0.0f, value);
  }
  if (training) {
    value_hidden_relu_cache_ = hidden;
  }
  std::vector<float> raw_values;
  if (value_output_.Forward(hidden, input.batch(), raw_values, training) !=
      FloatLinear::SUCCESS) {
    err_msg_ = value_output_.err_msg();
    return INVALID_DATA;
  }
  output.values_.resize(input.batch());
  for (int sample = 0; sample < input.batch(); sample++) {
    output.values_[sample] = std::tanh(raw_values[sample]);
  }
  output.batch_ = input.batch();
  if (training) {
    value_cache_ = output.values_;
    cached_batch_ = input.batch();
    has_cache_ = true;
  } else {
    has_cache_ = false;
    cached_batch_ = 0;
  }
  return SUCCESS;
}

PolicyValueResNet::RC PolicyValueResNet::Backward(
    const std::vector<float> &grad_policy_logits,
    const std::vector<float> &grad_values, FloatTensor4D &grad_input) {
  if (!is_init_) {
    err_msg_ = "[PolicyValueResNet::Backward] not initialized";
    return NOT_INIT;
  }
  if (!has_cache_) {
    err_msg_ = "[PolicyValueResNet::Backward] missing training cache";
    return MISSING_CACHE;
  }
  if (grad_policy_logits.size() !=
          static_cast<std::size_t>(cached_batch_) * config_.policy_size_ ||
      grad_values.size() != static_cast<std::size_t>(cached_batch_)) {
    err_msg_ = "[PolicyValueResNet::Backward] invalid gradient shape";
    return INVALID_DATA;
  }

  std::vector<float> grad_policy_flat;
  if (policy_linear_.Backward(grad_policy_logits, grad_policy_flat) !=
      FloatLinear::SUCCESS) {
    err_msg_ = policy_linear_.err_msg();
    return INVALID_DATA;
  }
  FloatTensor4D grad_policy = UnflattenNCHW(
      grad_policy_flat, cached_batch_, config_.policy_channels_,
      config_.board_height_, config_.board_width_);
  for (std::size_t index = 0; index < grad_policy.size(); index++) {
    if (policy_relu_cache_.values()[index] <= 0.0f) {
      grad_policy.values()[index] = 0.0f;
    }
  }
  FloatTensor4D grad_policy_conv;
  if (policy_norm_.Backward(grad_policy, grad_policy_conv) !=
      BatchNorm2D::SUCCESS) {
    err_msg_ = policy_norm_.err_msg();
    return INVALID_DATA;
  }
  FloatTensor4D grad_trunk_policy;
  if (policy_conv_.Backward(grad_policy_conv, grad_trunk_policy) !=
      BatchedConv2D::SUCCESS) {
    err_msg_ = policy_conv_.err_msg();
    return INVALID_DATA;
  }

  std::vector<float> grad_raw_value(cached_batch_);
  for (int sample = 0; sample < cached_batch_; sample++) {
    grad_raw_value[sample] =
        grad_values[sample] *
        (1.0f - value_cache_[sample] * value_cache_[sample]);
  }
  std::vector<float> grad_hidden;
  if (value_output_.Backward(grad_raw_value, grad_hidden) !=
      FloatLinear::SUCCESS) {
    err_msg_ = value_output_.err_msg();
    return INVALID_DATA;
  }
  for (std::size_t index = 0; index < grad_hidden.size(); index++) {
    if (value_hidden_relu_cache_[index] <= 0.0f) {
      grad_hidden[index] = 0.0f;
    }
  }
  std::vector<float> grad_value_flat;
  if (value_hidden_.Backward(grad_hidden, grad_value_flat) !=
      FloatLinear::SUCCESS) {
    err_msg_ = value_hidden_.err_msg();
    return INVALID_DATA;
  }
  FloatTensor4D grad_value = UnflattenNCHW(
      grad_value_flat, cached_batch_, config_.value_channels_,
      config_.board_height_, config_.board_width_);
  for (std::size_t index = 0; index < grad_value.size(); index++) {
    if (value_relu_cache_.values()[index] <= 0.0f) {
      grad_value.values()[index] = 0.0f;
    }
  }
  FloatTensor4D grad_value_conv;
  if (value_norm_.Backward(grad_value, grad_value_conv) !=
      BatchNorm2D::SUCCESS) {
    err_msg_ = value_norm_.err_msg();
    return INVALID_DATA;
  }
  FloatTensor4D grad_trunk_value;
  if (value_conv_.Backward(grad_value_conv, grad_trunk_value) !=
      BatchedConv2D::SUCCESS) {
    err_msg_ = value_conv_.err_msg();
    return INVALID_DATA;
  }

  FloatTensor4D grad_trunk = grad_trunk_policy;
  for (std::size_t index = 0; index < grad_trunk.size(); index++) {
    grad_trunk.values()[index] += grad_trunk_value.values()[index];
  }
  for (auto block = blocks_.rbegin(); block != blocks_.rend(); ++block) {
    FloatTensor4D previous;
    if (block->Backward(grad_trunk, previous) != ResidualBlock2D::SUCCESS) {
      err_msg_ = block->err_msg();
      return INVALID_DATA;
    }
    grad_trunk = std::move(previous);
  }
  for (std::size_t index = 0; index < grad_trunk.size(); index++) {
    if (stem_relu_cache_.values()[index] <= 0.0f) {
      grad_trunk.values()[index] = 0.0f;
    }
  }
  FloatTensor4D grad_stem;
  if (stem_norm_.Backward(grad_trunk, grad_stem) !=
      BatchNorm2D::SUCCESS) {
    err_msg_ = stem_norm_.err_msg();
    return INVALID_DATA;
  }
  if (stem_conv_.Backward(grad_stem, grad_input) !=
      BatchedConv2D::SUCCESS) {
    err_msg_ = stem_conv_.err_msg();
    return INVALID_DATA;
  }
  return SUCCESS;
}

void PolicyValueResNet::ZeroGrad() {
  stem_conv_.ZeroGrad();
  stem_norm_.ZeroGrad();
  for (ResidualBlock2D &block : blocks_) {
    block.ZeroGrad();
  }
  policy_conv_.ZeroGrad();
  policy_norm_.ZeroGrad();
  policy_linear_.ZeroGrad();
  value_conv_.ZeroGrad();
  value_norm_.ZeroGrad();
  value_hidden_.ZeroGrad();
  value_output_.ZeroGrad();
}

std::vector<PolicyValueResNet::TrainableParameter>
PolicyValueResNet::TrainableParameters() {
  std::vector<TrainableParameter> parameters;
  auto add_convolution =
      [&](const std::string &prefix, BatchedConv2D &convolution) {
        parameters.push_back({
            prefix + ".weight",
            &convolution.mutable_weight(),
            &convolution.grad_weight(),
            true,
        });
        if (convolution.config().use_bias_) {
          parameters.push_back({
              prefix + ".bias",
              &convolution.mutable_bias(),
              &convolution.grad_bias(),
              false,
          });
        }
      };
  auto add_normalization =
      [&](const std::string &prefix, BatchNorm2D &normalization) {
        parameters.push_back({
            prefix + ".scale",
            &normalization.mutable_scale(),
            &normalization.grad_scale(),
            false,
        });
        parameters.push_back({
            prefix + ".bias",
            &normalization.mutable_bias(),
            &normalization.grad_bias(),
            false,
        });
      };
  auto add_linear = [&](const std::string &prefix, FloatLinear &linear) {
    parameters.push_back({
        prefix + ".weight",
        &linear.mutable_weight(),
        &linear.grad_weight(),
        true,
    });
    if (linear.config().use_bias_) {
      parameters.push_back({
          prefix + ".bias",
          &linear.mutable_bias(),
          &linear.grad_bias(),
          false,
      });
    }
  };

  add_convolution("stem.conv", stem_conv_);
  add_normalization("stem.norm", stem_norm_);
  for (int index = 0; index < static_cast<int>(blocks_.size()); index++) {
    const std::string prefix = "block." + std::to_string(index);
    add_convolution(prefix + ".conv1", blocks_[index].conv1());
    add_normalization(prefix + ".norm1", blocks_[index].norm1());
    add_convolution(prefix + ".conv2", blocks_[index].conv2());
    add_normalization(prefix + ".norm2", blocks_[index].norm2());
  }
  add_convolution("policy.conv", policy_conv_);
  add_normalization("policy.norm", policy_norm_);
  add_linear("policy.fc", policy_linear_);
  add_convolution("value.conv", value_conv_);
  add_normalization("value.norm", value_norm_);
  add_linear("value.hidden", value_hidden_);
  add_linear("value.output", value_output_);
  return parameters;
}

PolicyValueResNet::RC PolicyValueResNet::Save(const std::string &file) const {
  if (!is_init_) {
    return NOT_INIT;
  }
  std::ofstream output(file, std::ios::binary | std::ios::trunc);
  const char magic[8] = {'X', 'Q', 'P', 'V', 'R', 'N', '0', '1'};
  output.write(magic, sizeof(magic));
  const std::uint32_t version = 1;
  if (!WriteValue(output, version) ||
      !WriteValue(output, config_.input_channels_) ||
      !WriteValue(output, config_.board_height_) ||
      !WriteValue(output, config_.board_width_) ||
      !WriteValue(output, config_.trunk_channels_) ||
      !WriteValue(output, config_.residual_block_num_) ||
      !WriteValue(output, config_.policy_channels_) ||
      !WriteValue(output, config_.policy_size_) ||
      !WriteValue(output, config_.value_channels_) ||
      !WriteValue(output, config_.value_hidden_dim_) ||
      !WriteValue(output, config_.thread_num_) ||
      !WriteValue(output, config_.rand_seed_) ||
      !WriteValue(output, config_.batch_norm_epsilon_) ||
      !WriteValue(output, config_.batch_norm_momentum_) ||
      !WriteConvolution(output, stem_conv_) ||
      !WriteNormalization(output, stem_norm_)) {
    return INVALID_DATA;
  }
  for (const ResidualBlock2D &block : blocks_) {
    if (!WriteConvolution(output, block.conv1()) ||
        !WriteNormalization(output, block.norm1()) ||
        !WriteConvolution(output, block.conv2()) ||
        !WriteNormalization(output, block.norm2())) {
      return INVALID_DATA;
    }
  }
  if (!WriteConvolution(output, policy_conv_) ||
      !WriteNormalization(output, policy_norm_) ||
      !WriteLinear(output, policy_linear_) ||
      !WriteConvolution(output, value_conv_) ||
      !WriteNormalization(output, value_norm_) ||
      !WriteLinear(output, value_hidden_) ||
      !WriteLinear(output, value_output_)) {
    return INVALID_DATA;
  }
  output.flush();
  return output.good() ? SUCCESS : INVALID_DATA;
}

PolicyValueResNet::RC PolicyValueResNet::Load(const std::string &file) {
  std::ifstream input(file, std::ios::binary);
  char magic[8] = {};
  input.read(magic, sizeof(magic));
  const char expected_magic[8] = {'X', 'Q', 'P', 'V', 'R', 'N', '0', '1'};
  if (!input.good() ||
      !std::equal(std::begin(magic), std::end(magic),
                  std::begin(expected_magic))) {
    err_msg_ = "[PolicyValueResNet::Load] invalid magic";
    return INVALID_DATA;
  }
  std::uint32_t version = 0;
  Config loaded;
  if (!ReadValue(input, version) || version != 1 ||
      !ReadValue(input, loaded.input_channels_) ||
      !ReadValue(input, loaded.board_height_) ||
      !ReadValue(input, loaded.board_width_) ||
      !ReadValue(input, loaded.trunk_channels_) ||
      !ReadValue(input, loaded.residual_block_num_) ||
      !ReadValue(input, loaded.policy_channels_) ||
      !ReadValue(input, loaded.policy_size_) ||
      !ReadValue(input, loaded.value_channels_) ||
      !ReadValue(input, loaded.value_hidden_dim_) ||
      !ReadValue(input, loaded.thread_num_) ||
      !ReadValue(input, loaded.rand_seed_) ||
      !ReadValue(input, loaded.batch_norm_epsilon_) ||
      !ReadValue(input, loaded.batch_norm_momentum_)) {
    err_msg_ = "[PolicyValueResNet::Load] truncated config";
    return INVALID_DATA;
  }
  if (!is_init_) {
    if (Init(loaded) != SUCCESS) {
      return INVALID_DATA;
    }
  } else if (!SameStructure(config_, loaded)) {
    err_msg_ = "[PolicyValueResNet::Load] model structure mismatch";
    return INVALID_DATA;
  }
  if (!ReadConvolution(input, stem_conv_) ||
      !ReadNormalization(input, stem_norm_)) {
    err_msg_ = "[PolicyValueResNet::Load] invalid stem";
    return INVALID_DATA;
  }
  for (ResidualBlock2D &block : blocks_) {
    if (!ReadConvolution(input, block.conv1()) ||
        !ReadNormalization(input, block.norm1()) ||
        !ReadConvolution(input, block.conv2()) ||
        !ReadNormalization(input, block.norm2())) {
      err_msg_ = "[PolicyValueResNet::Load] invalid residual block";
      return INVALID_DATA;
    }
  }
  if (!ReadConvolution(input, policy_conv_) ||
      !ReadNormalization(input, policy_norm_) ||
      !ReadLinear(input, policy_linear_) ||
      !ReadConvolution(input, value_conv_) ||
      !ReadNormalization(input, value_norm_) ||
      !ReadLinear(input, value_hidden_) ||
      !ReadLinear(input, value_output_)) {
    err_msg_ = "[PolicyValueResNet::Load] invalid head parameters";
    return INVALID_DATA;
  }
  char trailing = 0;
  if (input.read(&trailing, 1)) {
    err_msg_ = "[PolicyValueResNet::Load] trailing data";
    return INVALID_DATA;
  }
  return input.eof() ? SUCCESS : INVALID_DATA;
}

std::size_t PolicyValueResNet::parameter_count() const {
  if (!is_init_) {
    return 0;
  }
  std::size_t count =
      ConvParameterCount(stem_conv_) + NormParameterCount(stem_norm_);
  for (const ResidualBlock2D &block : blocks_) {
    count += ConvParameterCount(block.conv1()) +
             NormParameterCount(block.norm1()) +
             ConvParameterCount(block.conv2()) +
             NormParameterCount(block.norm2());
  }
  count += ConvParameterCount(policy_conv_) +
           NormParameterCount(policy_norm_) +
           LinearParameterCount(policy_linear_) +
           ConvParameterCount(value_conv_) +
           NormParameterCount(value_norm_) +
           LinearParameterCount(value_hidden_) +
           LinearParameterCount(value_output_);
  return count;
}

} // namespace deeplearning
