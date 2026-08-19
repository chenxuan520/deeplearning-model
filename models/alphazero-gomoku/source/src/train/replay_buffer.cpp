#include "train/replay_buffer.h"

#include <cstdio>

namespace az {

void ReplayBuffer::Push(const Sample &sample) {
  std::lock_guard<std::mutex> lock(mutex_);
  data_[write_pos_] = sample;
  write_pos_ = (write_pos_ + 1) % capacity_;
  if (size_ < capacity_) {
    ++size_;
  }
}

void ReplayBuffer::SampleIndices(int count, std::mt19937 &rng,
                                 std::vector<int> &indices) const {
  indices.clear();
  if (size_ == 0) {
    return;
  }
  std::uniform_int_distribution<int> dist(0, static_cast<int>(size_) - 1);
  indices.reserve(count);
  for (int i = 0; i < count; ++i) {
    indices.push_back(dist(rng));
  }
}

void ReplayBuffer::SampleIndicesRecency(int count, std::mt19937 &rng,
                                        double p_recent, double mean_age,
                                        std::vector<int> &indices) const {
  indices.clear();
  std::lock_guard<std::mutex> lock(mutex_);
  if (size_ == 0) {
    return;
  }
  std::uniform_real_distribution<double> roll(0.0, 1.0);
  std::uniform_int_distribution<int> any_index(0, static_cast<int>(size_) - 1);
  std::exponential_distribution<double> expo(1.0 / mean_age);
  // newest physical index = (write_pos_ - 1 + capacity_) % capacity_
  const int newest = (static_cast<int>(write_pos_) +
                      static_cast<int>(capacity_) - 1) %
                     static_cast<int>(capacity_);
  indices.reserve(count);
  for (int i = 0; i < count; ++i) {
    if (roll(rng) >= p_recent) {
      indices.push_back(any_index(rng));
    } else {
      int age = static_cast<int>(expo(rng));
      if (age > static_cast<int>(size_) - 1) {
        age = static_cast<int>(size_) - 1;
      }
      int idx = newest - age;
      idx %= static_cast<int>(capacity_);
      if (idx < 0) {
        idx += static_cast<int>(capacity_);
      }
      if (idx >= static_cast<int>(size_)) {
        idx = any_index(rng); // ring not full yet: fall back to uniform
      }
      indices.push_back(idx);
    }
  }
}

std::size_t ReplayBuffer::Size() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return size_;
}

bool ReplayBuffer::Save(const std::string &file) const {
  std::lock_guard<std::mutex> lock(mutex_);
  FILE *out = std::fopen(file.c_str(), "wb");
  if (out == nullptr) {
    return false;
  }
  const std::size_t sample_size = sizeof(Sample);
  bool ok = std::fwrite(&size_, sizeof(size_), 1, out) == 1 &&
            std::fwrite(&write_pos_, sizeof(write_pos_), 1, out) == 1 &&
            std::fwrite(data_.data(), sample_size, size_, out) == size_ &&
            std::fflush(out) == 0;
  if (std::fclose(out) != 0) ok = false;
  return ok;
}

bool ReplayBuffer::Load(const std::string &file) {
  std::lock_guard<std::mutex> lock(mutex_);
  FILE *in = std::fopen(file.c_str(), "rb");
  if (in == nullptr) {
    return false;
  }
  const std::size_t sample_size = sizeof(Sample);
  std::size_t loaded_size = 0;
  std::size_t loaded_write_pos = 0;
  if (std::fread(&loaded_size, sizeof(loaded_size), 1, in) != 1 ||
      std::fread(&loaded_write_pos, sizeof(loaded_write_pos), 1, in) != 1 ||
      loaded_size > capacity_ || loaded_write_pos >= capacity_) {
    std::fclose(in);
    return false;
  }
  bool ok = std::fread(data_.data(), sample_size, loaded_size, in) ==
            loaded_size;
  std::fclose(in);
  if (!ok) {
    // Do not leave partially read records addressable. TryResume rejects the
    // checkpoint, but the buffer itself remains in a safe empty state without
    // allocating a second ~859 MiB default-capacity array.
    size_ = 0;
    write_pos_ = 0;
    return false;
  }
  size_ = loaded_size;
  write_pos_ = loaded_write_pos;
  return ok;
}

} // namespace az
