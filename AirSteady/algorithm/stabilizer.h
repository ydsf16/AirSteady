#pragma once

#include <atomic>
#include <chrono>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include <Eigen/Core>
#include <Eigen/Sparse>

#include "common/types.h"

namespace airsteady {

// ---------------- Debug info ----------------
struct ResidualStats {
  int count = 0;
  double mean = 0.0;
  double rms = 0.0;
  double p95 = 0.0;
  double max = 0.0;
};

struct StabilizerDebugInfo {
  bool enabled = true;
  std::string output_dir;
  std::string csv_path;
  std::string svg_path;

  ResidualStats step1_global;  // detector constraint residuals
  ResidualStats step1_odom;    // delta constraint residuals
  ResidualStats step2_data;    // smooth vs object residuals

  double total_rms = 0.0;      // RMS of |delta(object-target)|
};

// ---------------- Config ----------------
struct StabilizerConfig {
  // Debug outputs.
  bool debug = true;
  std::string debug_output_dir = "airsteady_stabilizer_debug";

  // Detailed timing logs.
  bool timing_log = true;
  int timing_log_every_irls_iter = 1;  // 1 = every iter

  // Step-1 (object centers) base sigmas (pixel-domain).
  double global_sigma_px = 1000000000.0;            // when global_center_valid
  double weak_center_sigma_px = 20000000.0;     // when global invalid, weak pull to image center
  double odom_sigma_px = 1.0;              // when delta_valid but delta_noise absent
  double odom_invalid_sigma_px = 80.0;     // when delta invalid, weak equality

  // Step-2 (smooth centers) base sigmas.
  double smooth_vel_sigma_px = 10.0;       // velocity smoothness
  double smooth_acc_sigma_px = 10.0;       // acceleration smoothness
  double smooth_data_sigma_px = 3.0;       // data term when smooth_factor ~ 0
  double smooth_data_sigma_min_px = 50.0;  // minimal data influence when smooth_factor ~ 1

  // Robust / IRLS.
  bool robust_enable = false;
  int irls_max_iters = 100;
  double irls_eps_px = 1e-8;               // early stop threshold (avg delta change)
  double huber_delta_px = 6.0;             // Huber threshold in pixels

  // Numerical safety.
  double min_sigma_px = 1e-3;

  // Special case: smooth_factor==1 hard mode.
  double smooth_factor_one_eps = 1e-6;     // if smooth_factor >= 1 - eps => hard mode

  double default_smooth_factor = 1.0;      // [0,1]
};

// Minimal scope timer used for timing logs.
class ScopedTimer {
 public:
  ScopedTimer(const char* name, bool enabled);
  ~ScopedTimer();

  ScopedTimer(const ScopedTimer&) = delete;
  ScopedTimer& operator=(const ScopedTimer&) = delete;

  double ElapsedMs() const;
  void Stop();

 private:
  const char* name_ = "";
  bool enabled_ = false;
  bool stopped_ = false;
  std::chrono::steady_clock::time_point start_;
};

class Stabilizer {
 public:
  using FinishedCallback = std::function<void()>;

  Stabilizer();
  explicit Stabilizer(const StabilizerConfig& config);
  ~Stabilizer();

  // Preferred API.
  void StartStabilize(const std::vector<FrameTrackingResult>& track_ress,
                      const VideoInfo& video_info,
                      double smooth_factor);

  // Backward compatible to your current signature.
  void StartStablilize(const std::vector<FrameTrackingResult>& track_ress,
                       const VideoInfo& video_info) {
    StartStabilize(track_ress, video_info, config_.default_smooth_factor);
  }

  bool Finished() const { return finished_.load(); }
  bool Running() const { return running_.load(); }

  void Stop();

  std::vector<FrameStableResult> GetStabilizedResults() const;

  void AddFinishedCallback(FinishedCallback cb);

  StabilizerDebugInfo GetDebugInfo() const;

 private:
  void WorkerMain(double smooth_factor);

  // Step 1.
  std::vector<Eigen::Vector2d> ComputeObjectCentersIRLS(StabilizerDebugInfo* dbg) const;

  // Step 2.
  std::vector<Eigen::Vector2d> ComputeSmoothCentersIRLS(
      const std::vector<Eigen::Vector2d>& object_centers,
      double smooth_factor,
      StabilizerDebugInfo* dbg) const;

  // Step 3 (delta = object - target).
  std::vector<FrameStableResult> ComputeStabilizedResults(
      const std::vector<Eigen::Vector2d>& object_centers,
      const std::vector<Eigen::Vector2d>& target_centers) const;

  // Debug artifacts.
  void EmitDebugArtifacts(const std::vector<Eigen::Vector2d>& object_centers,
                          const std::vector<Eigen::Vector2d>& target_centers,
                          const std::vector<FrameStableResult>& stable,
                          StabilizerDebugInfo* dbg) const;

  // Helpers: residual statistics.
  static ResidualStats ComputeResidualStats(const std::vector<double>& norms);

  // Helpers: robust weights.
  double ClampSigma(double sigma) const;
  double WeightFromSigma(double sigma) const;
  double HuberWeight(double r_norm) const;  // returns in (0,1]

  // Linear solver helper (per axis).
  bool SolveSparseSPD(int n,
                      const std::vector<Eigen::Triplet<double>>& H_triplets,
                      const Eigen::VectorXd& g,
                      const char* tag,
                      Eigen::VectorXd* x_out,
                      std::string* err) const;

 private:
  StabilizerConfig config_;

  // Inputs
  std::vector<FrameTrackingResult> track_ress_;
  VideoInfo video_info_{};

  // Outputs
  mutable std::mutex mutex_;
  std::vector<FrameStableResult> stable_ress_;
  StabilizerDebugInfo debug_info_;

  // Callbacks
  std::vector<FinishedCallback> finished_cbs_;

  // Thread state
  std::shared_ptr<std::thread> worker_;
  std::atomic<bool> stop_{false};
  std::atomic<bool> running_{false};
  std::atomic<bool> finished_{false};
};

}  // namespace airsteady
