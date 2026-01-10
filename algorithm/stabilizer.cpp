#include "algorithm/stabilizer.h"

#include <algorithm>
#include <fstream>
#include <iomanip>
#include <limits>
#include <numeric>
#include <sstream>
#include <utility>

#include <Eigen/Sparse>
#include <Eigen/SparseCholesky>
#include <glog/logging.h>

#if __has_include(<filesystem>)
#include <filesystem>
namespace fs = std::filesystem;
#endif

namespace airsteady {
namespace {

inline double Clamp01(double v) {
  if (v < 0.0) return 0.0;
  if (v > 1.0) return 1.0;
  return v;
}

inline Eigen::Vector2d ImageCenterProxy(const VideoInfo& info) {
  return Eigen::Vector2d(static_cast<double>(info.proxy_width) * 0.5,
                         static_cast<double>(info.proxy_height) * 0.5);
}

inline double Percentile95(std::vector<double> v) {
  if (v.empty()) return 0.0;
  std::sort(v.begin(), v.end());
  const size_t idx = static_cast<size_t>(std::floor(0.95 * (v.size() - 1)));
  return v[idx];
}

// ---------- Simple SVG plot (no external deps) ----------
std::string MakeSvgPlot(const std::vector<double>& xs,
                        const std::vector<double>& y1,
                        const std::vector<double>& y2,
                        const std::vector<double>& y3,
                        const std::string& title,
                        int width = 1200,
                        int height = 420) {
  const int pad_l = 60;
  const int pad_r = 20;
  const int pad_t = 40;
  const int pad_b = 40;

  const int w = width - pad_l - pad_r;
  const int h = height - pad_t - pad_b;

  auto minmax = [&](const std::vector<double>& v) -> std::pair<double, double> {
    double mn = std::numeric_limits<double>::infinity();
    double mx = -std::numeric_limits<double>::infinity();
    for (double a : v) {
      if (!std::isfinite(a)) continue;
      mn = std::min(mn, a);
      mx = std::max(mx, a);
    }
    if (!std::isfinite(mn) || !std::isfinite(mx) || mn == mx) {
      mn = 0.0;
      mx = 1.0;
    }
    return {mn, mx};
  };

  std::vector<double> all = y1;
  all.insert(all.end(), y2.begin(), y2.end());
  all.insert(all.end(), y3.begin(), y3.end());
  auto [ymin, ymax] = minmax(all);

  auto x_minmax = minmax(xs);
  const double xmin = x_minmax.first;
  const double xmax = x_minmax.second;

  auto x_map = [&](double x) -> double {
    if (xmax == xmin) return pad_l;
    return pad_l + (x - xmin) / (xmax - xmin) * w;
  };
  auto y_map = [&](double y) -> double {
    return pad_t + (1.0 - (y - ymin) / (ymax - ymin)) * h;
  };

  auto polyline = [&](const std::vector<double>& yy) -> std::string {
    std::ostringstream oss;
    for (size_t i = 0; i < yy.size() && i < xs.size(); ++i) {
      const double px = x_map(xs[i]);
      const double py = y_map(yy[i]);
      oss << px << "," << py;
      if (i + 1 < yy.size() && i + 1 < xs.size()) oss << " ";
    }
    return oss.str();
  };

  std::ostringstream svg;
  svg << "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"" << width
      << "\" height=\"" << height << "\">\n";
  svg << "<rect x=\"0\" y=\"0\" width=\"" << width << "\" height=\"" << height
      << "\" fill=\"white\" />\n";
  svg << "<text x=\"" << pad_l << "\" y=\"24\" font-size=\"16\" "
      << "font-family=\"Arial\" fill=\"#111\">" << title << "</text>\n";

  // Axes
  svg << "<line x1=\"" << pad_l << "\" y1=\"" << pad_t
      << "\" x2=\"" << pad_l << "\" y2=\"" << (pad_t + h)
      << "\" stroke=\"#333\" stroke-width=\"1\" />\n";
  svg << "<line x1=\"" << pad_l << "\" y1=\"" << (pad_t + h)
      << "\" x2=\"" << (pad_l + w) << "\" y2=\"" << (pad_t + h)
      << "\" stroke=\"#333\" stroke-width=\"1\" />\n";

  // y labels
  svg << "<text x=\"10\" y=\"" << (pad_t + 10) << "\" font-size=\"12\" "
      << "font-family=\"Arial\" fill=\"#333\">" << ymax << "</text>\n";
  svg << "<text x=\"10\" y=\"" << (pad_t + h) << "\" font-size=\"12\" "
      << "font-family=\"Arial\" fill=\"#333\">" << ymin << "</text>\n";

  // Curves: object (blue), target (orange), delta (green)
  svg << "<polyline fill=\"none\" stroke=\"#1f77b4\" stroke-width=\"2\" points=\""
      << polyline(y1) << "\" />\n";
  svg << "<polyline fill=\"none\" stroke=\"#ff7f0e\" stroke-width=\"2\" points=\""
      << polyline(y2) << "\" />\n";
  svg << "<polyline fill=\"none\" stroke=\"#2ca02c\" stroke-width=\"2\" points=\""
      << polyline(y3) << "\" />\n";

  // Legend
  const int lx = pad_l + 10;
  const int ly = pad_t + 10;
  svg << "<rect x=\"" << lx << "\" y=\"" << (ly - 10) << "\" width=\"260\" height=\"56\" "
      << "fill=\"white\" stroke=\"#ddd\" />\n";
  svg << "<line x1=\"" << lx + 10 << "\" y1=\"" << ly << "\" x2=\"" << lx + 50
      << "\" y2=\"" << ly << "\" stroke=\"#1f77b4\" stroke-width=\"2\" />\n";
  svg << "<text x=\"" << lx + 60 << "\" y=\"" << (ly + 4) << "\" font-size=\"12\" "
      << "font-family=\"Arial\">object</text>\n";

  svg << "<line x1=\"" << lx + 10 << "\" y1=\"" << ly + 18 << "\" x2=\"" << lx + 50
      << "\" y2=\"" << ly + 18 << "\" stroke=\"#ff7f0e\" stroke-width=\"2\" />\n";
  svg << "<text x=\"" << lx + 60 << "\" y=\"" << (ly + 22) << "\" font-size=\"12\" "
      << "font-family=\"Arial\">target</text>\n";

  svg << "<line x1=\"" << lx + 10 << "\" y1=\"" << ly + 36 << "\" x2=\"" << lx + 50
      << "\" y2=\"" << ly + 36 << "\" stroke=\"#2ca02c\" stroke-width=\"2\" />\n";
  svg << "<text x=\"" << lx + 60 << "\" y=\"" << (ly + 40) << "\" font-size=\"12\" "
      << "font-family=\"Arial\">delta (object-target)</text>\n";

  svg << "</svg>\n";
  return svg.str();
}

std::string StripSvgContent(const std::string& s) {
  const auto p0 = s.find('\n');
  const auto p1 = s.rfind("</svg>");
  if (p0 == std::string::npos || p1 == std::string::npos || p1 <= p0) return s;
  return s.substr(p0 + 1, p1 - (p0 + 1));
}

}  // namespace

// ---------------- ScopedTimer ----------------
ScopedTimer::ScopedTimer(const char* name, bool enabled)
    : name_(name), enabled_(enabled), start_(std::chrono::steady_clock::now()) {}

ScopedTimer::~ScopedTimer() {
  Stop();
}

double ScopedTimer::ElapsedMs() const {
  const auto now = std::chrono::steady_clock::now();
  return std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(now - start_).count();
}

void ScopedTimer::Stop() {
  if (!enabled_ || stopped_) return;
  stopped_ = true;
  std::ostringstream oss;
  oss << "[Stabilizer][Timing] " << name_ << ": " << std::fixed << std::setprecision(3)
      << ElapsedMs() << " ms";
  LOG(INFO) << oss.str();
}

// ---------------- Stabilizer ----------------
Stabilizer::Stabilizer() : Stabilizer(StabilizerConfig{}) {}

Stabilizer::Stabilizer(const StabilizerConfig& config) : config_(config) {}

Stabilizer::~Stabilizer() {
  Stop();
}

void Stabilizer::AddFinishedCallback(FinishedCallback cb) {
  std::lock_guard<std::mutex> lk(mutex_);
  finished_cbs_.push_back(std::move(cb));
}

StabilizerDebugInfo Stabilizer::GetDebugInfo() const {
  std::lock_guard<std::mutex> lk(mutex_);
  return debug_info_;
}

std::vector<FrameStableResult> Stabilizer::GetStabilizedResults() const {
  std::lock_guard<std::mutex> lk(mutex_);
  return stable_ress_;
}

void Stabilizer::Stop() {
  stop_.store(true);
  if (worker_ && worker_->joinable()) {
    worker_->join();
  }
  worker_.reset();
  running_.store(false);
}

void Stabilizer::StartStabilize(const std::vector<FrameTrackingResult>& track_ress,
                                const VideoInfo& video_info,
                                double smooth_factor) {
  Stop();

  {
    std::lock_guard<std::mutex> lk(mutex_);
    track_ress_ = track_ress;
    video_info_ = video_info;
    stable_ress_.clear();

    debug_info_ = StabilizerDebugInfo{};
    debug_info_.enabled = config_.debug;
    debug_info_.output_dir = config_.debug_output_dir;
  }

  stop_.store(false);
  finished_.store(false);
  running_.store(true);

  const double sf = Clamp01(smooth_factor);
  worker_ = std::make_shared<std::thread>(&Stabilizer::WorkerMain, this, sf);
}

void Stabilizer::WorkerMain(double smooth_factor) {
  ScopedTimer t_total("Total", config_.timing_log);

  std::vector<FrameTrackingResult> tracks;
  VideoInfo info;
  StabilizerDebugInfo dbg;
  {
    std::lock_guard<std::mutex> lk(mutex_);
    tracks = track_ress_;
    info = video_info_;
    dbg = debug_info_;
  }

  if (config_.timing_log) {
    LOG(INFO) << "[Stabilizer] start: frames=" << tracks.size()
              << ", proxy=" << info.proxy_width << "x" << info.proxy_height
              << ", smooth_factor=" << std::fixed << std::setprecision(3) << smooth_factor
              << ", robust=" << static_cast<int>(config_.robust_enable)
              << ", huber=" << std::fixed << std::setprecision(2) << config_.huber_delta_px << "px"
              << ", irls=" << config_.irls_max_iters
              << ", debug=" << static_cast<int>(dbg.enabled);
  }

  if (tracks.empty() || info.proxy_width <= 0 || info.proxy_height <= 0) {
    finished_.store(true);
    running_.store(false);

    std::vector<FinishedCallback> cbs;
    {
      std::lock_guard<std::mutex> lk(mutex_);
      cbs = finished_cbs_;
      debug_info_ = dbg;
    }
    for (auto& cb : cbs) if (cb) cb();
    return;
  }

  // Bind to members for compute functions.
  {
    std::lock_guard<std::mutex> lk(mutex_);
    track_ress_ = tracks;
    video_info_ = info;
  }

  if (stop_.load()) {
    finished_.store(true);
    running_.store(false);
    return;
  }

  std::vector<Eigen::Vector2d> object_centers;
  {
    ScopedTimer t("Step1-ObjectCenters(IRLS)", config_.timing_log);
    object_centers = ComputeObjectCentersIRLS(&dbg);
  }

  if (stop_.load()) {
    finished_.store(true);
    running_.store(false);
    return;
  }

  const double sf = Clamp01(smooth_factor);
  bool hard_mode = (sf >= 1.0 - config_.smooth_factor_one_eps);
  if (config_.timing_log) {
    LOG(INFO) << "[Stabilizer] hard_mode=" << static_cast<int>(hard_mode) << " (sf>=1-eps)";
  }

  std::vector<Eigen::Vector2d> target_centers;
  hard_mode = true; // FAKE.
  if (hard_mode) {
    ScopedTimer t("Step2-Skipped(HardMode->ImageCenterTarget)", config_.timing_log);
    const Eigen::Vector2d c = ImageCenterProxy(info);
    target_centers.assign(object_centers.size(), c);
  } else {
    ScopedTimer t("Step2-SmoothCenters(IRLS)", config_.timing_log);
    target_centers = ComputeSmoothCentersIRLS(object_centers, sf, &dbg);
  }

  if (stop_.load()) {
    finished_.store(true);
    running_.store(false);
    return;
  }

  std::vector<FrameStableResult> stable;
  {
    ScopedTimer t("Step3-StableDeltas(object-target)", config_.timing_log);
    stable = ComputeStabilizedResults(object_centers, target_centers);
  }

  if (dbg.enabled && !stop_.load()) {
    ScopedTimer t("DebugArtifacts(CSV+SVG)", config_.timing_log);
    EmitDebugArtifacts(object_centers, target_centers, stable, &dbg);
  }

  {
    std::lock_guard<std::mutex> lk(mutex_);
    stable_ress_ = stable;
    debug_info_ = dbg;
  }

  if (config_.timing_log) {
    std::ostringstream oss;
    oss << "[Stabilizer] done: total_rms=" << std::fixed << std::setprecision(3) << dbg.total_rms << "px, "
        << "step1_global(rms=" << dbg.step1_global.rms << " p95=" << dbg.step1_global.p95
        << " max=" << dbg.step1_global.max << " n=" << dbg.step1_global.count << "), "
        << "step1_odom(rms=" << dbg.step1_odom.rms << " p95=" << dbg.step1_odom.p95
        << " max=" << dbg.step1_odom.max << " n=" << dbg.step1_odom.count << "), "
        << "step2_data(rms=" << dbg.step2_data.rms << " p95=" << dbg.step2_data.p95
        << " max=" << dbg.step2_data.max << " n=" << dbg.step2_data.count << "), "
        << "csv=" << dbg.csv_path << ", svg=" << dbg.svg_path;
    LOG(INFO) << oss.str();
  }

  finished_.store(true);
  running_.store(false);

  std::vector<FinishedCallback> cbs;
  {
    std::lock_guard<std::mutex> lk(mutex_);
    cbs = finished_cbs_;
  }
  for (auto& cb : cbs) if (cb) cb();
}

// ---------------- helpers ----------------
double Stabilizer::ClampSigma(double sigma) const {
  if (!std::isfinite(sigma) || sigma <= 0.0) return config_.min_sigma_px;
  return std::max(sigma, config_.min_sigma_px);
}

double Stabilizer::WeightFromSigma(double sigma) const {
  const double s = ClampSigma(sigma);
  return 1.0 / (s * s);
}

double Stabilizer::HuberWeight(double r_norm) const {
  const double d = std::max(config_.huber_delta_px, 1e-6);
  if (!std::isfinite(r_norm) || r_norm <= d) return 1.0;
  return d / r_norm;  // in (0,1]
}

ResidualStats Stabilizer::ComputeResidualStats(const std::vector<double>& norms) {
  ResidualStats st;
  st.count = static_cast<int>(norms.size());
  if (norms.empty()) return st;

  double sum = 0.0;
  double sum2 = 0.0;
  double mx = 0.0;
  for (double v : norms) {
    const double a = std::isfinite(v) ? v : 0.0;
    sum += a;
    sum2 += a * a;
    mx = std::max(mx, a);
  }
  st.mean = sum / norms.size();
  st.rms = std::sqrt(sum2 / norms.size());
  st.p95 = Percentile95(norms);
  st.max = mx;
  return st;
}

bool Stabilizer::SolveSparseSPD(int n,
                                const std::vector<Eigen::Triplet<double>>& H_triplets,
                                const Eigen::VectorXd& g,
                                const char* tag,
                                Eigen::VectorXd* x_out,
                                std::string* err) const {
  if (!x_out) return false;
  if (n <= 0) {
    if (err) *err = "SolveSparseSPD: n <= 0";
    return false;
  }
  if (g.size() != n) {
    if (err) *err = "SolveSparseSPD: g size mismatch";
    return false;
  }

  Eigen::SparseMatrix<double> H(n, n);
  H.setFromTriplets(H_triplets.begin(), H_triplets.end());
  H.makeCompressed();

  if (config_.timing_log) {
    LOG(INFO) << "[Stabilizer][Solve] " << (tag ? tag : "") << ": n=" << n
              << ", nnz=" << H.nonZeros();
  }

  Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>> ldlt;
  ldlt.compute(H);
  if (ldlt.info() != Eigen::Success) {
    if (err) *err = "SolveSparseSPD: LDLT factorization failed";
    return false;
  }

  *x_out = ldlt.solve(g);
  if (ldlt.info() != Eigen::Success) {
    if (err) *err = "SolveSparseSPD: solve failed";
    return false;
  }
  return true;
}

// std::vector<Eigen::Vector2d> Stabilizer::ComputeObjectCentersIRLS(
//     StabilizerDebugInfo* dbg) const {
//   const int n = static_cast<int>(track_ress_.size());
//   std::vector<Eigen::Vector2d> p(n, Eigen::Vector2d::Zero());
//   if (n == 0) return p;

//   const Eigen::Vector2d img_center = ImageCenterProxy(video_info_);
//   const double w = static_cast<double>(video_info_.proxy_width);
//   const double h = static_cast<double>(video_info_.proxy_height);

//   auto clamp_xy = [&](Eigen::Vector2d* v) {
//     // if (!v) return;
//     // return *v;
    
//     // // clamp to [0, w-1], [0, h-1]
//     // (*v).x() = std::max(0.0, std::min((*v).x(), std::max(0.0, w - 1.0)));
//     // (*v).y() = std::max(0.0, std::min((*v).y(), std::max(0.0, h - 1.0)));
//   };

//   // 1) Find the first valid global center as anchor.
//   int anchor = -1;
//   for (int i = 0; i < n; ++i) {
//     if (track_ress_[i].global_center_valid) {
//       anchor = i;
//       break;
//     }
//   }

//   if (anchor < 0) {
//     // No valid global center at all: fallback to image center + pure odom integration from frame 0.
//     p[0] = img_center;
//     clamp_xy(&p[0]);

//     for (int i = 1; i < n; ++i) {
//       const auto& tr = track_ress_[i];
//       Eigen::Vector2d di = Eigen::Vector2d::Zero();
//       if (tr.delta_valid) {
//         di = tr.delta;
//       } else {
//         // invalid delta: keep previous (0 increment). You can also choose
//         // to use last valid delta or damped velocity, but simplest is 0.
//         di.setZero();
//       }
//       p[i] = p[i - 1] + di;
//       clamp_xy(&p[i]);
//     }

//     if (dbg) {
//       // Residual stats are not meaningful for this integrator; set to zeros for clarity.
//       dbg->step1_global = ResidualStats{};
//       dbg->step1_odom = ResidualStats{};
//     }

//     LOG(INFO) << "[Stabilizer][Step1-Integrate] no global_center_valid found, "
//               << "fallback anchor at frame0=image_center, integrated with delta.";
//     return p;
//   }

//   // 2) Set anchor position.
//   p[anchor] = track_ress_[anchor].global_center;
//   clamp_xy(&p[anchor]);

//   // 3) Forward integration: anchor -> end.
//   int forward_valid_delta_cnt = 0;
//   int forward_invalid_delta_cnt = 0;
//   for (int i = anchor + 1; i < n; ++i) {
//     const auto& tr = track_ress_[i];  // delta(i) is between i-1 and i
//     Eigen::Vector2d di = Eigen::Vector2d::Zero();
//     if (tr.delta_valid) {
//       di = tr.delta;
//       ++forward_valid_delta_cnt;
//     } else {
//       di.setZero();
//       ++forward_invalid_delta_cnt;
//       LOG(WARNING) << "HAS INVALID DELTE!!!!";
//     }
//     p[i] = p[i - 1] + di;
//     clamp_xy(&p[i]);
//   }

//   // 4) Backward integration: anchor -> 0.
//   // Need delta between i and i+1 (stored at tr[i+1]).
//   int backward_valid_delta_cnt = 0;
//   int backward_invalid_delta_cnt = 0;
//   for (int i = anchor - 1; i >= 0; --i) {
//     const auto& tr_next = track_ress_[i + 1];  // delta(i+1) = p[i+1] - p[i]
//     Eigen::Vector2d di = Eigen::Vector2d::Zero();
//     if (tr_next.delta_valid) {
//       di = tr_next.delta;
//       ++backward_valid_delta_cnt;
//     } else {
//       di.setZero();
//       ++backward_invalid_delta_cnt;
//     }
//     // p[i] = p[i+1] - delta(i+1)
//     p[i] = p[i + 1] - di;
//     clamp_xy(&p[i]);
//   }

//   if (config_.timing_log) {
//     LOG(INFO) << "[Stabilizer][Step1-Integrate] anchor=" << anchor
//               << " (frame_idx=" << track_ress_[anchor].frame_idx << ")"
//               << ", forward(valid=" << forward_valid_delta_cnt
//               << ", invalid=" << forward_invalid_delta_cnt << ")"
//               << ", backward(valid=" << backward_valid_delta_cnt
//               << ", invalid=" << backward_invalid_delta_cnt << ")";
//   }

//   if (dbg) {
//     // Optional: compute diagnostic residuals vs detector & delta for reporting only.
//     // (Not used to optimize.)
//     std::vector<double> r_global;
//     std::vector<double> r_odom;
//     r_global.reserve(n);
//     r_odom.reserve(std::max(0, n - 1));

//     for (int i = 0; i < n; ++i) {
//       const auto& tr = track_ress_[i];
//       if (tr.global_center_valid) {
//         r_global.push_back((p[i] - tr.global_center).norm());
//       }
//     }

//     for (int i = 1; i < n; ++i) {
//       const auto& tr = track_ress_[i];
//       if (!tr.delta_valid) continue;
//       const Eigen::Vector2d r = (p[i] - p[i - 1]) - tr.delta;
//       r_odom.push_back(r.norm());
//     }

//     dbg->step1_global = ComputeResidualStats(r_global);
//     dbg->step1_odom = ComputeResidualStats(r_odom);
//   }

//   return p;
// }

// ---------------- Step 1: Object centers (IRLS + Huber) ----------------
std::vector<Eigen::Vector2d> Stabilizer::ComputeObjectCentersIRLS(StabilizerDebugInfo* dbg) const {
  const int n = static_cast<int>(track_ress_.size());
  std::vector<Eigen::Vector2d> p(n, Eigen::Vector2d::Zero());
  if (n == 0) return p;

  const Eigen::Vector2d img_center = ImageCenterProxy(video_info_);

  // Initial guess: detector if valid; otherwise image center.
  for (int i = 0; i < n; ++i) {
    if (track_ress_[i].global_center_valid) {
      p[i] = track_ress_[i].global_center;
    } else {
      p[i] = img_center;
    }
  }

  struct Unary {
    int i;
    Eigen::Vector2d meas;
    double base_w;
    bool is_detector;  // true if from detector
  };
  struct Binary {
    int i0;
    int i1;
    Eigen::Vector2d meas;  // delta or 0
    double base_w;
    bool is_delta;         // true if delta_valid
  };

  std::vector<Unary> unaries;
  std::vector<Binary> binaries;
  unaries.reserve(n);
  binaries.reserve(std::max(0, n - 1));

  int detector_cnt = 0;
  int weak_center_cnt = 0;
  int delta_cnt = 0;
  int weak_odom_cnt = 0;

  for (int i = 0; i < n; ++i) {
    const auto& tr = track_ress_[i];
    if (tr.global_center_valid) {
      unaries.push_back(Unary{i, tr.global_center, WeightFromSigma(config_.global_sigma_px), true});
      ++detector_cnt;
    } else {
      unaries.push_back(Unary{i, img_center, WeightFromSigma(config_.weak_center_sigma_px), false});
      ++weak_center_cnt;
    }
  }

  for (int i = 1; i < n; ++i) {
    const auto& tr = track_ress_[i];
    if (tr.delta_valid) {
      double sigma = (std::isfinite(tr.delta_noise) && tr.delta_noise > 0.0)
                               ? std::max(tr.delta_noise, config_.odom_sigma_px)
                               : config_.odom_sigma_px;
      // debug :sigma = 
      sigma = config_.odom_sigma_px;
      binaries.push_back(Binary{i - 1, i, tr.delta, WeightFromSigma(sigma), true});
      ++delta_cnt;
    } else {
      // LOG(FATAL) << "MUST NOT BE HRERE!!!";
      binaries.push_back(Binary{i - 1, i, Eigen::Vector2d::Zero(),
                                WeightFromSigma(config_.odom_invalid_sigma_px), false});
      ++weak_odom_cnt;
    }
  }

  if (config_.timing_log) {
    LOG(INFO) << "[Stabilizer][Step1] constraints: unary=" << unaries.size()
              << " (detector=" << detector_cnt << " weak_center=" << weak_center_cnt << ")"
              << ", binary=" << binaries.size()
              << " (delta=" << delta_cnt << " weak=" << weak_odom_cnt << ")";
  }

  auto solve_axis = [&](int axis,
                        const std::vector<double>& robust_w_unary,
                        const std::vector<double>& robust_w_binary,
                        const char* tag) -> Eigen::VectorXd {
    std::vector<Eigen::Triplet<double>> Ht;
    Ht.reserve(n * 12);
    Eigen::VectorXd g = Eigen::VectorXd::Zero(n);

    auto add_unary = [&](int i, double w, double meas) {
      Ht.emplace_back(i, i, w);
      g(i) += w * meas;
    };

    auto add_binary = [&](int i0, int i1, double w, double d01) {
      Ht.emplace_back(i1, i1, w);
      Ht.emplace_back(i0, i0, w);
      Ht.emplace_back(i1, i0, -w);
      Ht.emplace_back(i0, i1, -w);
      g(i1) += w * d01;
      g(i0) -= w * d01;
    };

    for (size_t k = 0; k < unaries.size(); ++k) {
      const auto& c = unaries[k];
      const double meas = (axis == 0) ? c.meas.x() : c.meas.y();
      const double w = c.base_w * robust_w_unary[k];
      add_unary(c.i, w, meas);
    }
    for (size_t k = 0; k < binaries.size(); ++k) {
      const auto& c = binaries[k];
      const double meas = (axis == 0) ? c.meas.x() : c.meas.y();
      const double w = c.base_w * robust_w_binary[k];
      add_binary(c.i0, c.i1, w, meas);
    }

    Eigen::VectorXd x(n);
    std::string err;
    if (!SolveSparseSPD(n, Ht, g, tag, &x, &err)) {
      LOG(INFO) << "[Stabilizer][Solve] FAILED: " << err << " (fallback used)";
      return Eigen::VectorXd::Constant(n, (axis == 0) ? img_center.x() : img_center.y());
    }
    return x;
  };

  const bool use_robust = config_.robust_enable;

  for (int it = 0; it < std::max(1, config_.irls_max_iters); ++it) {
    if (stop_.load()) break;

    const bool log_this_iter =
        config_.timing_log && ((it % std::max(1, config_.timing_log_every_irls_iter)) == 0);

    ScopedTimer t_iter("  Step1-IRLS-Iter", log_this_iter);

    std::vector<double> rw_u(unaries.size(), 1.0);
    std::vector<double> rw_b(binaries.size(), 1.0);

    if (use_robust) {
      for (size_t k = 0; k < unaries.size(); ++k) {
        const auto& c = unaries[k];
        const Eigen::Vector2d r = p[c.i] - c.meas;
        rw_u[k] = HuberWeight(r.norm());
      }
      for (size_t k = 0; k < binaries.size(); ++k) {
        const auto& c = binaries[k];
        const Eigen::Vector2d r = (p[c.i1] - p[c.i0]) - c.meas;
        rw_b[k] = HuberWeight(r.norm());
      }
    }

    Eigen::VectorXd xs, ys;
    {
      ScopedTimer t("    Step1-Solve(X)", log_this_iter);
      xs = solve_axis(0, rw_u, rw_b, "Step1-X");
    }
    {
      ScopedTimer t("    Step1-Solve(Y)", log_this_iter);
      ys = solve_axis(1, rw_u, rw_b, "Step1-Y");
    }

    double avg_change = 0.0;
    for (int i = 0; i < n; ++i) {
      const Eigen::Vector2d p_new(xs(i), ys(i));
      avg_change += (p_new - p[i]).norm();
      p[i] = p_new;
    }
    avg_change /= std::max(1, n);

    if (log_this_iter) {
      LOG(INFO) << "[Stabilizer][Step1] iter=" << it
                << " avg_change=" << std::fixed << std::setprecision(4) << avg_change << " px";
    }

    if (avg_change < config_.irls_eps_px) {
      if (config_.timing_log) {
        LOG(INFO) << "[Stabilizer][Step1] early stop: avg_change("
                  << std::fixed << std::setprecision(4) << avg_change
                  << ") < eps(" << config_.irls_eps_px << ")";
      }
      break;
    }
  }

  // Residual stats (final).
  if (dbg) {
    std::vector<double> r_global;
    std::vector<double> r_odom;
    r_global.reserve(unaries.size());
    r_odom.reserve(binaries.size());

    for (const auto& c : unaries) {
      if (!c.is_detector) continue;
      r_global.push_back((p[c.i] - c.meas).norm());
    }
    for (const auto& c : binaries) {
      if (!c.is_delta) continue;
      r_odom.push_back(((p[c.i1] - p[c.i0]) - c.meas).norm());
    }

    dbg->step1_global = ComputeResidualStats(r_global);
    dbg->step1_odom = ComputeResidualStats(r_odom);
  }

  return p;
}

// ---------------- Step 2: Smooth/target centers (IRLS + Huber on data term) ----------------
std::vector<Eigen::Vector2d> Stabilizer::ComputeSmoothCentersIRLS(
    const std::vector<Eigen::Vector2d>& object_centers,
    double smooth_factor,
    StabilizerDebugInfo* dbg) const {
  const int n = static_cast<int>(object_centers.size());
  std::vector<Eigen::Vector2d> s(n, Eigen::Vector2d::Zero());
  if (n == 0) return s;

  const double sf = Clamp01(smooth_factor);

  const double data_sigma = std::max(
      config_.smooth_data_sigma_px * (1.0 - sf) + config_.smooth_data_sigma_min_px * sf,
      config_.min_sigma_px);
  const double w_data_base = WeightFromSigma(data_sigma);

  const double vel_sigma = std::max(config_.smooth_vel_sigma_px / std::max(sf, 1e-3), config_.min_sigma_px);
  const double acc_sigma = std::max(config_.smooth_acc_sigma_px / std::max(sf, 1e-3), config_.min_sigma_px);
  const double w_vel = WeightFromSigma(vel_sigma);
  const double w_acc = WeightFromSigma(acc_sigma);

  if (config_.timing_log) {
    std::ostringstream oss;
    oss << "[Stabilizer][Step2] weights: data_sigma=" << std::fixed << std::setprecision(3) << data_sigma
        << " (w=" << std::scientific << w_data_base << "), vel_sigma=" << std::fixed << vel_sigma
        << " (w=" << std::scientific << w_vel << "), acc_sigma=" << std::fixed << acc_sigma
        << " (w=" << std::scientific << w_acc << ")";
    LOG(INFO) << oss.str();
  }

  s = object_centers;

  auto solve_axis = [&](int axis, const std::vector<double>& robust_w_data, const char* tag) -> Eigen::VectorXd {
    std::vector<Eigen::Triplet<double>> Ht;
    Ht.reserve(n * 30);
    Eigen::VectorXd g = Eigen::VectorXd::Zero(n);

    auto add_linear = [&](const std::vector<int>& idx,
                          const std::vector<double>& a,
                          double w,
                          double b) {
      const int k = static_cast<int>(idx.size());
      for (int p = 0; p < k; ++p) {
        const int i = idx[p];
        const double ap = a[p];
        g(i) += w * ap * b;
        for (int q = 0; q < k; ++q) {
          const int j = idx[q];
          const double aq = a[q];
          Ht.emplace_back(i, j, w * ap * aq);
        }
      }
    };

    // Data: s_i = object_i (robust)
    for (int i = 0; i < n; ++i) {
      const double meas = (axis == 0) ? object_centers[i].x() : object_centers[i].y();
      const double w = w_data_base * robust_w_data[i];
      add_linear({i}, {1.0}, w, meas);
    }

    // Velocity: s_i - s_{i-1} = 0
    for (int i = 1; i < n; ++i) {
      add_linear({i - 1, i}, {-1.0, 1.0}, w_vel, 0.0);
    }

    // Acceleration: s_{i+1} - 2*s_i + s_{i-1} = 0
    for (int i = 1; i + 1 < n; ++i) {
      add_linear({i - 1, i, i + 1}, {1.0, -2.0, 1.0}, w_acc, 0.0);
    }

    Eigen::VectorXd x(n);
    std::string err;
    if (!SolveSparseSPD(n, Ht, g, tag, &x, &err)) {
      LOG(INFO) << "[Stabilizer][Solve] FAILED: " << err << " (fallback used)";
      Eigen::VectorXd xf(n);
      for (int i = 0; i < n; ++i) xf(i) = (axis == 0) ? object_centers[i].x() : object_centers[i].y();
      return xf;
    }
    return x;
  };

  const bool use_robust = config_.robust_enable;

  for (int it = 0; it < std::max(1, config_.irls_max_iters); ++it) {
    if (stop_.load()) break;

    const bool log_this_iter =
        config_.timing_log && ((it % std::max(1, config_.timing_log_every_irls_iter)) == 0);

    ScopedTimer t_iter("  Step2-IRLS-Iter", log_this_iter);

    std::vector<double> rw_data(n, 1.0);
    if (use_robust) {
      for (int i = 0; i < n; ++i) {
        const Eigen::Vector2d r = s[i] - object_centers[i];
        rw_data[i] = HuberWeight(r.norm());
      }
    }

    Eigen::VectorXd xs, ys;
    {
      ScopedTimer t("    Step2-Solve(X)", log_this_iter);
      xs = solve_axis(0, rw_data, "Step2-X");
    }
    {
      ScopedTimer t("    Step2-Solve(Y)", log_this_iter);
      ys = solve_axis(1, rw_data, "Step2-Y");
    }

    double avg_change = 0.0;
    for (int i = 0; i < n; ++i) {
      const Eigen::Vector2d s_new(xs(i), ys(i));
      avg_change += (s_new - s[i]).norm();
      s[i] = s_new;
    }
    avg_change /= std::max(1, n);

    if (log_this_iter) {
      LOG(INFO) << "[Stabilizer][Step2] iter=" << it
                << " avg_change=" << std::fixed << std::setprecision(4) << avg_change << " px";
    }

    if (avg_change < config_.irls_eps_px) {
      if (config_.timing_log) {
        LOG(INFO) << "[Stabilizer][Step2] early stop: avg_change("
                  << std::fixed << std::setprecision(4) << avg_change
                  << ") < eps(" << config_.irls_eps_px << ")";
      }
      break;
    }
  }

  if (dbg) {
    std::vector<double> r_data;
    r_data.reserve(n);
    for (int i = 0; i < n; ++i) {
      r_data.push_back((s[i] - object_centers[i]).norm());
    }
    dbg->step2_data = ComputeResidualStats(r_data);
  }

  return s;
}

// ---------------- Step 3: stable deltas ----------------
// delta = object_center - target_center
std::vector<FrameStableResult> Stabilizer::ComputeStabilizedResults(
    const std::vector<Eigen::Vector2d>& object_centers,
    const std::vector<Eigen::Vector2d>& target_centers) const {
  const int n = static_cast<int>(track_ress_.size());
  std::vector<FrameStableResult> out;
  out.reserve(n);

  const int m = std::min(static_cast<int>(object_centers.size()),
                         static_cast<int>(target_centers.size()));

  for (int i = 0; i < n; ++i) {
    FrameStableResult r;
    r.frame_idx = track_ress_[i].frame_idx;
    r.time_ns = track_ress_[i].time_ns;

    if (i < m) {
      const Eigen::Vector2d delta = object_centers[i] - target_centers[i];
      r.delta_x = delta.x();
      r.delta_y = delta.y();
    } else {
      r.delta_x = 0.0;
      r.delta_y = 0.0;
      LOG(FATAL) << "Must not be here!!!";
    }
    out.push_back(r);
  }
  return out;
}

// ---------------- Debug artifacts: CSV + SVG ----------------
void Stabilizer::EmitDebugArtifacts(const std::vector<Eigen::Vector2d>& object_centers,
                                    const std::vector<Eigen::Vector2d>& target_centers,
                                    const std::vector<FrameStableResult>& stable,
                                    StabilizerDebugInfo* dbg) const {
  if (!dbg) return;

#if __has_include(<filesystem>)
  try {
    fs::path out_dir(dbg->output_dir);
    fs::create_directories(out_dir);

    dbg->csv_path = (out_dir / "debug.csv").string();
    dbg->svg_path = (out_dir / "debug.svg").string();
  } catch (...) {
    dbg->csv_path.clear();
    dbg->svg_path.clear();
    return;
  }
#else
  (void)object_centers;
  (void)target_centers;
  (void)stable;
  return;
#endif

  const int n = static_cast<int>(stable.size());
  if (n == 0) return;

  std::vector<double> frame_delta_norm(n, 0.0);
  for (int i = 0; i < n; ++i) {
    const double dx = stable[i].delta_x;
    const double dy = stable[i].delta_y;
    frame_delta_norm[i] = std::sqrt(dx * dx + dy * dy);
  }

  // CSV
  {
    std::ofstream ofs(dbg->csv_path, std::ios::out | std::ios::trunc);
    if (ofs.is_open()) {
      ofs << "frame_idx,time_ns,"
          << "obj_x,obj_y,"
          << "target_x,target_y,"
          << "delta_x,delta_y,delta_norm\n";
      for (int i = 0; i < n; ++i) {
        const auto& tr = track_ress_[i];
        const Eigen::Vector2d obj = (i < static_cast<int>(object_centers.size()))
                                        ? object_centers[i]
                                        : Eigen::Vector2d::Zero();
        const Eigen::Vector2d tgt = (i < static_cast<int>(target_centers.size()))
                                        ? target_centers[i]
                                        : Eigen::Vector2d::Zero();
        ofs << tr.frame_idx << "," << tr.time_ns << ","
            << obj.x() << "," << obj.y() << ","
            << tgt.x() << "," << tgt.y() << ","
            << stable[i].delta_x << "," << stable[i].delta_y << ","
            << frame_delta_norm[i] << "\n";
      }
    }
  }

  // SVG (X and Y stacked)
  {
    std::vector<double> xs;
    xs.reserve(n);
    std::vector<double> obj_x, tgt_x, dx;
    std::vector<double> obj_y, tgt_y, dy;

    obj_x.reserve(n); tgt_x.reserve(n); dx.reserve(n);
    obj_y.reserve(n); tgt_y.reserve(n); dy.reserve(n);

    for (int i = 0; i < n; ++i) {
      xs.push_back(static_cast<double>(i));
      const Eigen::Vector2d obj = (i < static_cast<int>(object_centers.size()))
                                      ? object_centers[i]
                                      : Eigen::Vector2d::Zero();
      const Eigen::Vector2d tgt = (i < static_cast<int>(target_centers.size()))
                                      ? target_centers[i]
                                      : Eigen::Vector2d::Zero();

      obj_x.push_back(obj.x());
      tgt_x.push_back(tgt.x());
      dx.push_back(stable[i].delta_x);

      obj_y.push_back(obj.y());
      tgt_y.push_back(tgt.y());
      dy.push_back(stable[i].delta_y);
    }

    const std::string top = MakeSvgPlot(xs, obj_x, tgt_x, dx, "Stabilizer Debug: X curves");
    const std::string bot = MakeSvgPlot(xs, obj_y, tgt_y, dy, "Stabilizer Debug: Y curves");

    const int W = 1200;
    const int H = 420 * 2;

    std::ofstream ofs(dbg->svg_path, std::ios::out | std::ios::trunc);
    if (ofs.is_open()) {
      ofs << "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"" << W
          << "\" height=\"" << H << "\">\n";
      ofs << "<g transform=\"translate(0,0)\">\n" << StripSvgContent(top) << "\n</g>\n";
      ofs << "<g transform=\"translate(0,420)\">\n" << StripSvgContent(bot) << "\n</g>\n";
      ofs << "</svg>\n";
    }
  }

  // total_rms (delta magnitude as applied-correction proxy)
  {
    double sum2 = 0.0;
    for (double v : frame_delta_norm) sum2 += v * v;
    dbg->total_rms = std::sqrt(sum2 / std::max(1, n));
  }
}

}  // namespace airsteady
