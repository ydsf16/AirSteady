#include "common/file_utils.hpp"

#include <filesystem>
#include <system_error>
#include <vector>

#if defined(_WIN32)
  #ifndef NOMINMAX
    #define NOMINMAX
  #endif
  #include <Windows.h>
#elif defined(__APPLE__)
  #include <mach-o/dyld.h>
  #include <unistd.h>
  #include <limits.h>
#endif

namespace airsteady {
namespace fs = std::filesystem;

#if defined(_WIN32)
static std::string WideToUtf8(const std::wstring& w) {
  if (w.empty()) return std::string();
  const int size_needed = WideCharToMultiByte(CP_UTF8, 0, w.data(),
                                              static_cast<int>(w.size()),
                                              nullptr, 0, nullptr, nullptr);
  if (size_needed <= 0) return std::string();
  std::string result(size_needed, '\0');
  WideCharToMultiByte(CP_UTF8, 0, w.data(), static_cast<int>(w.size()),
                      result.data(), size_needed, nullptr, nullptr);
  return result;
}
#endif

bool CreateFolder(const std::string& folder) {
  if (folder.empty()) return false;
  std::error_code ec;
  // create_directories returns true if it created something, false if it already existed
  // but both cases are "success" for our use.
  fs::create_directories(fs::path(folder), ec);
  if (ec) return false;
  return fs::exists(fs::path(folder), ec) && !ec;
}

bool RemoveFolder(const std::string& folder) {
  if (folder.empty()) return false;
  std::error_code ec;
  if (!fs::exists(fs::path(folder), ec)) {
    // Not existing is treated as success (idempotent).
    return !ec;
  }
  fs::remove_all(fs::path(folder), ec);
  return !ec;
}

bool GetVideoName(const std::string& video_path, std::string* out_name) {
  if (!out_name) return false;
  out_name->clear();
  if (video_path.empty()) return false;

  std::error_code ec;
  fs::path p(video_path);

  // If user passes folder path, stem() can be empty; treat as failure.
  if (!p.has_filename()) return false;

  const std::string stem = p.stem().string();
  if (stem.empty()) return false;

  *out_name = stem;
  (void)ec;
  return true;
}

bool GetVideoFolder(const std::string& video_path, std::string* out_folder) {
  if (!out_folder) return false;
  out_folder->clear();
  if (video_path.empty()) return false;

  fs::path p(video_path);
  fs::path parent = p.parent_path();
  if (parent.empty()) return false;

  *out_folder = parent.string();
  return true;
}

bool GetExeFolder(std::string* out_folder) {
  if (!out_folder) return false;
  out_folder->clear();

#if defined(_WIN32)
  // GetModuleFileNameW needs a buffer; grow if required.
  std::vector<wchar_t> buffer(260);
  while (true) {
    const DWORD len = GetModuleFileNameW(nullptr, buffer.data(),
                                         static_cast<DWORD>(buffer.size()));
    if (len == 0) return false;

    // If len == buffer.size(), it may be truncated; grow and retry.
    if (len >= buffer.size() - 1) {
      buffer.resize(buffer.size() * 2);
      continue;
    }

    std::wstring wpath(buffer.data(), len);
    fs::path exe_path(wpath);
    fs::path exe_dir = exe_path.parent_path();
    if (exe_dir.empty()) return false;

    // Return UTF-8
    *out_folder = WideToUtf8(exe_dir.wstring());
    return !out_folder->empty();
  }

#elif defined(__APPLE__)
  uint32_t size = 0;
  // First call to get required size.
  if (_NSGetExecutablePath(nullptr, &size) != -1) {
    // Should not happen; size should be set and function should return -1.
  }

  std::vector<char> buf(size + 1, '\0');
  if (_NSGetExecutablePath(buf.data(), &size) != 0) return false;

  // Resolve symlinks / relative pieces.
  char resolved[PATH_MAX] = {0};
  if (realpath(buf.data(), resolved) == nullptr) {
    // If realpath fails, fall back to the raw path.
    fs::path exe_path(buf.data());
    fs::path exe_dir = exe_path.parent_path();
    if (exe_dir.empty()) return false;
    *out_folder = exe_dir.string();
    return !out_folder->empty();
  }

  fs::path exe_path(resolved);
  fs::path exe_dir = exe_path.parent_path();
  if (exe_dir.empty()) return false;

  *out_folder = exe_dir.string();
  return !out_folder->empty();

#else
  // Not requested, but leaving a safe fallback.
  return false;
#endif
}

bool IsFileExist(const std::string& file_path) {
  if (file_path.empty()) return false;
  std::error_code ec;
  const fs::path p(file_path);
  if (!fs::exists(p, ec) || ec) return false;
  return fs::is_regular_file(p, ec) && !ec;
}

bool IsFolderExist(const std::string& folder_path) {
  if (folder_path.empty()) return false;
  std::error_code ec;
  const fs::path p(folder_path);
  if (!fs::exists(p, ec) || ec) return false;
  return fs::is_directory(p, ec) && !ec;
}

}  // namespace airsteady
