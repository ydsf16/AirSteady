#pragma once

#include <string>

namespace airsteady {

// Create folder recursively (mkdir -p semantics).
bool CreateFolder(const std::string& folder);

// Remove folder recursively (rm -rf semantics).
bool RemoveFolder(const std::string& folder);

// Extract video file name (without extension) from path.
// e.g. "/a/b/c.mp4" -> "c"
bool GetVideoName(const std::string& video_path, std::string* out_name);

// Extract parent folder from path.
// e.g. "/a/b/c.mp4" -> "/a/b"
bool GetVideoFolder(const std::string& video_path, std::string* out_folder);

// Get folder where the current executable resides.
bool GetExeFolder(std::string* out_folder);

bool IsFileExist(const std::string& file_path);

bool IsFolderExist(const std::string& folder_path);

}  // namespace airsteady
