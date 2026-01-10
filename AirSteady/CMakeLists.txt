cmake_minimum_required(VERSION 3.21)

project(AirSteady
    VERSION 0.1.0
    LANGUAGES CXX)

# C++ 标准
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Qt 自动 MOC / UIC / RCC
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)
set(CMAKE_AUTOUIC ON)

# ---------------- 依赖 ----------------
find_package(Qt6 CONFIG REQUIRED COMPONENTS Core Gui Widgets)
find_package(OpenCV REQUIRED)
find_package(Glog REQUIRED)
find_package(FFMPEG REQUIRED)
message(STATUS "FFMPEG found: ${FFMPEG_FOUND}")
find_package(cxxopts CONFIG REQUIRED)
find_package(fmt CONFIG REQUIRED)

# Direct ML Yolo.
option(ORT_DML_VERSION "Microsoft.ML.OnnxRuntime.DirectML NuGet version" "1.23.0")
set (ORT_DML_VERSION "1.23.0")
include(${CMAKE_CURRENT_LIST_DIR}/cmake/FetchOrtDml.cmake)
fetch_ort_dml(${ORT_DML_VERSION})

include_directories(
    .
    ${ORT_DML_INCLUDE_DIR}
)

file(GLOB_RECURSE AIRSTEADY_LIB_SOURCES CONFIGURE_DEPENDS
  "${CMAKE_CURRENT_SOURCE_DIR}/ui/*.cpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/algorithm/*.cpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/common/*.cpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/yolo/*.cpp"
)

file(GLOB_RECURSE AIRSTEADY_LIB_HEADERS CONFIGURE_DEPENDS
  "${CMAKE_CURRENT_SOURCE_DIR}/ui/*.h"
  "${CMAKE_CURRENT_SOURCE_DIR}/ui/*.hpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/algorithm/*.h"
  "${CMAKE_CURRENT_SOURCE_DIR}/algorithm/*.hpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/common/*.h"
  "${CMAKE_CURRENT_SOURCE_DIR}/common/*.hpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/yolo/*.h
  "${CMAKE_CURRENT_SOURCE_DIR}/yolo/*.hpp
)

# Exclude patterns: *_exe.cpp and *_test.cpp
list(FILTER AIRSTEADY_LIB_SOURCES EXCLUDE REGEX ".*_exe\\.cpp$")
list(FILTER AIRSTEADY_LIB_SOURCES EXCLUDE REGEX ".*_test\\.cpp$")

# Windows 默认用 STATIC，避免 DLL 导出/导入库问题
if (WIN32)
  add_library(airsteady_lib STATIC
      ${AIRSTEADY_LIB_SOURCES}
      ${AIRSTEADY_LIB_HEADERS}
  )
else()
  add_library(airsteady_lib SHARED
      ${AIRSTEADY_LIB_SOURCES}
      ${AIRSTEADY_LIB_HEADERS}
  )
endif()

target_compile_definitions(airsteady_lib PRIVATE
  __STDC_CONSTANT_MACROS
  __STDC_FORMAT_MACROS
  __STDC_LIMIT_MACROS
)

# 头文件包含路径：作为 PUBLIC 让 exe 自动继承
target_include_directories(airsteady_lib
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/algorithm
        ${CMAKE_CURRENT_SOURCE_DIR}/ui
    PRIVATE
        ${OpenCV_INCLUDE_DIRS}
)

# 链接 Qt + OpenCV：作为 PUBLIC 让 exe 自动继承（通常更省事）
target_link_libraries(airsteady_lib
    PUBLIC
        Qt6::Core
        Qt6::Gui
        Qt6::Widgets
        ${OpenCV_LIBS}
        glog::glog
        ${FFMPEG_LIBRARIES}
        ${ORT_DML_LIB}
)

# Windows MSVC 一些常用编译选项
if (MSVC)
    target_compile_options(airsteady_lib PRIVATE
        /W4
        /permissive-
    )
endif()

# ---------------- 可执行程序 ----------------
add_executable(AirSteady
    exe/main.cpp
)

target_link_libraries(AirSteady
    PRIVATE
        airsteady_lib
)

# Windows 下：如果你未来切换为 WIN32 子系统，把这里改：
# add_executable(AirSteady WIN32 exe/main.cpp)

# ---------------- 安装规则 ----------------
install(TARGETS AirSteady
    RUNTIME DESTINATION bin
)

# 库的安装（Windows 下会产出 .dll + .lib；Linux/macOS 产出 .so/.dylib）
install(TARGETS airsteady_lib
    RUNTIME DESTINATION bin
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
)

if (WIN32)
  get_target_property(_qt6_qmake_location Qt6::qmake IMPORTED_LOCATION)
  get_filename_component(_qt6_bin_dir "${_qt6_qmake_location}" DIRECTORY)

  add_custom_command(TARGET AirSteady POST_BUILD
    COMMAND "${_qt6_bin_dir}/windeployqt.exe" --release "$<TARGET_FILE:AirSteady>"
    COMMAND ${CMAKE_COMMAND} -E copy_directory
            "${CMAKE_CURRENT_SOURCE_DIR}/assets"
            "$<TARGET_FILE_DIR:AirSteady>/assets"
    COMMAND ${CMAKE_COMMAND} -E copy_directory
            "${CMAKE_CURRENT_SOURCE_DIR}/model"
            "$<TARGET_FILE_DIR:AirSteady>/model"
    COMMENT "Deploying Qt + copying assets/model to output directory"
    VERBATIM
  )
endif()