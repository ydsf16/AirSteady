# FetchOrtDml.cmake
# Downloads and extracts Microsoft.ML.OnnxRuntime.DirectML (native) NuGet package.
#
# Outputs:
#   ORT_DML_ROOT
#   ORT_DML_INCLUDE_DIR
#   ORT_DML_LIB
#   ORT_DML_DLL
#   ORT_DML_PROVIDERS_SHARED_DLL
#
# Note: We use NuGet because vcpkg's onnxruntime port does not currently expose a DirectML feature.
#       This keeps the app "any DX12 GPU" compatible via DirectML.


function(fetch_ort_dml ORT_VERSION)
  set(ORT_DML_ROOT "${CMAKE_BINARY_DIR}/_deps/ort_dml_${ORT_VERSION}")
  set(NUPKG_PATH "${ORT_DML_ROOT}/ort_dml_${ORT_VERSION}.nupkg")
  set(EXTRACT_DIR "${ORT_DML_ROOT}/pkg")

  if(NOT EXISTS "${EXTRACT_DIR}/build/native/include/onnxruntime_cxx_api.h")
    file(MAKE_DIRECTORY "${ORT_DML_ROOT}")
    message(STATUS "Downloading Microsoft.ML.OnnxRuntime.DirectML ${ORT_VERSION} ...")
    set(URL "https://www.nuget.org/api/v2/package/Microsoft.ML.OnnxRuntime.DirectML/1.23.0")
    # set(URL "https://www.nuget.org/api/v2/package/Microsoft.ML.OnnxRuntime.DirectML")

    file(DOWNLOAD "${URL}" "${NUPKG_PATH}"
      SHOW_PROGRESS
      STATUS DL_STATUS
      TLS_VERIFY ON
    )
    list(GET DL_STATUS 0 DL_CODE)
    list(GET DL_STATUS 1 DL_MSG)
    if(NOT DL_CODE EQUAL 0)
      message(FATAL_ERROR "Failed to download NuGet package: ${DL_MSG} (code=${DL_CODE})")
    endif()

    file(MAKE_DIRECTORY "${EXTRACT_DIR}")
    message(STATUS "Extracting ${NUPKG_PATH} ...")
    file(ARCHIVE_EXTRACT INPUT "${NUPKG_PATH}" DESTINATION "${EXTRACT_DIR}")
  endif()

  set(ORT_DML_INCLUDE_DIR "${EXTRACT_DIR}/build/native/include" PARENT_SCOPE)

  if(CMAKE_SYSTEM_PROCESSOR MATCHES "ARM64|aarch64|ARM64EC")
    set(_ort_runtime "win-arm64")
  else()
    set(_ort_runtime "win-x64")
  endif()

  set(_runtime_dir "${EXTRACT_DIR}/runtimes/${_ort_runtime}/native")
  set(_ort_lib "${_runtime_dir}/onnxruntime.lib")
  set(_ort_dll "${_runtime_dir}/onnxruntime.dll")
  set(_ort_ps "${_runtime_dir}/onnxruntime_providers_shared.dll")

  if(NOT EXISTS "${_ort_lib}")
    message(FATAL_ERROR "onnxruntime.lib not found at ${_ort_lib}. NuGet layout may have changed.")
  endif()
  if(NOT EXISTS "${_ort_dll}")
    message(FATAL_ERROR "onnxruntime.dll not found at ${_ort_dll}. NuGet layout may have changed.")
  endif()
  if(NOT EXISTS "${_ort_ps}")
    message(FATAL_ERROR "onnxruntime_providers_shared.dll not found at ${_ort_ps}. NuGet layout may have changed.")
  endif()

  set(ORT_DML_ROOT "${EXTRACT_DIR}" PARENT_SCOPE)
  set(ORT_DML_LIB "${_ort_lib}" PARENT_SCOPE)
  set(ORT_DML_DLL "${_ort_dll}" PARENT_SCOPE)
  set(ORT_DML_PROVIDERS_SHARED_DLL "${_ort_ps}" PARENT_SCOPE)

  message(STATUS "ORT(DML) include: ${EXTRACT_DIR}/build/native/include")
  message(STATUS "ORT(DML) runtime: ${_runtime_dir}")
endfunction()
