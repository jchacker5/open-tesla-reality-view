# doctest (single header, MIT), downloaded at configure time and pinned by hash.
set(OTRV_DOCTEST_VERSION 2.4.12)
set(OTRV_DOCTEST_DIR ${CMAKE_BINARY_DIR}/_deps/doctest)
set(OTRV_DOCTEST_HEADER ${OTRV_DOCTEST_DIR}/doctest/doctest.h)
if(NOT EXISTS ${OTRV_DOCTEST_HEADER})
  file(DOWNLOAD https://raw.githubusercontent.com/doctest/doctest/v${OTRV_DOCTEST_VERSION}/doctest/doctest.h
       ${OTRV_DOCTEST_HEADER}
       EXPECTED_HASH SHA256=94029a7d32da24a56249658147dbd2b33ff0b9ed665295cbbaf19aafff5b0ced
       TLS_VERIFY ON)
endif()
add_library(doctest INTERFACE)
target_include_directories(doctest SYSTEM INTERFACE ${OTRV_DOCTEST_DIR})
