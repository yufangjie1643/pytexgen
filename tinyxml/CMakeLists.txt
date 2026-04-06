ADD_LIBRARY(TinyXML STATIC
tinystr.cpp
tinyxml.cpp
tinyxmlerror.cpp
tinyxmlparser.cpp
tinystr.h
tinyxml.h
)
SET_TARGET_PROPERTIES(TinyXML PROPERTIES POSITION_INDEPENDENT_CODE ON)
# The .h files are not strictly necessary but it adds them to the project IDE making development easier
ADD_DEFINITIONS(-DTIXML_USE_STL)
