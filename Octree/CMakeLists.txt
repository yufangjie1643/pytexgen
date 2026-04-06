ADD_LIBRARY(Octree STATIC
Array.cpp
Octree.cpp
OctreeAuxiliary.cpp
OctreeImplementation.cpp
OctreeStreamOut.cpp
Vector3f.cpp
Array.hpp
hxa7241_general.hpp
hxa7241_graphics.hpp
Octree.hpp
OctreeAuxiliary.hpp
OctreeImplementation.hpp
OctreeStreamOut.hpp
Primitives.hpp
Vector3f.hpp
)
SET_TARGET_PROPERTIES(Octree PROPERTIES POSITION_INDEPENDENT_CODE ON)
# The .h files are not strictly necessary but it adds them to the project IDE making development easier
