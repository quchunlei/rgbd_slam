ADD_EXECUTABLE( main main.cpp )

FIND_PACKAGE( PCL REQUIRED )
FIND_PACKAGE( OpenCV 2.4 REQUIRED)
INCLUDE_DIRECTORIES( ${Opencv_INCLUDE_DIRS} )

list(REMOVE_ITEM PCL_LIBRARIES "vtkproj4") # use this in Ubuntu 16.04

ADD_DEFINITIONS( ${PCL_DEFINITIONS})
INCLUDE_DIRECTORIES(${PCL_INCLUDE_DIRS} )
LINK_DIRECTORIES( ${PCL_LIBERARY_DIRS} )

# 添加g2o的依赖
LIST( APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake_modules )
SET( G2O_ROOT /usr/local/include/g2o )
FIND_PACKAGE( G2O )
# CSparse
FIND_PACKAGE( CSparse )
INCLUDE_DIRECTORIES( ${G2O_INCLUDE_DIR} ${CSPARSE_INCLUDE_DIR} )

ADD_EXECUTABLE( generate_pointcloud generatePointCloud.cpp)
TARGET_LINK_LIBRARIES( generate_pointcloud ${OpenCV_LIBS} ${PCL_LIBRARIES} )

ADD_LIBRARY( slambase slamBase.cpp )
TARGET_LINK_LIBRARIES( slambase ${OpenCV_LIBS} ${PCL_LIBRARIES} )

ADD_EXECUTABLE( detecFeature detecFeature.cpp)
TARGET_LINK_LIBRARIES( detecFeature slambase ${OpenCV_LIBS} ${PCL_LIBRARIES} )

add_executable(joinRgbdCloud joinRgbdCloud.cpp)
target_link_libraries(joinRgbdCloud slambase ${OpenCV_LIBS} ${PCL_LIBRARIES})

add_executable(visualOdometry visualOdometry.cpp)
target_link_libraries(visualOdometry slambase ${OpenCV_LIBS} ${PCL_LIBRARIES})

add_executable(slamEnd slamEnd.cpp)
target_link_libraries(slamEnd slambase ${OpenCV_LIBS} ${PCL_LIBRARIES} 
                      g2o_core g2o_types_slam3d g2o_solver_csparse g2o_stuff g2o_csparse_extension ${CSPARSE_LIBRARY})

add_executable(slam slam.cpp)
target_link_libraries(slam slambase ${OpenCV_LIBS} ${PCL_LIBRARIES} 
                      g2o_core g2o_types_slam3d g2o_solver_csparse g2o_stuff g2o_csparse_extension ${CSPARSE_LIBRARY})