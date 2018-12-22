ADD_EXECUTABLE( main main.cpp )

FIND_PACKAGE( PCL REQUIRED )
FIND_PACKAGE( OpenCV 2.4 REQUIRED)
INCLUDE_DIRECTORIES( ${Opencv_INCLUDE_DIRS} )
FIND_PACKAGE( libpointmatcher REQUIRED)

#list(REMOVE_ITEM PCL_LIBRARIES "vtkproj4") # use this in Ubuntu 16.04

ADD_DEFINITIONS( ${PCL_DEFINITIONS})
INCLUDE_DIRECTORIES(${PCL_INCLUDE_DIRS} )
LINK_DIRECTORIES( ${PCL_LIBERARY_DIRS} )

ADD_EXECUTABLE( generate_pointcloud generatePointCloud.cpp)
TARGET_LINK_LIBRARIES( generate_pointcloud ${OpenCV_LIBS} ${PCL_LIBRARIES} )

ADD_LIBRARY( slambase slamBase.cpp )
TARGET_LINK_LIBRARIES( slambase ${OpenCV_LIBS} ${PCL_LIBRARIES} )

ADD_EXECUTABLE( detecFeature detecFeature.cpp)
TARGET_LINK_LIBRARIES( detecFeature slambase ${OpenCV_LIBS} ${PCL_LIBRARIES} )

add_executable(joinRgbdCloud joinRgbdCloud.cpp)
target_link_libraries(joinRgbdCloud slambase ${OpenCV_LIBS} ${PCL_LIBRARIES})
