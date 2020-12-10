FROM ros:kinetic-ros-base

# install ROS dependencies and pip
RUN apt-get update && apt-get install -y \
    ros-kinetic-stage-ros \
    ros-kinetic-rviz \
    ros-kinetic-fake-localization \
    ros-kinetic-map-server \
    ros-kinetic-move-base \
    ros-kinetic-dwa-local-planner \
    python-pip \
&& rm -rf /var/lib/apt/lists/*

# install tf dependency
# specific version of pip in case a future one might break things in 2.7
RUN python -m pip install --upgrade pip==20.3.1 && \
python -m pip install virtualenv &&  \
python -m virtualenv --system-site-packages /home/venv && \
/home/venv/bin/pip install --upgrade tensorflow==1.9.0

# copy over and build our package
RUN mkdir -p /home/catkin_ws/src
WORKDIR /home
COPY . catkin_ws/src/robot_nav
RUN /bin/bash -c '. /opt/ros/kinetic/setup.bash; cd /home/catkin_ws; catkin_make'

# add source ros and our package to .bashrc
RUN echo source "/opt/ros/kinetic/setup.bash" >> ~/.bashrc
RUN echo source "/home/catkin_ws/devel/setup.bash" >> ~/.bashrc

