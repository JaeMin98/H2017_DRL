
# 🤖 H2017 로봇팔 제어 강화학습 환경 설정

이 가이드는 H2017 로봇팔 제어를 위한 강화학습 환경을 단계별로 설정하는 방법을 제공합니다. 운영체제 설치부터 ROS, MoveIt 설치, 그래픽 드라이버 및 CUDA 설정까지 모두 포함되어 있습니다.

---

## 📋 목차

1. [운영체제 설치](#운영체제-설치)
2. [ROS 설치](#ros-설치)
3. [MoveIt 설치](#moveit-설치)
4. [ROS 작업공간 설정](#ros-작업공간-설정)
5. [ROS 패키지 생성](#ros-패키지-생성)
6. [선택 옵션](#선택-옵션)
   - 시스템 업데이트
   - 한국어 키보드 설정
   - pip 설치
   - 추가 프로그램 설치
7. [그래픽 드라이버 및 CUDA 설치](#그래픽-드라이버-및-cuda-설치)
8. [H2017 ROS 패키지 다운로드](#h2017-ros-패키지-다운로드)
9. [bashrc 설정](#bashrc-설정)
10. [PyTorch 및 CUDA 확인](#pytorch-및-cuda-확인)

---

## 💻 운영체제 설치

운영체제 설치는 [이 가이드](https://blog.naver.com/jm_0820/223001100698)를 참고하여 진행합니다.

---

## 🛠️ ROS 설치

ROS Noetic을 설치하려면 [ROS Noetic 설치 가이드](http://wiki.ros.org/noetic/Installation/Ubuntu)를 참고하십시오.

---

## 🦾 MoveIt 설치

MoveIt과 관련 패키지를 설치하려면 다음 명령어를 실행하십시오:

```bash
sudo apt install ros-noetic-moveit
sudo apt-get install ros-noetic-joint-trajectory-controller
sudo apt-get install ros-noetic-rosbridge-server
```

---

## 📁 ROS 작업공간 설정

ROS 작업공간을 설정하려면 다음 단계를 따릅니다:

1. ROS 환경을 불러오기:

    ```bash
    source /opt/ros/noetic/setup.sh
    ```

2. 작업공간 생성 및 초기화:

    ```bash
    mkdir -p ~/catkin_ws/src
    cd ~/catkin_ws/src
    catkin_init_workspace
    ```

3. 컴파일 및 환경 설정:

    ```bash
    cd ~/catkin_ws
    catkin_make
    source devel/setup.bash
    ```

[자세한 가이드](http://wiki.ros.org/ko/catkin/Tutorials/create_a_workspace)도 참고 가능합니다.

---

## 📁 ROS 패키지 생성

1. ROS 패키지를 생성하려면:

    ```bash
    cd ~/catkin_ws/src
    catkin_create_pkg my_package
    ```

2. 패키지 컴파일 및 환경 설정:

    ```bash
    cd ~/catkin_ws
    catkin_make
    source devel/setup.bash
    ```

---

## ⚙️ 선택 옵션

### 📅 시스템 업데이트

```bash
sudo apt-get update
sudo apt-get upgrade
```

### ⌨️ 한국어 키보드 설정

[한국어 키보드 설정 가이드](https://shanepark.tistory.com/231)를 참고하여 설정하십시오.

### 🐍 pip 설치

```bash
sudo apt-get install python3-pip
```

### 💻 추가 프로그램 설치

필요한 추가 프로그램을 설치할 수 있습니다:

- [GitHub Desktop](https://gist.github.com/berkorbay/6feda478a00b0432d13f1fc0a50467f1)
- [TeamViewer](https://www.teamviewer.com/ko/download/linux/)
- [VSCode](https://code.visualstudio.com/download)

```bash
sudo apt install barrier -y  # KVM 스위치 소프트웨어
sudo apt-get install terminator  # 편리한 터미널
```

---

## 🎨 그래픽 드라이버 및 CUDA 설치

### 🚮 기존 그래픽 드라이버 제거

```bash
sudo apt --purge remove *nvidia*
sudo apt-get autoremove
sudo apt-get autoclean
sudo rm -rf /usr/local/cuda*
```

### 🎯 그래픽 드라이버 설치

1. 설치 가능한 드라이버 확인:

    ```bash
    ubuntu-drivers devices
    ```

2. 드라이버 설치:

    ```bash
    sudo apt-get install nvidia-driver-<버전번호>
    sudo apt-get install dkms nvidia-modprobe
    sudo apt-get update
    sudo apt-get upgrade
    sudo reboot now
    ```

3. 설치 확인:

    ```bash
    nvidia-smi
    ```

### 🖥️ CUDA 설치

1. CUDA 설치:

    ```bash
    sudo apt install nvidia-cuda-toolkit
    ```

2. [CUDA 설치 가이드](https://developer.nvidia.com/cuda-toolkit-archive)를 참고하여 설치를 진행합니다.

3. 설치 확인:

    ```bash
    nvcc -V
    ```

### 💾 cuDNN 설치

[cuDNN 설치 가이드](https://developer.nvidia.com/rdp/cudnn-archive)를 참고하여 설치합니다.

---

## 🦾 H2017 ROS 패키지 다운로드

1. [패키지 다운로드](https://drive.google.com/file/d/1KZgBUNl1ph5HkjhQQP4d1OzQcDZyZeMA/view?usp=drive_link) 후, 패키지를 `~/catkin_ws/src`에 넣습니다.

2. 컴파일 및 실행:

    ```bash
    cd ~/catkin_ws
    catkin_make
    source ./devel/setup.bash
    source ~/.bashrc
    roslaunch h2017 demo_gazebo.launch
    ```

---

## 🛠️ bashrc 설정

`~/.bashrc` 파일에 아래 라인을 추가하여 환경 설정을 편리하게 구성합니다:

```bash
# CUDA 경로 설정
export PATH=/usr/local/cuda-12.1/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-12.1/lib64:$LD_LIBRARY_PATH

# python3 기본 설정
alias python=python3
alias pip=pip3

# ROS 환경 설정
source /opt/ros/noetic/setup.bash
source ~/catkin_ws/devel/setup.bash

# 단축 명령어
alias sb="source ~/.bashrc"
alias cm="catkin_make & source ./devel/setup.bash"
alias rc='rosclean purge -y'
alias run='rosclean purge -y & roslaunch h2017 demo_gazebo.launch'

# ROS IP 및 포트 설정
# default 포트번호는 11311임
# example) export ROS_MASTER_URI=http://192.168.0.1:11311
export ROS_MASTER_URI=http://<IP주소>:<포트번호>
export ROS_HOSTNAME=<IP주소>
```

---

## 🔥 PyTorch 및 CUDA 확인

다음 Python 코드를 실행하여 CUDA와 cuDNN이 올바르게 설정되었는지 확인합니다:

```python
import torch

print(torch.cuda.is_available())
if torch.cuda.is_available():
    print(torch.cuda.current_device())
    print(torch.cuda.get_device_name(torch.cuda.current_device()))

print(torch.backends.cudnn.enabled)
print(torch.backends.cudnn.version())
```

---
