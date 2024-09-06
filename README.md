
# 🤖 H2017 로봇팔 강화학습 환경 설정
![제목 없음](https://github.com/user-attachments/assets/9274718f-938d-416a-ac4b-b98b2dc8bf8e)
본 가이드는 H2017 로봇팔 제어를 위한 강화학습 환경을 단계별로 설정하는 방법을 제공합니다. [두산 로보틱스 GitHub](https://github.com/doosan-robotics/doosan-robot)에서 제공하는 URDF를 기반으로 하며, end-effector와 base의 collision, visual 모델은 자체 제작된 모델입니다. 또한, DSR 에뮬레이터(dsr_emulator)를 도커(Docker)로 실행해야 하는 번거로움과 배속 조절이 불가능한 문제를 개선한 환경을 제공합니다.
이 가이드에서는 운영체제 설치부터 ROS와 MoveIt 설치, 그래픽 드라이버 및 CUDA 설정까지 환경 구성을 위한 모든 절차를 포함하고 있습니다.


---

## 📋 목차

1. 운영체제 설치
2. ROS 설치
3. MoveIt 설치
4. ROS 작업공간 설정
5. ROS 패키지 생성
6. 선택 옵션
7. 그래픽 드라이버 및 CUDA 설치
8. H2017 ROS 패키지 다운로드
9. bashrc 설정
10. PyTorch 및 CUDA 확인

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

### 1️⃣ 그래픽 드라이버 설치

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

### 2️⃣ CUDA 설치 (11.8 혹은 12.1 설치 권장)

1. [GPU Driver와 CUDA 버전 호환성 확인](https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html#id4).
2. 아래 명령어를 실행하여 nvidia cuda toolkit을 설치합니다.

    ```bash
    sudo apt install nvidia-cuda-toolkit
    ```

3. [CUDA 설치 가이드](https://developer.nvidia.com/cuda-toolkit-archive)를 참고하여 설치합니다.

4. 설치 옵션 중 "runfile (local)"을 추천하며, runfile 다운로드 후 실행 전 아래 명령어로 실행 권한을 부여합니다.

    ```bash
    chmod 777 <runfile>
    ```

5. runfile을 실행 후 설치를 진행합니다. 설치 옵션 중 Driver 설치 옵션은 체크 해제합니다.

6. 설치가 완료되면 아래 명령어로 CUDA 설치를 확인합니다.
    ```bash
    nvcc -V
    ```

### 3️⃣ cuDNN 설치

1. [cuDNN 버전 호환성 확인](https://en.wikipedia.org/wiki/CUDA#GPUs_supported).

2. [cuDNN 설치 가이드](https://developer.nvidia.com/rdp/cudnn-archive)를 참고하여 설치하세요.

3. "Local Installer for Ubuntu20.04 x86_64 (Deb)"와 같은 deb 형식의 파일을 다운로드합니다.

4. 시스템 패키지 업데이트를 진행합니다.

    ```bash
    sudo apt update
    ```

5. 만약 에러가 발생하면, 아래 명령어로 sources.list.d에서 CUDA 및 cuDNN 소스 목록을 삭제합니다.

    ```bash
    sudo rm /etc/apt/sources.list.d/cuda*
    sudo rm /etc/apt/sources.list.d/cudnn*
    ```


## 🦾 H2017 ROS 패키지 다운로드

1. [패키지 다운로드](https://drive.google.com/file/d/1WegURoWhEUYmu9ryMMx_myGjUmXnEEFA/view?usp=drive_link) 후, 패키지를 `~/catkin_ws/src`에 넣습니다.

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
export PATH=/usr/local/cuda-<CUDA version>/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-<CUDA version>/lib64:$LD_LIBRARY_PATH
source /opt/ros/noetic/setup.bash

alias python=python3
alias pip=pip3

source /opt/ros/noetic/setup.bash
source ~/catkin_ws/devel/setup.bash

alias sb="source ~/.bashrc"
alias cm="catkin_make & source ./devel/setup.bash"
alias rc='rosclean purge -y'
alias run='rosclean purge -y & roslaunch h2017 demo_gazebo.launch'

# input your IP
# default PORT : 11311
export ROS_MASTER_URI=http://<IP>:11311
export ROS_HOSTNAME=<IP>
# default PORT : 11345
export GAZEBO_MASTER_URI=http://<IP>:11345
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
