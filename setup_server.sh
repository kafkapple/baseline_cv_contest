#!/bin/bash

# 스크립트 실행 중 오류 발생 시 즉시 종료
set -e

# 함수: 오류 메시지 출력 후 종료
function error_exit {
    echo "에러: $1" >&2
    exit 1
}

# 루트 권한 확인
if [ "$EUID" -ne 0 ]; then
    error_exit "이 스크립트는 루트 권한으로 실행되어야 합니다. 'sudo ./setup_server.sh' 형식으로 실행해주세요."
fi

echo "=============================="
echo "  기본 Linux 서버 설정 시작  "
echo "=============================="

# 시스템 패키지 목록 업데이트
echo "시스템 패키지 목록 업데이트 중..."
apt update -y || error_exit "apt update 실패"

# Git 설치
echo "Git 설치 중..."
apt install -y git || error_exit "Git 설치 실패"

# 기본 JDK 설치
echo "기본 JDK 설치 중..."
apt install -y default-jdk || error_exit "default-jdk 설치 실패"

# vim 설치
echo "vim 설치 중..."
apt install -y vim || echo "vim 설치 실패. nano 에디터를 사용할 것입니다."

# 설치된 에디터 확인 및 설정
if command -v vim &> /dev/null
then
    GIT_EDITOR="vim"
    echo "vim이 설치되어 있어 Git 기본 에디터로 설정됩니다."
else
    echo "vim이 설치되지 않았습니다. nano를 Git 기본 에디터로 설정합니다."
    GIT_EDITOR="nano"
    apt install -y nano || error_exit "nano 설치 실패"
fi

echo "기본 패키지 설치 완료."
echo ""

echo "=============================="
echo "        Conda 설정 시작        "
echo "=============================="

# 현재 사용자를 SUDO_USER로 설정 (sudo를 사용하지 않고 직접 실행된 경우 root 유지)
if [ -n "$SUDO_USER" ]; then
    CURRENT_USER="$SUDO_USER"
else
    CURRENT_USER="root"
fi

echo "Conda 설정을 위해 사용자 '$CURRENT_USER'로 전환합니다."

# Conda 설정 스크립트 내용
CONDA_SETUP=$(cat <<'EOF'
# 오류 발생 시 즉시 종료
set -e

# conda 실행 파일 경로 설정
CONDA_PATH="/opt/conda/bin/conda"

# conda 실행 파일 존재 여부 확인
if [ ! -f "$CONDA_PATH" ]; then
    echo "에러: $CONDA_PATH 에서 conda를 찾을 수 없습니다."
    exit 1
fi

# conda 초기화
echo "Conda 초기화 중..."
$CONDA_PATH init

# bash 셸용 conda 설정 적용
echo "Conda 셸 훅 적용 중..."
eval "$($CONDA_PATH shell.bash hook)"

# .bashrc 파일 소스
echo ".bashrc 파일을 소싱 중..."
source ~/.bashrc

echo "Conda가 성공적으로 초기화되었습니다."
EOF
)

# 임시 스크립트 파일 생성
TEMP_SCRIPT="/tmp/setup_conda_temp.sh"
echo "$CONDA_SETUP" > "$TEMP_SCRIPT"
chmod +x "$TEMP_SCRIPT"

# 사용자 권한으로 스크립트 실행
if [ "$CURRENT_USER" != "root" ]; then
    sudo -u "$CURRENT_USER" bash "$TEMP_SCRIPT" || error_exit "Conda 설정 실패"
else
    bash "$TEMP_SCRIPT" || error_exit "Conda 설정 실패"
fi

# 임시 스크립트 삭제
rm -f "$TEMP_SCRIPT"

echo "=============================="
echo "        Conda 설정 완료       "
echo "=============================="
echo ""

echo "=============================="
echo "        Git 설정 시작         "
echo "=============================="

# Git 설정을 위한 사용자 권한 스크립트
GIT_SETUP=$(cat <<EOF
# 오류 발생 시 즉시 종료
set -e

# Git 글로벌 설정
echo "Git 글로벌 설정을 시작합니다."

# Git 사용자 이름 입력
read -p "Git 사용자 이름을 입력하세요: " GIT_USERNAME

# Git 사용자 이메일 입력
read -p "Git 사용자 이메일을 입력하세요: " GIT_USEREMAIL

# Personal Access Token (PAT) 입력
read -s -p "GitHub Personal Access Token (PAT)을 입력하세요: " PAT
echo # 새 줄 추가

# Repository URL 입력
read -p "클론할 리포지토리의 URL을 입력하세요 (예: https://github.com/username/repo.git): " REPO_URL

# PAT를 사용하여 리포지토리 클론
echo "리포지토리를 클론하는 중..."
git clone https://\$PAT@\${REPO_URL#https://} || { echo "리포지토리 클론 실패"; exit 1; }

# Git 글로벌 설정
echo "Git 글로벌 설정을 적용 중..."
git config --global user.name "${GIT_USERNAME}"
git config --global user.email "${GIT_USEREMAIL}"
git config --global core.editor "$GIT_EDITOR"
git config --global core.pager cat
git config --global alias.lg "log --color --graph --pretty=format:'%Cred%h%Creset -%C(yellow)%d%Creset %s %Cgreen(%cr) %C(bold blue)<%an>%Creset' --abbrev-commit"

# 자격 증명 도우미 설정
echo "Git 자격 증명 도우미를 설정 중..."
git config --global credential.helper store

# 업스트림 리모트 설정 여부 확인
read -p "업스트림 리모트를 설정하시겠습니까? (y/n): " CONFIGURE_UPSTREAM
if [[ "\$CONFIGURE_UPSTREAM" == "y" ]]; then
    read -p "업스트림 리포지토리의 URL을 입력하세요 (HTTPS 또는 SSH): " UPSTREAM_URL
    git remote set-url upstream "\$UPSTREAM_URL" || { echo "업스트림 리모트 설정 실패"; exit 1; }
    echo "업스트림 리모트가 설정되었습니다."
fi

# Git 글로벌 설정 확인
echo "현재 Git 글로벌 설정을 표시합니다:"
git config --list --show-origin

# Git 설정 초기화 여부 확인
read -p "Git 설정을 초기화하시겠습니까? (y/n): " RESET_CONFIG
if [[ "\$RESET_CONFIG" == "y" ]]; then
    if [[ "\$OSTYPE" == "linux-gnu"* || "\$OSTYPE" == "darwin"* ]]; then
        rm -f ~/.gitconfig
        echo "Git 설정이 초기화되었습니다 (Linux/Mac)."
    elif [[ "\$OSTYPE" == "msys"* || "\$OSTYPE" == "win32" ]]; then
        del %USERPROFILE%\\.gitconfig
        echo "Git 설정이 초기화되었습니다 (Windows)."
    else
        echo "알 수 없는 OS 유형입니다. Git 설정 초기화를 수동으로 수행하세요."
    fi
fi

echo "Git 설정이 완료되었습니다!"
EOF
)

# 임시 Git 설정 스크립트 생성
GIT_TEMP_SCRIPT="/tmp/setup_git_temp.sh"
echo "$GIT_SETUP" > "$GIT_TEMP_SCRIPT"
chmod +x "$GIT_TEMP_SCRIPT"

# 사용자 권한으로 Git 설정 스크립트 실행
if [ "$CURRENT_USER" != "root" ]; then
    sudo -u "$CURRENT_USER" bash "$GIT_TEMP_SCRIPT" || error_exit "Git 설정 실패"
else
    bash "$GIT_TEMP_SCRIPT" || error_exit "Git 설정 실패"
fi

# 임시 Git 설정 스크립트 삭제
rm -f "$GIT_TEMP_SCRIPT"

echo "=============================="
echo "          Git 설정 완료        "
echo "=============================="

echo ""
echo "=============================="
echo "        서버 설정이 완료되었습니다.    "
echo "=============================="


#chmod +x setup_server.sh
# bash setup_server.sh