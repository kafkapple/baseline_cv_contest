#!/bin/bash

# 스크립트 실행 중 오류 발생 시 즉시 종료
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


#chmod +x setup_conda.sh
# source setup_conda.sh