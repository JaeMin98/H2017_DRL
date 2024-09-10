import csv
import random

def add_noise(value, noise_level):
    """지정된 노이즈 수준으로 값에 노이즈를 추가합니다."""
    noise = random.uniform(-noise_level, noise_level)
    return value + value * noise

def generate_symmetric_data_with_noise(input_file, output_file, noise_level=0.05):
    # 원본 데이터 읽기
    with open(input_file, 'r', newline='') as csvfile:
        csv_reader = csv.reader(csvfile)
        data = list(csv_reader)

    # y=0 평면에 대한 대칭 데이터 생성 (노이즈 포함)
    symmetric_data = []
    for row in data:
        new_row = []
        for i, value in enumerate(row):
            if i < 3:  # x, y, z 좌표에만 노이즈 적용
                float_value = float(value)
                if i == 1:  # y 좌표만 부호 변경
                    float_value = -float_value
                noisy_value = add_noise(float_value, noise_level)
                new_row.append(str(noisy_value))
            else:
                new_row.append(value)
        symmetric_data.append(new_row)

    # 원본 데이터와 대칭 데이터 합치기
    combined_data = data + symmetric_data

    # 결과를 CSV 파일로 저장
    with open(output_file, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerows(combined_data)

    print(f"원본 데이터와 y=0 평면에 대해 대칭된 노이즈 추가 데이터가 {output_file}에 저장되었습니다.")

    # 결과 출력 (선택적)
    print("\n원본 및 y=0 평면 대칭 데이터(노이즈 추가)의 처음 몇 줄:")
    for row in combined_data[:5]:  # 처음 5줄만 출력
        print(','.join(row))

# 스크립트 실행
if __name__ == "__main__":
    input_file = "data_points_origin.csv"  # 입력 파일명
    output_file = "y_plane_symmetric_data_with_noise_output.csv"  # 출력 파일명
    noise_level = 0.05  # 노이즈 수준 (5%)
    generate_symmetric_data_with_noise(input_file, output_file, noise_level)