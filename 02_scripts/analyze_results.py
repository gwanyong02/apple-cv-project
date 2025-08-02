import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

# --- [공통 설정 영역] ---
# 모든 팀원이 동일하게 사용하는 설정입니다.

# 결과 CSV 파일이 저장된 경로
# 이 파일은 모든 팀원이 자신의 모델 평가 결과를 추가하는 공유 파일입니다.
RESULTS_DIR = '04_results'
CSV_FILENAME = 'performance_summary.csv'
CSV_PATH = os.path.join(RESULTS_DIR, CSV_FILENAME)

# 생성된 그래프를 저장할 경로
# PNG 파일로 저장됩니다.
OUTPUT_GRAPH_DIR = RESULTS_DIR

# --- [주의사항: 변수명 및 데이터 형식 통일의 중요성] ---
#
# 이 스크립트가 올바르게 작동하려면, 모든 팀원이 `performance_summary.csv` 파일에
# 결과를 기록할 때 아래의 '컬럼명'을 **정확히 동일하게** 사용해야 합니다.
#
# 예시: 'Model', 'Precision', 'Recall', 'F1-Score', 'mAP', 'FPS'
#
# 변수명 반드시 통일!
#
# ---

def create_graphs(df):
    """
    데이터프레임을 받아 성능 비교 그래프를 생성하고 저장합니다.
    """
    if df.empty:
        print(f"오류: {CSV_PATH} 파일이 비어있습니다. 데이터를 채워주세요.")
        return

    # 1. 정확도 관련 지표 비교 그래프 생성 (mAP, F1-Score, Precision, Recall)
    plt.style.use('seaborn-v0_8-talk') # 그래프 스타일 설정
    
    accuracy_metrics = ['mAP', 'F1-Score', 'Precision', 'Recall']
    
    # 데이터프레임을 시각화에 용이한 'long' 형태로 변환
    df_long = df.melt(id_vars='Model', value_vars=accuracy_metrics, var_name='Metric', value_name='Score')

    plt.figure(figsize=(14, 8))
    sns.barplot(data=df_long, x='Metric', y='Score', hue='Model', palette='viridis')
    
    plt.title('모델별 정확도 성능 비교', fontsize=20, pad=20)
    plt.xlabel('평가 지표', fontsize=14)
    plt.ylabel('점수 (Score)', fontsize=14)
    plt.ylim(0, 1.1) # y축 범위를 0~1.1로 설정하여 비교 용이
    plt.legend(title='Model', fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # 그래프 위에 실제 값 표시
    for p in plt.gca().patches:
        plt.gca().annotate(f'{p.get_height():.3f}', (p.get_x() + p.get_width() / 2., p.get_height()),
                           ha='center', va='center', fontsize=11, color='black', xytext=(0, 10),
                           textcoords='offset points')

    accuracy_graph_path = os.path.join(OUTPUT_GRAPH_DIR, 'accuracy_comparison.png')
    plt.savefig(accuracy_graph_path)
    print(f"정확도 비교 그래프가 '{accuracy_graph_path}'에 저장되었습니다.")
    plt.close()


    # 2. 속도(FPS) 비교 그래프 생성
    plt.figure(figsize=(10, 6))
    sns.barplot(data=df, x='Model', y='FPS', palette='rocket')

    plt.title('모델별 추론 속도 비교 (FPS)', fontsize=20, pad=20)
    plt.xlabel('모델', fontsize=14)
    plt.ylabel('초당 프레임 수 (Frames Per Second)', fontsize=14)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # 그래프 위에 실제 값 표시
    for index, value in enumerate(df['FPS']):
        plt.text(index, value + 0.5, f'{value:.1f}', ha='center', va='bottom', fontsize=12)

    speed_graph_path = os.path.join(OUTPUT_GRAPH_DIR, 'speed_comparison.png')
    plt.savefig(speed_graph_path)
    print(f"속도 비교 그래프가 '{speed_graph_path}'에 저장되었습니다.")
    plt.close()


if __name__ == '__main__':
    # 이 스크립트를 직접 실행하면, CSV 파일을 읽어 그래프를 생성합니다.
    
    # 필요한 라이브러리 설치
    # pip install pandas seaborn matplotlib
    
    try:
        # 1. 공유된 CSV 파일 읽기
        performance_df = pd.read_csv(CSV_PATH)
        
        # 2. 그래프 생성 함수 호출
        create_graphs(performance_df)
        
    except FileNotFoundError:
        print(f"오류: '{CSV_PATH}' 파일을 찾을 수 없습니다.")
        print("먼저 각 모델의 성능 평가를 완료하고, 합의된 형식으로 CSV 파일을 생성해주세요.")
        # 예시 CSV 파일 생성
        example_data = {
            'Model': ['YOLOv8', 'YOLOv11', 'Faster R-CNN'],
            'Precision': [0.0, 0.0, 0.0],
            'Recall': [0.0, 0.0, 0.0],
            'F1-Score': [0.0, 0.0, 0.0],
            'mAP': [0.0, 0.0, 0.0],
            'FPS': [0.0, 0.0, 0.0]
        }
        pd.DataFrame(example_data).to_csv(CSV_PATH, index=False)
        print(f"'{CSV_PATH}'에 예시 파일을 생성했습니다. 이 파일에 각자 결과를 채워주세요.")

