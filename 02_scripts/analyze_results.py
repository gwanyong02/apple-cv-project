import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

# --- [수정된 부분: 한글 폰트 설정] ---
# Colab 환경에서 한글이 깨지지 않도록 나눔고딕 폰트를 설정합니다.
# 위 1, 2단계를 먼저 실행해야 이 코드가 정상적으로 작동합니다.
try:
    plt.rc('font', family='NanumGothic')
except:
    print("나눔고딕 폰트가 설치되어 있지 않습니다. Colab에서 '!sudo apt-get install -y fonts-nanum'을 실행하고 런타임을 다시 시작해주세요.")
# --- [수정 끝] ---


# --- [공통 설정 영역] ---
RESULTS_DIR = '04_results'
CSV_FILENAME = 'performance_summary_all.csv' 
CSV_PATH = os.path.join(RESULTS_DIR, CSV_FILENAME)
OUTPUT_GRAPH_DIR = RESULTS_DIR
# ---------------------------------------------------

def analyze_and_create_final_graphs():
    """
    모든 실행 결과가 담긴 CSV 파일을 읽어, 모델별 평균 성능을 계산하고
    비교 그래프를 생성합니다.
    """
    try:
        all_results_df = pd.read_csv(CSV_PATH)
    except FileNotFoundError:
        print(f"오류: '{CSV_PATH}' 파일을 찾을 수 없습니다.")
        print("1. 04_results 폴더에 performance_summary_all.csv 파일이 있는지 확인해주세요.")
        print("2. clean_results.py를 실행하여 파일을 생성했는지 확인해주세요.")
        return
    except Exception as e:
        print(f"CSV 파일을 읽는 중 오류 발생: {e}")
        return

    # 모델별로 모든 지표의 평균(mean)과 표준편차(std) 계산
    final_summary = all_results_df.groupby('Model').agg(['mean', 'std'])
    
    print("--- 최종 모델별 평균 성능 및 표준편차 요약 ---")
    print(final_summary)
    
    # 그래프 생성을 위해 평균값만 사용
    df_mean = final_summary.xs('mean', axis=1, level=1).reset_index()

    # 정확도 관련 지표 비교 그래프 생성
    plt.style.use('seaborn-v0_8-talk')
    accuracy_metrics = ['mAP', 'F1-Score', 'Precision', 'Recall']
    df_long = df_mean.melt(id_vars='Model', value_vars=accuracy_metrics, var_name='Metric', value_name='Score')

    plt.figure(figsize=(16, 9))
    ax = sns.barplot(data=df_long, x='Metric', y='Score', hue='Model', palette='viridis')
    
    plt.title('모델별 평균 정확도 성능 비교 (3회 실행 평균)', fontsize=22, pad=20)
    plt.xlabel('평가 지표', fontsize=16)
    plt.ylabel('평균 점수 (Mean Score)', fontsize=16)
    plt.ylim(0, 1.1)
    plt.legend(title='Model', fontsize=14)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    for p in ax.patches:
        ax.annotate(f'{p.get_height():.3f}', (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='center', fontsize=12, color='black', xytext=(0, 10),
                    textcoords='offset points')

    accuracy_graph_path = os.path.join(OUTPUT_GRAPH_DIR, 'final_accuracy_comparison.png')
    plt.savefig(accuracy_graph_path, dpi=300, bbox_inches='tight')
    print(f"\n최종 정확도 비교 그래프가 '{accuracy_graph_path}'에 저장되었습니다.")
    plt.close()

    # 속도(FPS) 비교 그래프 생성
    plt.figure(figsize=(12, 7))
    ax_fps = sns.barplot(data=df_mean, x='Model', y='FPS', palette='rocket')

    plt.title('모델별 평균 추론 속도 비교 (FPS)', fontsize=22, pad=20)
    plt.xlabel('모델', fontsize=16)
    plt.ylabel('평균 초당 프레임 수 (Mean FPS)', fontsize=16)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    for index, row in df_mean.iterrows():
        plt.text(index, row['FPS'], f"{row['FPS']:.1f}", ha='center', va='bottom', fontsize=14)

    speed_graph_path = os.path.join(OUTPUT_GRAPH_DIR, 'final_speed_comparison.png')
    plt.savefig(speed_graph_path, dpi=300, bbox_inches='tight')
    print(f"최종 속도 비교 그래프가 '{speed_graph_path}'에 저장되었습니다.")
    plt.close()

if __name__ == '__main__':
    analyze_and_create_final_graphs()
