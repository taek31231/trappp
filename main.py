import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import time

# --- 1. 기본 상수 및 우물 정의 ---
L_default = 1.0 # 우물 폭 (단위: nm)
SCALED_MASS = 1.0 # m_e
SCALED_HBAR = 1.0 # hbar

st.set_page_config(layout="wide")
st.title("1차원 무한 퍼텐셜 우물 내 전자의 파동함수 진화")
st.write("시간에 따른 전자의 파동함수($\Psi(x, t)$)와 확률 밀도($|\Psi(x, t)|^2$)를 시각화합니다.")

# --- 2. 사이드바에 입력 위젯 배치 ---
st.sidebar.header("우물 및 시뮬레이션 설정")
L = st.sidebar.slider("우물 폭 (L)", 0.5, 5.0, L_default, 0.1)
initial_state_type = st.sidebar.selectbox(
    "초기 상태 선택",
    ["단일 고유 상태", "두 고유 상태 중첩"]
)

if initial_state_type == "단일 고유 상태":
    n_state = st.sidebar.slider("주양자수 (n)", 1, 5, 1)
    n_states_to_consider = [n_state]
    coefficients = [1.0] 
else: # 두 고유 상태 중첩
    n1 = st.sidebar.slider("첫 번째 주양자수 (n1)", 1, 5, 1)
    n2 = st.sidebar.slider("두 번째 주양자수 (n2)", 1, 5, 2)
    if n1 == n2:
        st.sidebar.warning("두 주양자수는 달라야 합니다. 첫 번째 양자수로 설정됩니다.")
        n_states_to_consider = [n1]
        coefficients = [1.0]
    else:
        n_states_to_consider = [n1, n2]
        coefficients = [1.0, 1.0]

total_time = st.sidebar.slider("총 시뮬레이션 시간 (단위: 임의)", 1.0, 100.0, 20.0, 1.0)
time_steps = st.sidebar.slider("시간 단계 수", 50, 500, 200, 10)
animation_speed = st.sidebar.slider("애니메이션 속도 (s/프레임)", 0.01, 0.5, 0.05, 0.01)

# --- 3. 양자역학 함수 정의 ---

# 이 함수는 고유 함수 자체를 반환하지 않고,
# 고유 에너지 값만 캐싱하도록 수정합니다.
@st.cache_data
def get_eigen_energies(n_max, L_val, m_val, hbar_val):
    """
    고유 에너지 목록을 반환합니다.
    """
    energies = {}
    for n in range(1, n_max + 1):
        energies[n] = (n**2 * np.pi**2 * hbar_val**2) / (2 * m_val * L_val**2)
    return energies

# 고유 함수는 캐싱하지 않고 필요할 때 직접 계산하도록 합니다.
# 이는 함수 객체를 캐싱하려 할 때 발생하는 문제를 피하기 위함입니다.
def calculate_phi_n(x, n, L_val):
    """
    n번째 고유 함수를 계산합니다.
    """
    return np.sqrt(2/L_val) * np.sin(n * np.pi * x / L_val)

# 초기 파동함수 계산 및 정규화
def get_initial_psi(x_vals, n_states, coeffs, L_val):
    """
    초기 파동함수 Psi(x, 0)를 계산하고 정규화합니다.
    """
    initial_psi = np.zeros_like(x_vals, dtype=complex)
    
    # 계수를 제곱합이 1이 되도록 정규화
    sum_of_squares = sum([c**2 for c in coeffs])
    if sum_of_squares == 0: # 모든 계수가 0일 경우 (예외 처리)
        st.error("초기 상태 계수가 모두 0입니다. 유효한 계수를 설정해주세요.")
        return initial_psi, [0.0] * len(coeffs)

    normalized_coeffs = [c / np.sqrt(sum_of_squares) for c in coeffs]

    for i, n in enumerate(n_states):
        initial_psi += normalized_coeffs[i] * calculate_phi_n(x_vals, n, L_val)
    return initial_psi, normalized_coeffs

# 시간 t에서의 파동함수 계산
def get_psi_xt(x_vals, t_val, L_val, m_val, hbar_val, n_states, normalized_coeffs, energies_dict):
    """
    시간 t에서의 파동함수 Psi(x, t)를 계산합니다.
    """
    psi_xt = np.zeros_like(x_vals, dtype=complex)
    for i, n in enumerate(n_states):
        phi_n_val = calculate_phi_n(x_vals, n, L_val) # 매번 계산
        E_n_val = energies_dict[n] # 캐싱된 에너지 사용
        time_evolution_term = np.exp(-1j * E_n_val * t_val / hbar_val)
        psi_xt += normalized_coeffs[i] * phi_n_val * time_evolution_term
    return psi_xt

# --- 4. 시뮬레이션 실행 및 그래프 그리기 ---
if st.sidebar.button("시뮬레이션 시작"):
    x_coords = np.linspace(0, L, 500)
    
    # 에너지 값만 캐싱하여 가져옴
    max_n_for_energy = max(n_states_to_consider) if n_states_to_consider else 1
    energies_dict = get_eigen_energies(max_n_for_energy, L, SCALED_MASS, SCALED_HBAR)

    # 초기 파동함수 계산 및 계수 정규화
    initial_psi_at_x, final_coeffs = get_initial_psi(x_coords, n_states_to_consider, coefficients, L)

    # 그래프를 업데이트할 placeholder 생성
    chart_placeholder = st.empty()

    for i in range(time_steps):
        current_time = total_time * (i / (time_steps - 1))
        
        # 현재 시간의 파동함수 계산
        psi_at_t = get_psi_xt(x_coords, current_time, L, SCALED_MASS, SCALED_HBAR, n_states_to_consider, final_coeffs, energies_dict)
        
        # 확률 밀도 계산 (|Psi|^2)
        prob_density = np.abs(psi_at_t)**2

        # 그래프 그리기
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(x_coords, psi_at_t.real, label='$\Psi(x,t)$ 실수부', color='blue')
        ax.plot(x_coords, psi_at_t.imag, label='$\Psi(x,t)$ 허수부', color='red', linestyle='--')
        ax.plot(x_coords, prob_density, label='$|\Psi(x,t)|^2$ (확률 밀도)', color='green', linewidth=2)
        
        ax.set_title(f"시간: {current_time:.2f} (임의 단위)")
        ax.set_xlabel("위치 x")
        ax.set_ylabel("파동함수 값")
        ax.set_xlim(0, L)
        # y축 범위는 파동함수 스케일에 따라 조정
        max_psi_val = np.max([np.max(np.abs(psi_at_t.real)), np.max(np.abs(psi_at_t.imag)), np.max(prob_density)])
        ax.set_ylim(-max_psi_val * 1.2, max_psi_val * 1.2)
        ax.legend()
        ax.grid(True)

        chart_placeholder.pyplot(fig)
        plt.close(fig) # 메모리 누수 방지

        time.sleep(animation_speed)

    st.success("시뮬레이션 완료!")

else:
    st.info("왼쪽 사이드바에서 설정을 조절하고 '시뮬레이션 시작' 버튼을 눌러주세요.")
    st.markdown("---")
    st.markdown("""
    **팁:**
    * **단일 고유 상태**를 선택하면 파동함수의 모양은 변하지 않고 위상만 시간에 따라 변합니다 (확률 밀도는 변하지 않음).
    * **두 고유 상태 중첩**을 선택하면 파동함수와 확률 밀도 모두 시간에 따라 복잡하게 진화하는 것을 볼 수 있습니다.
    * `L` 값은 우물의 크기를 조절하며, 이는 에너지 준위와 파동함수의 모양에 영향을 줍니다.
    """)
