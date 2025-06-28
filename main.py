import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import time

# --- 1. 기본 상수 및 우물 정의 ---
# 실제 값 대신 시뮬레이션에 적합한 스케일로 조정할 수 있습니다.
# 예: L = 1 (nm), hbar = 1 (eV*fs), m = 1 (electron mass equivalent)
L_default = 1.0 # 우물 폭 (단위: nm)
m_electron = 9.109e-31 # 전자의 실제 질량 (kg)
hbar_val = 1.054e-34 # 디랙 상수 (J*s)

# 시뮬레이션 스케일을 위한 상수 (단위 변환 고려)
# 예를 들어, 에너지를 eV, 시간을 fs 로 나타내고 싶다면 hbar 값을 조정해야 합니다.
# 여기서는 일반적인 단위로 계산하고 그래프만 그립니다.
# E_n = n^2 * pi^2 * hbar^2 / (2 * m * L^2)
# 시간을 고려할 때, exp(-i * E_n * t / hbar)
# E_n / hbar 의 단위가 [1/시간] 이 되도록 단위를 맞춰야 합니다.
# 예시 코드에서는 계산을 단순화하기 위해 적절히 스케일링된 상수를 사용합니다.
# 실제 양자 역학 계산 시에는 단위 일관성에 매우 주의해야 합니다.

# 시뮬레이션에 사용할 스케일링된 상수 (예시)
# 만약 L=1(nm)이고 싶다면, m, hbar도 그에 맞게 조정하거나
# 에너지 단위를 맞춰야 합니다.
# 여기서는 간단히 L을 조절 가능한 파라미터로만 둡니다.
# 계산의 편의를 위해 E_n/hbar 의 단위를 임의로 맞춥니다.
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
    coefficients = [1.0] # 정규화는 나중에 수행

else: # 두 고유 상태 중첩
    n1 = st.sidebar.slider("첫 번째 주양자수 (n1)", 1, 5, 1)
    n2 = st.sidebar.slider("두 번째 주양자수 (n2)", 1, 5, 2)
    if n1 == n2:
        st.sidebar.warning("두 주양자수는 달라야 합니다.")
        n_states_to_consider = [n1]
        coefficients = [1.0]
    else:
        n_states_to_consider = [n1, n2]
        # 예시: 같은 비율로 중첩 (정규화 필요)
        # c1 = 1.0, c2 = 1.0 -> 정규화하면 c1=1/sqrt(2), c2=1/sqrt(2)
        coefficients = [1.0, 1.0]
        # 실제 계수 계산은 나중에 수행

# 시뮬레이션 시간 설정
total_time = st.sidebar.slider("총 시뮬레이션 시간 (단위: 임의)", 1.0, 100.0, 20.0, 1.0)
time_steps = st.sidebar.slider("시간 단계 수", 50, 500, 200, 10)
animation_speed = st.sidebar.slider("애니메이션 속도 (s/프레임)", 0.01, 0.5, 0.05, 0.01)

# --- 3. 양자역학 함수 정의 ---
@st.cache_data
def get_eigen_functions_and_energies(n_max, L_val, m_val, hbar_val):
    """
    고유 함수 및 고유 에너지 목록을 반환합니다.
    """
    phis = {}
    energies = {}
    for n in range(1, n_max + 1):
        # 고유 함수
        phis[n] = lambda x, n=n, L=L_val: np.sqrt(2/L) * np.sin(n * np.pi * x / L)
        # 고유 에너지
        energies[n] = (n**2 * np.pi**2 * hbar_val**2) / (2 * m_val * L_val**2)
    return phis, energies

phis, energies = get_eigen_functions_and_energies(
    max(n_states_to_consider) if n_states_to_consider else 1,
    L, SCALED_MASS, SCALED_HBAR
)

def get_initial_psi(x_vals, n_states, coeffs):
    """
    초기 파동함수 Psi(x, 0)를 계산하고 정규화합니다.
    """
    initial_psi = np.zeros_like(x_vals, dtype=complex)
    
    # 계수를 제곱합이 1이 되도록 정규화
    sum_of_squares = sum([c**2 for c in coeffs])
    normalized_coeffs = [c / np.sqrt(sum_of_squares) for c in coeffs]

    for i, n in enumerate(n_states):
        initial_psi += normalized_coeffs[i] * phis[n](x_vals)
    return initial_psi, normalized_coeffs

def get_psi_xt(x_vals, t_val, L_val, m_val, hbar_val, n_states, normalized_coeffs):
    """
    시간 t에서의 파동함수 Psi(x, t)를 계산합니다.
    """
    psi_xt = np.zeros_like(x_vals, dtype=complex)
    for i, n in enumerate(n_states):
        phi_n_val = phis[n](x_vals)
        E_n_val = energies[n]
        # 시간에 따른 위상 변화 항
        time_evolution_term = np.exp(-1j * E_n_val * t_val / hbar_val)
        psi_xt += normalized_coeffs[i] * phi_n_val * time_evolution_term
    return psi_xt

# --- 4. 시뮬레이션 실행 및 그래프 그리기 ---
if st.sidebar.button("시뮬레이션 시작"):
    x_coords = np.linspace(0, L, 500)
    
    # 초기 파동함수 계산 및 계수 정규화
    initial_psi_at_x, final_coeffs = get_initial_psi(x_coords, n_states_to_consider, coefficients)

    # 그래프를 업데이트할 placeholder 생성
    chart_placeholder = st.empty()

    for i in range(time_steps):
        current_time = total_time * (i / (time_steps - 1))
        
        # 현재 시간의 파동함수 계산
        psi_at_t = get_psi_xt(x_coords, current_time, L, SCALED_MASS, SCALED_HBAR, n_states_to_consider, final_coeffs)
        
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
