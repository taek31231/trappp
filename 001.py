import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import time

# --- 1. 상수 정의 (스케일링 필요) ---
SCALED_MASS = 1.0 # 전자의 질량 (단위 조정)
SCALED_HBAR = 1.0 # 디랙 상수 (단위 조정)

st.set_page_config(layout="wide")
st.title("양자 터널링 시뮬레이터: 유한 퍼텐셜 장벽")
st.write("""
유한한 퍼텐셜 장벽에 충돌하는 자유 전자의 파동함수와 확률 밀도의 시간에 따른 변화를 시뮬레이션합니다.
전자의 에너지($E$)와 장벽 높이($V_0$)를 비교하여 터널링 현상을 관찰해보세요!
""")

# --- 2. 사이드바 입력 위젯 ---
st.sidebar.header("장벽 및 전자 설정")
V0 = st.sidebar.slider("장벽 높이 (V₀, 임의 단위)", 0.5, 10.0, 5.0, 0.1)
L = st.sidebar.slider("장벽 폭 (L, 임의 단위)", 0.1, 5.0, 1.0, 0.1)
E_electron = st.sidebar.slider("전자 에너지 (E, 임의 단위)", 0.1, 12.0, 3.0, 0.1) # V0보다 낮거나 높게 설정 가능

st.sidebar.markdown(f"**현재 E {E_electron:.1f} vs V₀ {V0:.1f}**")
if E_electron < V0:
    st.sidebar.info("E < V₀: 터널링 현상 관찰")
else:
    st.sidebar.info("E > V₀: 장벽 투과 및 부분 반사 관찰")

# 시뮬레이션 시간 및 속도
total_time_range = st.sidebar.slider("총 시뮬레이션 시간 범위 (임의 단위)", 1.0, 100.0, 50.0, 1.0)
time_steps = st.sidebar.slider("시간 단계 수", 50, 500, 200, 10)
animation_speed = st.sidebar.slider("애니메이션 속도 (s/프레임)", 0.01, 0.2, 0.05, 0.01)

# --- 3. 양자 역학 계산 함수 (핵심 로직) ---

# 경계 조건 및 파동함수 계산을 위한 함수 (이 부분이 가장 복잡하며 상세한 구현 필요)
# E < V0 (터널링)과 E > V0 (초과) 두 경우를 모두 처리해야 함.
@st.cache_data
def solve_barrier_problem(E, V0, L_val, m, hbar):
    """
    유한 퍼텐셜 장벽 문제의 정상 상태 파동함수 계수와 T, R 값을 계산합니다.
    이 함수는 실제 양자 역학 교재를 참고하여 상세히 구현해야 합니다.
    반환 값: (A, B, C, D, F), T, R
    여기서 A는 입사파 진폭 (보통 1), B는 영역 I의 반사파 진폭,
    C, D는 영역 II의 파동함수 계수, F는 영역 III의 투과파 진폭.
    """
    # 이 부분에 슈뢰딩거 방정식 해법 구현
    # E < V0 인 경우와 E >= V0 인 경우를 나누어 처리
    # 예시 (매우 단순화): 실제 계산은 복잡합니다.
    try:
        if E < V0:
            # 터널링 (E < V0)
            k1 = np.sqrt(2 * m * E) / hbar
            kappa = np.sqrt(2 * m * (V0 - E)) / hbar
            
            # 전송 계수 T 계산 (투과 확률)
            # T = 1 / (1 + (V0**2 * np.sinh(kappa * L_val)**2) / (4 * E * (V0 - E)))
            # 위 공식은 잘 알려진 유한 장벽의 T 계수. 여기서는 임시 값 사용.
            T_val = 1.0 / (1.0 + (V0**2 * np.sinh(kappa * L_val)**2) / (4 * E * (V0 - E) + 1e-9)) # +1e-9는 0 나눗셈 방지
            
            R_val = 1 - T_val

            # 실제 파동함수 계수 A, B, C, D, F 등을 계산하는 로직은 훨씬 더 복잡합니다.
            # 여기서는 단순히 T와 R만 반환하고 파동함수는 임의로 가정합니다.
            # 이 부분을 제대로 구현해야 합니다.
            
        else:
            # 장벽 투과 (E >= V0)
            k1 = np.sqrt(2 * m * E) / hbar
            k2 = np.sqrt(2 * m * (E - V0)) / hbar
            
            # 전송 계수 T 계산
            # T = 1 / (1 + (V0**2 * np.sin(k2 * L_val)**2) / (4 * E * (E - V0)))
            T_val = 1.0 / (1.0 + (V0**2 * np.sin(k2 * L_val)**2) / (4 * E * (E - V0) + 1e-9))
            
            R_val = 1 - T_val
        
        # 임시 반환값 (실제로는 A, B, C, D, F 등을 반환해야 함)
        return {"T": T_val, "R": R_val}
    except ZeroDivisionError:
        st.error("오류: 입력 값이 계산 범위를 벗어났습니다. 장벽 높이와 전자 에너지를 확인해주세요.")
        return {"T": 0, "R": 1} # 오류 시 임시 값

# x 위치에 따른 파동함수 값 계산 (정상 상태 파동함수 기반)
def get_psi_at_x(x, E, V0, L_val, m, hbar):
    """
    정상 상태 파동함수 psi(x)를 계산합니다.
    이 함수도 경계 조건을 이용하여 각 영역의 파동함수를 정의해야 합니다.
    파동 묶음 시뮬레이션이 아닌 단일 에너지 상태의 파동함수 형태를 가정합니다.
    """
    # 이 부분에 슈뢰딩거 방정식 해의 실제 형태를 구현
    # 영역 I, II, III에 대한 서로 다른 파동함수 형태를 정의하고 경계 조건을 통해 계수를 결정해야 함
    
    # 임시 구현: x에 따른 사인/코사인 또는 지수 함수 형태 (개념적)
    # 이 부분은 E, V0, L에 따라 정확한 해를 기반으로 작성되어야 합니다.
    
    # 실제로는 복소수 값의 psi(x)를 반환해야 합니다.
    # 여기서는 간단한 예시로 특정 n=1 상태와 유사하게 나타내봅니다.
    # 실제 터널링 시뮬레이션은 각 영역의 파동함수 계수를 정확히 구해야 합니다.
    
    # 매우 단순화된 형태 (이 부분은 실제 물리 공식을 기반으로 대체되어야 합니다.)
    k = np.sqrt(2 * m * E) / hbar
    if x < 0: # 영역 I (입사파 + 반사파)
        return np.exp(1j * k * x) + 0.5 * np.exp(-1j * k * x) # 0.5는 임의의 반사파 계수
    elif x <= L_val: # 영역 II (감쇠 또는 진동)
        if E < V0:
            kappa = np.sqrt(2 * m * (V0 - E)) / hbar
            return np.exp(-kappa * x) * np.exp(1j * k * x) # 임의의 감쇠파
        else:
            k_prime = np.sqrt(2 * m * (E - V0)) / hbar
            return np.sin(k_prime * x) * np.exp(1j * k * x) # 임의의 진동파
    else: # 영역 III (투과파)
        return 0.8 * np.exp(1j * k * x) # 0.8은 임의의 투과파 계수

# 시간에 따른 파동함수 Psi(x,t)
def get_psi_xt(x_vals, t_val, E, V0, L_val, m, hbar):
    """
    시간 t에서의 파동함수 Psi(x, t)를 계산합니다.
    """
    psi_static = np.array([get_psi_at_x(x, E, V0, L_val, m, hbar) for x in x_vals], dtype=complex)
    time_evolution_term = np.exp(-1j * E * t_val / hbar)
    return psi_static * time_evolution_term

# --- 4. 시뮬레이션 실행 및 그래프 그리기 ---
if st.sidebar.button("시뮬레이션 시작"):
    x_coords = np.linspace(-L*2, L*3, 500) # 우물 밖 영역까지 포함
    
    # 퍼텐셜 장벽 그리기 위한 데이터
    potential_x = np.array([x_coords[0], 0, 0, L, L, x_coords[-1]])
    potential_y = np.array([0, 0, V0, V0, 0, 0])

    # T, R 값 계산 (개념 구현에서 임시 값)
    results = solve_barrier_problem(E_electron, V0, L, SCALED_MASS, SCALED_HBAR)
    T = results["T"]
    R = results["R"]
    
    st.markdown(f"### 전송 계수 (투과 확률): **{T:.4f}**")
    st.markdown(f"### 반사 계수 (반사 확률): **{R:.4f}**")

    # 그래프를 업데이트할 placeholder 생성
    chart_placeholder = st.empty()

    for i in range(time_steps):
        current_time = total_time_range * (i / (time_steps - 1))
        
        # 현재 시간의 파동함수 계산
        psi_at_t = get_psi_xt(x_coords, current_time, E_electron, V0, L, SCALED_MASS, SCALED_HBAR)
        
        # 확률 밀도 계산 (|Psi|^2)
        prob_density = np.abs(psi_at_t)**2

        # 그래프 그리기
        fig, ax = plt.subplots(figsize=(12, 7))
        
        # 퍼텐셜 장벽 그리기
        ax.plot(potential_x, potential_y, color='gray', linestyle='--', label='퍼텐셜 장벽 ($V(x)$)', linewidth=2)
        ax.axhline(E_electron, color='purple', linestyle=':', label='전자 에너지 (E)')

        ax.plot(x_coords, psi_at_t.real, label='$\Psi(x,t)$ 실수부', color='blue')
        ax.plot(x_coords, psi_at_t.imag, label='$\Psi(x,t)$ 허수부', color='red', linestyle='--')
        ax.plot(x_coords, prob_density, label='$|\Psi(x,t)|^2$ (확률 밀도)', color='green', linewidth=2)
        
        ax.set_title(f"시간: {current_time:.2f} (임의 단위)")
        ax.set_xlabel("위치 x")
        ax.set_ylabel("파동함수 값 / 퍼텐셜 에너지")
        ax.set_xlim(x_coords[0], x_coords[-1])
        
        # y축 범위는 파동함수 스케일과 V0 스케일에 따라 조정
        max_val_psi = np.max([np.max(np.abs(psi_at_t.real)), np.max(np.abs(psi_at_t.imag)), np.max(prob_density)])
        max_y_lim = max(V0 * 1.2, E_electron * 1.2, max_val_psi * 1.5)
        min_y_lim = min(0, -max_val_psi * 1.5)
        ax.set_ylim(min_y_lim, max_y_lim)
        
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
    * **$E < V_0$ 일 때**: 전자가 장벽을 넘어가지 못해야 고전적으로는 불가능하지만, 장벽 반대편에서도 확률이 0이 아닌 것을 확인해 보세요 (터널링).
    * **$E > V_0$ 일 때**: 전자가 장벽을 넘어가지만, 완벽하게 넘어가는 것이 아니라 일부는 반사되는 것을 확인해 보세요.
    * **장벽의 폭(L)**을 늘리면 터널링 확률이 어떻게 변하는지 관찰해 보세요.
    """)
