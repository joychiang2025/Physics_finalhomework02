# Physics_finalhomework02

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
from matplotlib.animation import FuncAnimation
import matplotlib.patches as patches
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import LinearSegmentedColormap
import warnings
warnings.filterwarnings('ignore')

# 設置中文字體
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

class RLCFieldAnalyzer:
    def __init__(self, R=100, L=0.001, C=1e-6):
        """
        RLC電路電磁場分析器
        
        參數:
        R: 電阻值 (歐姆)
        L: 電感值 (亨利)  
        C: 電容值 (法拉)
        """
        self.R = R
        self.L = L
        self.C = C
        self.frequency = 50  # Hz
        self.time = 0
        self.is_playing = False
        
        # 計算共振頻率
        self.f_resonance = 1 / (2 * np.pi * np.sqrt(self.L * self.C))
        
        # 創建空間網格
        self.setup_spatial_grids()
        
        # 創建圖形界面
        self.setup_figure()
        
        # 初始化動畫
        self.setup_animation()
        
        # 初始化顯示
        self.update_fields()
        
    def setup_spatial_grids(self):
        """設置空間網格"""
        # 電容器空間 (平行板電容器)
        self.cap_x = np.linspace(-0.02, 0.02, 50)  # 4cm寬
        self.cap_y = np.linspace(-0.01, 0.01, 30)  # 2cm高
        self.cap_z = np.linspace(0, 0.005, 20)     # 0.5cm板間距
        self.cap_X, self.cap_Y, self.cap_Z = np.meshgrid(self.cap_x, self.cap_y, self.cap_z)
        
        # 電感器空間 (螺線管)
        self.ind_r = np.linspace(0, 0.01, 25)      # 半徑1cm
        self.ind_theta = np.linspace(0, 2*np.pi, 40)  # 角度
        self.ind_z = np.linspace(-0.02, 0.02, 50)  # 長度4cm
        self.ind_R, self.ind_THETA, self.ind_Z = np.meshgrid(self.ind_r, self.ind_theta, self.ind_z)
        
        # 轉換為笛卡爾坐標
        self.ind_X = self.ind_R * np.cos(self.ind_THETA)
        self.ind_Y = self.ind_R * np.sin(self.ind_THETA)
        
    def setup_figure(self):
        """設置圖形界面"""
        self.fig = plt.figure(figsize=(18, 12))
        self.fig.suptitle('RLC電路電磁場動態分析器', fontsize=16, fontweight='bold')
        
        # 創建子圖
        self.ax_circuit = plt.subplot2grid((3, 4), (0, 0), rowspan=1, colspan=1)
        self.ax_cap_field = plt.subplot2grid((3, 4), (0, 1), rowspan=1, colspan=1)
        self.ax_ind_field = plt.subplot2grid((3, 4), (0, 2), rowspan=1, colspan=1)
        self.ax_energy = plt.subplot2grid((3, 4), (0, 3), rowspan=1, colspan=1)
        
        self.ax_cap_3d = plt.subplot2grid((3, 4), (1, 0), rowspan=1, colspan=2, projection='3d')
        self.ax_ind_3d = plt.subplot2grid((3, 4), (1, 2), rowspan=1, colspan=2, projection='3d')
        
        self.ax_params = plt.subplot2grid((3, 4), (2, 0), rowspan=1, colspan=2)
        self.ax_phase = plt.subplot2grid((3, 4), (2, 2), rowspan=1, colspan=2)
        
        # 為控制元件留出空間
        plt.subplots_adjust(bottom=0.15)
        
        # 創建控制滑動條
        ax_freq = plt.axes([0.1, 0.05, 0.3, 0.03])
        self.freq_slider = Slider(ax_freq, '頻率 (Hz)', 10, 500, 
                                valinit=self.frequency, valfmt='%.0f')
        self.freq_slider.on_changed(self.update_frequency)
        
        ax_time = plt.axes([0.5, 0.05, 0.3, 0.03])
        self.time_slider = Slider(ax_time, '時間相位', 0, 2*np.pi, 
                                valinit=self.time, valfmt='%.2f')
        self.time_slider.on_changed(self.update_time)
        
        # 播放/暫停按鈕
        ax_button = plt.axes([0.85, 0.05, 0.1, 0.04])
        self.play_button = Button(ax_button, '播放')
        self.play_button.on_clicked(self.toggle_animation)
        
    def setup_animation(self):
        """設置動畫"""
        self.anim = FuncAnimation(self.fig, self.animate, interval=50, blit=False)
        self.anim.event_source.stop()  # 初始停止
        
    def calculate_circuit_response(self, freq):
        """計算電路響應"""
        omega = 2 * np.pi * freq
        XL = omega * self.L
        XC = 1 / (omega * self.C)
        Z = np.sqrt(self.R**2 + (XL - XC)**2)
        
        # 假設源電壓幅值為1V
        V0 = 1.0
        I0 = V0 / Z if Z != 0 else 0
        
        # 相位角
        phi = np.arctan2(XL - XC, self.R)
        
        return {
            'I0': I0,
            'V0': V0,
            'phi': phi,
            'XL': XL,
            'XC': XC,
            'Z': Z,
            'omega': omega
        }
    
    def calculate_electric_field(self, t):
        """計算電容器內電場分布"""
        response = self.calculate_circuit_response(self.frequency)
        
        # 電容器電壓
        V_cap = response['I0'] * response['XC'] * np.cos(response['omega'] * t - np.pi/2)
        
        # 平行板電容器電場 E = V/d
        plate_separation = 0.005  # 5mm
        E_magnitude = abs(V_cap) / plate_separation
        
        # 電場方向：從正極板指向負極板 (z方向)
        E_field = np.zeros_like(self.cap_Z)
        E_field[:, :, :] = E_magnitude * np.sign(V_cap)
        
        # 添加邊緣效應
        edge_factor = np.ones_like(self.cap_X)
        edge_factor = np.exp(-((self.cap_X**2 + self.cap_Y**2) / (0.01**2)) * 2)
        E_field *= edge_factor[:, :, np.newaxis]
        
        return E_field, V_cap
    
    def calculate_magnetic_field(self, t):
        """計算電感器內磁場分布"""
        response = self.calculate_circuit_response(self.frequency)
        
        # 電感器電流
        I_ind = response['I0'] * np.cos(response['omega'] * t)
        
        # 螺線管內部磁場 B = μ₀nI (軸向)
        mu0 = 4 * np.pi * 1e-7  # 真空磁導率
        turns_per_length = 1000  # 每米匝數
        B_magnitude = mu0 * turns_per_length * abs(I_ind)
        
        # 磁場主要沿z軸方向
        B_field_z = np.zeros_like(self.ind_Z)
        
        # 只在螺線管內部有均勻磁場
        mask = self.ind_R <= 0.008  # 內部半徑8mm
        B_field_z[mask] = B_magnitude * np.sign(I_ind)
        
        # 添加端部效應
        z_factor = np.exp(-((self.ind_Z / 0.015)**2) * 2)
        B_field_z *= z_factor
        
        return B_field_z, I_ind
    
    def calculate_energy_density(self, t):
        """計算能量密度"""
        E_field, _ = self.calculate_electric_field(t)
        B_field, _ = self.calculate_magnetic_field(t)
        
        epsilon0 = 8.854e-12  # 真空介電常數
        mu0 = 4 * np.pi * 1e-7  # 真空磁導率
        
        # 電場能量密度 u_E = ½ε₀E²
        u_electric = 0.5 * epsilon0 * E_field**2
        
        # 磁場能量密度 u_B = ½B²/μ₀
        u_magnetic = 0.5 * B_field**2 / mu0
        
        return u_electric, u_magnetic
    
    def draw_circuit_diagram(self):
        """繪製電路圖"""
        self.ax_circuit.clear()
        self.ax_circuit.set_xlim(0, 10)
        self.ax_circuit.set_ylim(0, 8)
        
        # 電阻
        rect = patches.Rectangle((1, 4), 1.5, 0.8, linewidth=2, 
                               edgecolor='orange', facecolor='none')
        self.ax_circuit.add_patch(rect)
        self.ax_circuit.text(1.75, 4.4, 'R', ha='center', va='center', fontsize=12, fontweight='bold')
        
        # 電感 (螺旋線)
        x_coil = np.linspace(3, 5, 50)
        y_coil = 4.4 + 0.3 * np.sin(10 * x_coil)
        self.ax_circuit.plot(x_coil, y_coil, 'b-', linewidth=3)
        self.ax_circuit.text(4, 3.5, 'L', ha='center', va='center', fontsize=12, 
                           fontweight='bold', color='blue')
        
        # 電容 (平行線)
        self.ax_circuit.plot([6, 6], [3.8, 5], 'g-', linewidth=4)
        self.ax_circuit.plot([7, 7], [3.8, 5], 'g-', linewidth=4)
        self.ax_circuit.text(6.5, 3.2, 'C', ha='center', va='center', fontsize=12, 
                           fontweight='bold', color='green')
        
        # 電壓源
        circle = patches.Circle((8.5, 4.4), 0.4, linewidth=2, 
                              edgecolor='red', facecolor='none')
        self.ax_circuit.add_patch(circle)
        self.ax_circuit.text(8.5, 4.4, '~', ha='center', va='center', 
                           fontsize=16, fontweight='bold', color='red')
        
        # 連接線
        self.ax_circuit.plot([0.5, 1], [4.4, 4.4], 'k-', linewidth=2)
        self.ax_circuit.plot([2.5, 3], [4.4, 4.4], 'k-', linewidth=2)
        self.ax_circuit.plot([5, 6], [4.4, 4.4], 'k-', linewidth=2)
        self.ax_circuit.plot([7, 8.1], [4.4, 4.4], 'k-', linewidth=2)
        self.ax_circuit.plot([8.9, 9.5], [4.4, 4.4], 'k-', linewidth=2)
        
        # 下方連接線
        self.ax_circuit.plot([0.5, 0.5], [4.4, 2], 'k-', linewidth=2)
        self.ax_circuit.plot([0.5, 9.5], [2, 2], 'k-', linewidth=2)
        self.ax_circuit.plot([9.5, 9.5], [2, 4.4], 'k-', linewidth=2)
        
        self.ax_circuit.set_title('RLC串聯電路')
        self.ax_circuit.axis('off')
    
    def draw_field_cross_sections(self, t):
        """繪製場的截面圖"""
        E_field, V_cap = self.calculate_electric_field(t)
        B_field, I_ind = self.calculate_magnetic_field(t)
        
        # 電場截面 (中間z層)
        self.ax_cap_field.clear()
        mid_z = E_field.shape[2] // 2
        E_slice = E_field[:, :, mid_z]
        
        im1 = self.ax_cap_field.imshow(E_slice, extent=[-20, 20, -10, 10], 
                                     cmap='RdBu_r', origin='lower')
        self.ax_cap_field.set_title(f'電容器電場 (V={V_cap:.3f}V)')
        self.ax_cap_field.set_xlabel('x (mm)')
        self.ax_cap_field.set_ylabel('y (mm)')
        
        # 磁場截面 (中間z層)
        self.ax_ind_field.clear()
        mid_z_ind = B_field.shape[2] // 2
        B_slice = B_field[:, :, mid_z_ind]
        
        im2 = self.ax_ind_field.imshow(B_slice, extent=[-10, 10, -10, 10],
                                     cmap='viridis', origin='lower')
        self.ax_ind_field.set_title(f'電感器磁場 (I={I_ind:.3f}A)')
        self.ax_ind_field.set_xlabel('x (mm)')
        self.ax_ind_field.set_ylabel('y (mm)')
    
    def draw_3d_fields(self, t):
        """繪製3D場分布"""
        u_electric, u_magnetic = self.calculate_energy_density(t)
        
        # 3D電場能量密度
        self.ax_cap_3d.clear()
        
        # 選擇有意義的等值面
        threshold_e = np.max(u_electric) * 0.3
        if threshold_e > 0:
            # 簡化顯示：只顯示幾個關鍵截面
            for i in range(0, u_electric.shape[2], 5):
                slice_data = u_electric[:, :, i]
                if np.max(slice_data) > threshold_e:
                    X, Y = np.meshgrid(self.cap_x * 1000, self.cap_y * 1000)
                    Z = np.full_like(X, self.cap_z[i] * 1000)
                    self.ax_cap_3d.contour(X, Y, Z, [slice_data], alpha=0.6, cmap='Reds')
        
        self.ax_cap_3d.set_title('電場能量密度 3D')
        self.ax_cap_3d.set_xlabel('x (mm)')
        self.ax_cap_3d.set_ylabel('y (mm)')
        self.ax_cap_3d.set_zlabel('z (mm)')
        
        # 3D磁場能量密度
        self.ax_ind_3d.clear()
        
        threshold_b = np.max(u_magnetic) * 0.3
        if threshold_b > 0:
            # 柱座標轉換為笛卡爾座標顯示
            for k in range(0, u_magnetic.shape[2], 8):
                slice_data = u_magnetic[:, :, k]
                if np.max(slice_data) > threshold_b:
                    self.ax_ind_3d.contour(self.ind_X[:, :, k] * 1000, 
                                         self.ind_Y[:, :, k] * 1000,
                                         self.ind_Z[:, :, k] * 1000, 
                                         [slice_data], alpha=0.6, cmap='Blues')
        
        self.ax_ind_3d.set_title('磁場能量密度 3D')
        self.ax_ind_3d.set_xlabel('x (mm)')
        self.ax_ind_3d.set_ylabel('y (mm)')
        self.ax_ind_3d.set_zlabel('z (mm)')
    
    def draw_energy_evolution(self, t):
        """繪製能量演化"""
        times = np.linspace(0, 4*np.pi/self.calculate_circuit_response(self.frequency)['omega'], 100)
        
        electric_energies = []
        magnetic_energies = []
        
        for time_point in times:
            u_e, u_b = self.calculate_energy_density(time_point)
            electric_energies.append(np.sum(u_e))
            magnetic_energies.append(np.sum(u_b))
        
        self.ax_energy.clear()
        self.ax_energy.plot(times * 1000, electric_energies, 'r-', 
                          label='電場能量', linewidth=2)
        self.ax_energy.plot(times * 1000, magnetic_energies, 'b-', 
                          label='磁場能量', linewidth=2)
        self.ax_energy.plot(times * 1000, np.array(electric_energies) + np.array(magnetic_energies), 
                          'k--', label='總能量', linewidth=2)
        
        # 標記當前時間
        current_time_ms = t * 1000
        self.ax_energy.axvline(current_time_ms, color='green', linestyle=':', 
                             linewidth=2, label='當前時刻')
        
        self.ax_energy.set_xlabel('時間 (ms)')
        self.ax_energy.set_ylabel('能量 (相對單位)')
        self.ax_energy.set_title('電磁能量時間演化')
        self.ax_energy.legend()
        self.ax_energy.grid(True, alpha=0.3)
    
    def display_parameters(self):
        """顯示參數信息"""
        self.ax_params.clear()
        self.ax_params.axis('off')
        
        response = self.calculate_circuit_response(self.frequency)
        
        params_text = f"""
電路參數:
• R = {self.R} Ω
• L = {self.L*1000:.1f} mH
• C = {self.C*1e6:.1f} μF
• 共振頻率 = {self.f_resonance:.1f} Hz

當前狀態 (f = {self.frequency:.0f} Hz):
• 阻抗 |Z| = {response['Z']:.2f} Ω
• 相位角 φ = {np.degrees(response['phi']):.1f}°
• 電流幅值 = {response['I0']:.4f} A
• 感抗 XL = {response['XL']:.2f} Ω
• 容抗 XC = {response['XC']:.2f} Ω

場特性:
• 電場頻率 = {self.frequency:.0f} Hz
• 磁場頻率 = {self.frequency:.0f} Hz
• 能量轉換週期 = {1000/self.frequency:.1f} ms
        """
        
        self.ax_params.text(0.05, 0.95, params_text, transform=self.ax_params.transAxes,
                          fontsize=10, verticalalignment='top', fontfamily='monospace')
        
        self.ax_params.set_title('電路與場參數', fontsize=12, fontweight='bold')
    
    def draw_phase_relationships(self):
        """繪製相位關係"""
        self.ax_phase.clear()
        
        response = self.calculate_circuit_response(self.frequency)
        t_array = np.linspace(0, 2*np.pi/response['omega'], 200)
        
        # 電流
        current = response['I0'] * np.cos(response['omega'] * t_array)
        
        # 各元件電壓
        v_R = response['I0'] * self.R * np.cos(response['omega'] * t_array)
        v_L = response['I0'] * response['XL'] * np.cos(response['omega'] * t_array + np.pi/2)
        v_C = response['I0'] * response['XC'] * np.cos(response['omega'] * t_array - np.pi/2)
        
        self.ax_phase.plot(t_array * 1000, current, 'purple', label='電流 i(t)', linewidth=2)
        self.ax_phase.plot(t_array * 1000, v_R, 'orange', label='電阻電壓', linewidth=2)
        self.ax_phase.plot(t_array * 1000, v_L, 'blue', label='電感電壓', linewidth=2)
        self.ax_phase.plot(t_array * 1000, v_C, 'green', label='電容電壓', linewidth=2)
        
        # 標記當前時間
        current_time_ms = self.time / response['omega'] * 1000
        self.ax_phase.axvline(current_time_ms, color='red', linestyle=':', 
                            linewidth=2, label='當前相位')
        
        self.ax_phase.set_xlabel('時間 (ms)')
        self.ax_phase.set_ylabel('幅值')
        self.ax_phase.set_title('電壓電流相位關係')
        self.ax_phase.legend()
        self.ax_phase.grid(True, alpha=0.3)
    
    def update_fields(self):
        """更新所有顯示"""
        omega = self.calculate_circuit_response(self.frequency)['omega']
        t = self.time / omega
        
        self.draw_circuit_diagram()
        self.draw_field_cross_sections(t)
        self.draw_3d_fields(t)
        self.draw_energy_evolution(t)
        self.display_parameters()
        self.draw_phase_relationships()
        
        self.fig.canvas.draw()
    
    def update_frequency(self, val):
        """頻率滑動條回調"""
        self.frequency = self.freq_slider.val
        self.update_fields()
    
    def update_time(self, val):
        """時間滑動條回調"""
        self.time = self.time_slider.val
        self.update_fields()
    
    def toggle_animation(self, event):
        """切換動畫播放狀態"""
        if self.is_playing:
            self.anim.event_source.stop()
            self.play_button.label.set_text('播放')
            self.is_playing = False
        else:
            self.anim.event_source.start()
            self.play_button.label.set_text('暫停')
            self.is_playing = True
    
    def animate(self, frame):
        """動畫更新函數"""
        if self.is_playing:
            self.time += 0.1
            if self.time > 2 * np.pi:
                self.time = 0
            
            self.time_slider.set_val(self.time)
            self.update_fields()
        
        return []

# 使用示例
if __name__ == "__main__":
    # 創建電磁場分析器
    # 使用適合觀察場效應的參數
    analyzer = RLCFieldAnalyzer(R=50, L=0.001, C=1e-6)
    
    plt.show()


        
      

   
    
   
     
    
    
  

        
  
   

