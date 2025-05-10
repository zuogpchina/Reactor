import openmc
import openmc.deplete
import numpy as np
import matplotlib.pyplot as plt

# ================== 核数据库配置 ==================
openmc.config['cross_sections'] = '/home/zgp/test/openNuclear/openmc-dev_data/mcnp_endfb70/cross_sections.xml'
CHAIN_FILE = '/home/zgp/test/openNuclear/openmc-dev_data/depletion/chain_endfb71_pwr.xml'

# ================== 材料定义 ==================
def create_uo2(enrichment, density):
    """创建UO2燃料"""
    uo2 = openmc.Material(name=f'UO2_{enrichment}%')
    uo2.add_element('U', 1.0, enrichment=enrichment/100, enrichment_type='wo')
    uo2.add_element('O', 2.0)
    uo2.set_density('g/cm3', density)
    return uo2

def create_gd_uo2(enrichment, density, pf):
    """创建含Gd毒物的UO2燃料"""
    uo2 = create_uo2(enrichment, density)
    gd2o3 = openmc.Material()
    gd2o3.add_element('Gd', 2.0)
    gd2o3.add_element('O', 3.0)
    gd2o3.set_density('g/cm3', 7.41)
    
    mixed = openmc.Material.mix_materials(
        [uo2, gd2o3],
        [1 - pf, pf],
        'wo'
    )
    mixed.name = f'UO2_Gd_{pf}%'
    return mixed

# 包壳材料（纯锆）
zircaloy = openmc.Material(name='Zircaloy')
zircaloy.add_element('Zr', 1.0)
zircaloy.set_density('g/cm3', 6.56)

# 冷却剂/慢化剂（水）
water = openmc.Material(name='Water')
water.add_elements_from_formula('H2O')
water.set_density('g/cm3', 1.0)
water.add_s_alpha_beta('c_H_in_H2O')

# ================== 几何定义 ==================
def build_geometry(config):
    # 堆芯参数
    core_height = 100.0  # cm
    core_radius = 76.0   # cm
    assembly_pitch = 21.5  # 组件间距 cm
    pin_pitch = 1.265    # 燃料棒间距 cm（12.65 mm）

    # 根据配置设置参数
    if config == 'I':
        density = 10.0
        zones = {
            'A': {'enrich': 4.0, 'pf': 0.001},
            'B': {'enrich': 7.0, 'pf': 0.003},
            'C': {'enrich': 10.0, 'pf': 0.007}
        }
    elif config == 'II':
        density = 5.0
        zones = {
            'A': {'enrich': 8.0, 'pf': 0.004},
            'B': {'enrich': 14.0, 'pf': 0.012},
            'C': {'enrich': 20.0, 'pf': 0.028}
        }
    elif config == 'III':
        density = 2.5
        zones = {
            'A': {'enrich': 16.0, 'pf': 0.016},
            'B': {'enrich': 28.0, 'pf': 0.048},
            'C': {'enrich': 40.0, 'pf': 0.112}
        }

    # 创建燃料棒组件
    assembly_universes = []
    for zone in ['A', 'B', 'C']:
        # 创建普通燃料棒和毒物燃料棒材料
        uo2 = create_uo2(zones[zone]['enrich'], density)
        poisoned_uo2 = create_gd_uo2(zones[zone]['enrich'], density, zones[zone]['pf'])

        # 定义燃料棒几何结构
        fuel_radius = 0.412  # cm（4.12 mm）
        cladding_outer = 0.476  # cm（包壳外径=燃料外径+0.64mm）

        # 创建燃料棒单元
        fuel_cell = openmc.Cell()
        fuel_surf = openmc.ZCylinder(r=fuel_radius)
        fuel_cell.region = -fuel_surf
        fuel_cell.fill = uo2

        cladding_cell = openmc.Cell()
        cladding_surf = openmc.ZCylinder(r=cladding_outer)
        cladding_cell.region = +fuel_surf & -cladding_surf
        cladding_cell.fill = zircaloy

        # 普通燃料棒Universe
        fuel_pin = openmc.Universe(cells=[fuel_cell, cladding_cell])

        # 毒物燃料棒（替换燃料材料）
        poison_cell = openmc.Cell()
        poison_cell.region = -fuel_surf
        poison_cell.fill = poisoned_uo2
        poison_pin = openmc.Universe(cells=[poison_cell, cladding_cell])

        # 构建17x17组件栅格
        assembly = openmc.RectLattice(name=f'Assembly_{zone}')
        assembly.lower_left = (-assembly_pitch/2, -assembly_pitch/2)
        assembly.pitch = (pin_pitch, pin_pitch)
        assembly.universes = np.full((17, 17), fuel_pin)

        # 设置毒物棒位置（示例：中心4根）
        poison_positions = [(8,8), (8,9), (9,8), (9,9)]
        for x, y in poison_positions:
            assembly.universes[x, y] = poison_pin

        # 组件容器（填充水到间隙）
        assembly_box = openmc.rectangular_prism(
            width=assembly_pitch,
            height=assembly_pitch,
            boundary_type='reflective'
        )
        assembly_z = openmc.ZPlane(z0=0) & openmc.ZPlane(z0=core_height)
        assembly_cell = openmc.Cell(
            fill=assembly,
            region=assembly_box & assembly_z
        )
        assembly_cell.fill = openmc.Matrix()  # 自动填充水到间隙
        assembly_universes.append(openmc.Universe(cells=[assembly_cell]))

    # 构建堆芯布局（3+5+5+5+3排列）
    core = openmc.RectLattice(name='Core')
    core.lower_left = (-3*assembly_pitch, -3*assembly_pitch)
    core.pitch = (assembly_pitch, assembly_pitch)
    core.universes = [
        [assembly_universes[0]]*3 + [assembly_universes[1]]*5 + [assembly_universes[2]]*5,
        [assembly_universes[1]]*5 + [assembly_universes[2]]*5 + [assembly_universes[1]]*5,
        [assembly_universes[0]]*3 + [assembly_universes[1]]*5 + [assembly_universes[2]]*5
    ]

    # 堆芯容器
    core_cyl = openmc.ZCylinder(r=core_radius, boundary_type='vacuum')
    bottom = openmc.ZPlane(z0=0, boundary_type='vacuum')
    top = openmc.ZPlane(z0=core_height, boundary_type='vacuum')
    core_region = -core_cyl & +bottom & -top

    core_cell = openmc.Cell(
        name='Core',
        region=core_region,
        fill=core
    )

    # 反射层
    reflector = openmc.Cell(
        name='Reflector',
        region=+core_cyl | +top | -bottom,
        fill=water
    )

    model = openmc.Model()
    model.geometry = openmc.Geometry([core_cell, reflector])
    model.materials = openmc.Materials([uo2, poisoned_uo2, zircaloy, water, gd2o3])

    return model

# ...（其余函数保持不变，详见原始代码）

if __name__ == "__main__":
    model = build_geometry('I')
    configure_settings(model)
    
    model.export_to_xml()
    openmc.run()
    
    # 可视化
    plot = model.plot(basis='xy', color_by='material')
    plt.title('Core Layout (Axial View)')
    plt.savefig('core_layout_xy.png')
    
    plot = model.plot(basis='xz', color_by='material')
    plt.title('Core Layout (Radial View)')
    plt.savefig('core_layout_xz.png')
    
    plot_results(['I'])