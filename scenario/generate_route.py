import random
random.seed(42)
import math
import os
from datetime import datetime
import traci
from sumolib import checkBinary


def generate_route_file(
    commuter_count=500,    # 固定通勤车数量
    taxi_count=200,        # 固定出租车数量
    flexible_count=500,    # 固定弹性车数量
    emergency_count=10,    # 紧急车辆数量
    random_count=1800,     # 随机车数量
    simulation_duration=3500  # 仿真持续时间
):
    # 设置默认出发时间范围
    DEFAULT_DEPART_RANGES = {
        "commuter": (800, 1500),
        "taxi": (200, 1600),
        "flexible": (300, 2000)
    }

    os.makedirs("scenario", exist_ok=True)

    if not os.path.exists("scenario/openmap.net.xml"):
        print("错误：找不到路网文件 scenario/openmap.net.xml")
        return

    try:
        sumo_binary = checkBinary('sumo')
        traci.start([sumo_binary, "-n", "scenario/openmap.net.xml", "--quit-on-end"])
        print("成功连接到SUMO")
    except Exception as e:
        print(f"无法连接SUMO: {str(e)}")
        return

    # 获取所有有效边（过滤内部边）
    all_edges = [e for e in traci.edge.getIDList() if not e.startswith(":")]
    print(f"找到 {len(all_edges)} 条有效道路")

    # 预生成所有可能的有效路线
    print("预生成随机路线库...")
    route_library = []
    MAX_ROUTES = 500  # 限制最大预生成路线数
    
    for _ in range(MAX_ROUTES):
        from_edge = random.choice(all_edges)
        to_edge = random.choice(all_edges)
        if from_edge != to_edge:
            try:
                route = traci.simulation.findRoute(from_edge, to_edge)
                if route.edges:
                    route_library.append({
                        "edges": route.edges,
                        "from": from_edge,
                        "to": to_edge
                    })
            except:
                continue

    print(f"已预生成 {len(route_library)} 条随机路线")

    with open("scenario/openmap.rou.xml", "w", encoding="utf-8") as routes:
        # 写入XML头
        print("""<?xml version="1.0" encoding="UTF-8"?>
<routes xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:noNamespaceSchemaLocation="http://sumo.dlr.de/xsd/routes_file.xsd">""", file=routes)

        # 车辆类型定义
        print("""    <vType id="DEFAULT_VEHTYPE" vClass="passenger" speedDev="0.1"/>
    <vType id="emergency" vClass="emergency" speedDev="0.1" minGap="0.5" color="255,0,0" speedFactor="1.5"/>
    <vType id="commuter" vClass="passenger" speedDev="0.2" minGap="1.0" color="0,0,255"/>
    <vType id="flexible" vClass="passenger" speedDev="0.3" minGap="2.0" color="0,255,0"/>
    <vType id="taxi" vClass="taxi" speedDev="0.3" minGap="2.0" color="255,255,0"/>""", file=routes)

        # 固定车流的预设OD对
        OD_PAIRS = {
            "commuter": [
                {"from": "80912579#3", "to": "-440881469#10"},
                {"from": "519662802#2", "to": "-456384920#2"},
                {"from": "440881630", "to": "-29142233#0"},
                {"from": "734144754#1", "to": "-543462683#4"}
            ],
            "taxi": [
                {"from": "1172199237#0", "to": "624720027#0"},
                {"from": "-440881469#1", "to": "444418076#1"},
                {"from": "576645493#0", "to": "440881630"},
                {"from": "444126136", "to": "-519662802#6"}
            ],
            "flexible": [
                {"from": "334124302#0", "to": "322629006#0"},
                {"from": "519199419#0", "to": "547065956#0"},
                {"from": "1373419751#0", "to": "-858600107#1"},
                {"from": "440774131#1", "to": "-519662802#0"}
            ]
        }

        # 生成固定路线
        fixed_routes = {}
        route_id = 0
        for vtype, pairs in OD_PAIRS.items():
            for idx, pair in enumerate(pairs):
                try:
                    route = traci.simulation.findRoute(pair["from"], pair["to"])
                    if route.edges:
                        route_id_str = f"{vtype}_fixed_{idx}"
                        print(f'    <route id="{route_id_str}" edges="{" ".join(route.edges)}"/>', file=routes)
                        fixed_routes[route_id_str] = True
                        route_id += 1
                except:
                    print(f"无法生成固定路线: {pair['from']} -> {pair['to']}")

        # 生成随机路线
        for i, route in enumerate(route_library):
            route_id_str = f"random_route_{i}"
            print(f'    <route id="{route_id_str}" edges="{" ".join(route["edges"])}"/>', file=routes)

        # 生成车辆定义
        vehicle_definitions = []
        vehNr = 0

        # 固定车流 - 均匀分布生成时间
        for vtype, count in [("commuter", commuter_count), 
                           ("taxi", taxi_count), 
                           ("flexible", flexible_count)]:
            depart_range = DEFAULT_DEPART_RANGES[vtype]
            time_range = depart_range[1] - depart_range[0]
            interval = time_range / count if count > 0 else 0
            
            for i in range(count):
                route_idx = i % 3
                route_id = f"{vtype}_fixed_{route_idx}"
                if route_id in fixed_routes:
                    # 均匀分布生成时间，加上小随机扰动
                    depart_time = depart_range[0] + int(i * interval) + random.randint(-30, 30)
                    depart_time = max(depart_range[0], min(depart_range[1], depart_time))
                    
                    vehicle_def = f'    <vehicle id="fixed_{vtype}_{i:03d}" type="{vtype}" route="{route_id}" depart="{depart_time}" departLane="random"/>'
                    vehicle_definitions.append((depart_time, vehicle_def))
                    vehNr += 1

        # 紧急车辆 - 均匀分布生成时间
        emergency_times = []
        if emergency_count > 0:
            interval = simulation_duration / emergency_count
            for i in range(emergency_count):
                # 均匀分布生成时间，加上小随机扰动
                depart_time = int(i * interval) + random.randint(-50, 50)
                depart_time = max(0, min(simulation_duration, depart_time))
                emergency_times.append(depart_time)
        
        # 随机车流
        avg_interval = simulation_duration / random_count if random_count > 0 else float('inf')
        time = 0
        generated_random = 0
        emergency_index = 0
        
        while time < simulation_duration and generated_random < random_count:
            # 检查是否需要生成紧急车辆
            if emergency_index < len(emergency_times) and time >= emergency_times[emergency_index]:
                vehicle_type = "emergency"
                emergency_index += 1
            else:
                vehicle_type = random.choices(
                    ["commuter", "flexible", "taxi"],
                    weights=[0.5, 0.3, 0.2]
                )[0]
            
            if route_library:
                random_route = random.choice(route_library)
                route_id = f"random_route_{route_library.index(random_route)}"
                
                if vehicle_type == "emergency":
                    vehicle_def = f'    <vehicle id="emergency_{vehNr:05d}" type="emergency" route="{route_id}" depart="{time}" departLane="best" departSpeed="max" color="255,0,0"/>'
                else:
                    vehicle_def = f'    <vehicle id="random_{vehicle_type}_{vehNr:05d}" type="{vehicle_type}" route="{route_id}" depart="{time}" departLane="random"/>'
                
                vehicle_definitions.append((time, vehicle_def))
                
                p = random.random()
                time += max(int(-math.log(1.0 - p) * avg_interval), 1)
                vehNr += 1
                generated_random += 1

        # 按时间排序并写入
        vehicle_definitions.sort(key=lambda x: x[0])
        for _, vehicle_def in vehicle_definitions:
            print(vehicle_def, file=routes)

        print("</routes>", file=routes)

    traci.close()
    print(f"已生成路由文件: scenario/openmap.rou.xml")
    print(f"固定车流: {commuter_count + taxi_count + flexible_count} 辆")
    print(f"随机车流: {generated_random} 辆 (包含{emergency_count}辆紧急车辆)")

if __name__ == "__main__":
    generate_route_file()