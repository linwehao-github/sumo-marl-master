from collections import defaultdict, deque
import os
import pickle
import random
import shutil
import time
random.seed(42)
import torch
import traci
from sumolib import checkBinary
import numpy as np
from sumolib.net import readNet
import torch.nn as nn
from xml.etree import ElementTree as ET
import xml.etree.ElementTree as ET
import statistics

class TrafficEnv:
    def __init__(self, mode='binary'):
        # If the mode is 'gui', it renders the scenario.
        if mode == 'gui':
            self.sumoBinary = checkBinary('sumo-gui')
        else:
            self.sumoBinary = checkBinary('sumo')
        self.sumoCmd = [self.sumoBinary, "-c", './scenario/openmap.sumocfg', '--no-step-log', '-W']
        net = readNet('./scenario/openmap.net.xml')
        self.time = None
        self.decision_time = 10
        self.tl_ids = [tl.getID() for tl in net.getTrafficLights()]  # 获取ID字符串
        self.n_intersections = len(self.tl_ids)
        self.lstm_extension = nn.LSTM(input_size=1, hidden_size=4, batch_first=True)
        self.extension_fc = nn.Linear(8, 1)
        self.extension_memory = deque(maxlen=3)  # 记忆最近5个时间步的数据
        # 初始化时先连接SUMO获取相位信息
        traci.start(self.sumoCmd)
        self.max_lanes = max(len(traci.trafficlight.getControlledLanes(tl_id)) 
                            for tl_id in traci.trafficlight.getIDList())
        self.phase_limits = {}  # 存储每个路口的最大相位索引
        for tl_id in self.tl_ids:
            # 获取该路口的所有相位定义
            logic = traci.trafficlight.getAllProgramLogics(tl_id)[0]
            self.phase_limits[tl_id] = len(logic.phases) - 1  # 最大相位索引
        self.n_phase = max(self.phase_limits.values()) + 1 if self.phase_limits else 0
        self.state_dim = 4 * self.max_lanes + self.n_phase
        traci.close()
        # 添加车辆类型配置
        self.vehicle_types = {
            'emergency': {'priority': 3, 'threshold': 1},
            'commuter': {'priority': 2, 'threshold': 5},
        }
        # 相位延长参数（秒）
        self.phase_extension = {
            'emergency': 30,'commuter': 15
        }
        # 出发时间调整参数
        self.departure_adjustment_factor = 0.5  # 调整幅度因子
        self.max_departure_adjustment = 300  # 最大调整时间（秒）
        self.original_rou_file = './scenario/openmap.rou.xml'
        self.modified_rou_file = './scenario/openmap_modified.rou.xml'
        self.vehicle_waiting_times = {}  # 存储车辆等待时间
        self.edge_congestion_levels = {}  # 存储道路拥堵程度
        self.vehicle_departure_adjustments = {}  # 存储车辆的出发时间调整
        self.congestion_threshold = 0.7  # 拥堵判断阈值

        
    def _parse_tripinfo(self, tripinfo_file):
        try:
            tree = ET.parse(tripinfo_file)
            root = tree.getroot()
            
            for trip in root.findall('tripinfo'):
                veh_id = trip.get('id')
                waiting_time = float(trip.get('waitingTime', 0))
                
                # 处理可能缺失的route属性
                route = trip.get('route')
                route_edges = route.split() if route is not None else []
                
                self.vehicle_waiting_times[veh_id] = waiting_time
                
                # 计算拥堵程度（仅当有路径时）
                if route_edges:
                    congestion_sum = 0
                    valid_edges = 0
                    for edge in route_edges:
                        if edge in self.edge_congestion_levels:
                            congestion_sum += self.edge_congestion_levels[edge]
                            valid_edges += 1
                    avg_congestion = congestion_sum / valid_edges if valid_edges > 0 else 0
                else:
                    avg_congestion = 0
                
                # 更新出发时间调整
                adjustment = self._calculate_departure_adjustment(waiting_time, avg_congestion)
                if veh_id in self.vehicle_departure_adjustments:
                    self.vehicle_departure_adjustments[veh_id] = 0.7 * self.vehicle_departure_adjustments[veh_id] + 0.3 * adjustment
                else:
                    self.vehicle_departure_adjustments[veh_id] = adjustment
                    
        except ET.ParseError as e:
            print(f"XML解析错误: {e}")
        except Exception as e:
            print(f"其他错误: {e}")
    
    def _calculate_departure_adjustment(self, waiting_time, congestion_level):
        """根据等待时间和拥堵程度计算出发时间调整"""
        # 等待时间越长，越应该提前出发
        # 拥堵程度越高，越应该延后出发（避免加入拥堵）
        
        # 基于等待时间的调整（正数表示提前出发）
        wait_adjustment = min(waiting_time * self.departure_adjustment_factor, 
                             self.max_departure_adjustment)
        congestion_adjustment = -min(congestion_level * 30, 60)  # 拥堵权重提高
        # 基于拥堵程度的调整（负数表示延后出发）
        total_adjustment = (wait_adjustment * 0.7 + congestion_adjustment * 0.3)
        # 限制在最大调整范围内
        return max(-self.max_departure_adjustment, 
                 min(self.max_departure_adjustment, total_adjustment))
    
    def _update_edge_congestion(self):
        """更新道路拥堵程度"""
        for edge in traci.edge.getIDList():
            # 计算车道占用率
            lane_count = traci.edge.getLaneNumber(edge)
            total_vehicles = 0
            total_length = 0
            
            for lane_idx in range(lane_count):
                lane_id = f"{edge}_{lane_idx}"
                vehicles = traci.lane.getLastStepVehicleNumber(lane_id)
                total_vehicles += vehicles
                total_length += traci.lane.getLength(lane_id)
            
            # 计算拥堵程度（0-1之间）
            if total_length > 0:
                congestion = min(1.0, total_vehicles / (total_length * 0.2))  # 假设每米最多0.2辆车
            else:
                congestion = 0
                
            self.edge_congestion_levels[edge] = congestion
    
    def _modify_rou_file(self):
        """修改rou.xml文件，调整车辆的出发时间并确保按时间排序"""
        try:
            # 解析原始rou文件
            tree = ET.parse(self.original_rou_file)
            root = tree.getroot()
            
            # 收集所有车辆元素
            vehicles = root.findall('vehicle')
            
            # 创建包含车辆ID和调整后出发时间的列表
            vehicle_data = []
            for vehicle in vehicles:
                veh_id = vehicle.get('id')
                original_depart = float(vehicle.get('depart'))
                adjustment = self.vehicle_departure_adjustments.get(
                    veh_id, 
                    random.uniform(-50, 50)  # 默认随机调整-50到50秒
                )
                
                # 计算新的出发时间（确保不小于0）
                new_depart = max(0, original_depart + adjustment)
                vehicle_data.append((new_depart, vehicle))
            
            # 按出发时间排序
            vehicle_data.sort(key=lambda x: x[0])
            
            # 先移除所有车辆元素
            for vehicle in vehicles:
                root.remove(vehicle)
            
            # 按排序后的顺序添加车辆
            for depart, vehicle in vehicle_data:
                vehicle.set('depart', str(depart))
                root.append(vehicle)
            
            # 保存修改后的文件
            tree.write(self.modified_rou_file)
            
            # 更新配置文件使用修改后的rou文件
            self._update_sumo_config()
            
        except Exception as e:
            print(f"Error modifying rou file: {e}")
            shutil.copyfile(self.original_rou_file, self.modified_rou_file)
    
    def _update_sumo_config(self):
        """更新sumo配置文件以使用修改后的rou文件"""
        try:
            config_file = './scenario/openmap.sumocfg'
            tree = ET.parse(config_file)
            root = tree.getroot()
            
            # 找到input元素
            for input_elem in root.findall('input'):
                # 更新rou文件路径
                for rou_file in input_elem.findall('route-files'):
                    rou_file.set('value', 'openmap_modified.rou.xml')
            
            # 保存修改后的配置文件
            tree.write(config_file)
        except Exception as e:
            print(f"Error updating sumo config: {e}")
    
    def _detect_congestion(self):
        """检测路网中的拥堵情况"""
        congested_edges = set()
        for edge in traci.edge.getIDList():
            # 计算车道占用率
            lane_count = traci.edge.getLaneNumber(edge)
            total_vehicles = 0
            total_capacity = 0
            
            for lane_idx in range(lane_count):
                lane_id = f"{edge}_{lane_idx}"
                total_vehicles += traci.lane.getLastStepVehicleNumber(lane_id)
                total_capacity += traci.lane.getLastStepVehicleNumber(lane_id) / traci.lane.getLength(lane_id)
            
            if total_capacity > 0 and (total_vehicles / total_capacity) > self.congestion_threshold:
                congested_edges.add(edge)
        
        return congested_edges
    
    def detect_special_vehicles(self, intersection_id):
        """检测当前路口需要优先处理的车辆"""
        lanes = traci.trafficlight.getControlledLanes(intersection_id)
        vehicle_counts = {'emergency': 0, 'commuter': 0}
        
        for lane in lanes:
            vehicles = traci.lane.getLastStepVehicleIDs(lane)
            for veh in vehicles:
                v_type = traci.vehicle.getTypeID(veh)
                if v_type in self.vehicle_types:
                    if v_type == 'emergency':
                        vehicle_counts['emergency'] += 1
                    elif v_type == 'commuter':
                        vehicle_counts['commuter'] += 1
        return vehicle_counts
    
    def adaptive_phase_control(self, intersection_id):
        # 获取当前相位信息
        current_phase = traci.trafficlight.getPhase(intersection_id)
        logic = traci.trafficlight.getAllProgramLogics(intersection_id)[0]
        phase_def = logic.phases[current_phase].state
        controlled_lanes = traci.trafficlight.getControlledLanes(intersection_id)
        
        # 识别绿灯方向
        green_lanes = []
        for lane_idx, signal in enumerate(phase_def):
            if signal == 'G' and lane_idx < len(controlled_lanes):
                green_lanes.append(controlled_lanes[lane_idx])
                traci.lane.setParameter(controlled_lanes[lane_idx], "color", "0,255,0")

        # 统计绿灯方向车流
        vehicle_count = sum(
            len(traci.lane.getLastStepVehicleIDs(lane))
            for lane in green_lanes
        )
        
        # 更新记忆
        self.extension_memory.append([vehicle_count])
        
        if len(self.extension_memory) == 3:  # 只有记忆满了才使用LSTM
            # 转换为tensor并添加batch维度
            history = torch.FloatTensor([self.extension_memory])
            
            # LSTM处理
            lstm_out, (hn, cn) = self.lstm_extension(history)
            
            # 正确获取趋势值
            trend = hn[0, -1, 0].item()  # 获取最后一个时间步的第一个隐藏单元
            
            # 决策规则
            if green_lanes and (vehicle_count >= 5 or trend > 0.5):
                remaining = traci.trafficlight.getNextSwitch(intersection_id) - traci.simulation.getTime()
                extension = min(10, 20)
                new_duration = remaining + extension
                traci.trafficlight.setPhaseDuration(intersection_id, new_duration)
                return True
        
        return False

    def reset(self):
        traci.start(self.sumoCmd)
        traci.simulationStep()
        self.time = 0
        return self.get_state()

    def get_state(self):
        # 定义每个路口的状态维度
        # 每个车道: 车辆数(1) + 停车数(1) + 紧急车辆数(1) + 通勤车辆数(1) = 4个特征
        # 加上相位信息(n_phase)
        state_dim = 4 * self.max_lanes + self.n_phase
        
        states = []
        for tl_id in traci.trafficlight.getIDList():
            obs = []
            lanes = traci.trafficlight.getControlledLanes(tl_id)
            
            # 收集车道数据
            for lane in lanes:
                # 基础数据
                obs.append(traci.lane.getLastStepVehicleNumber(lane))
                obs.append(traci.lane.getLastStepHaltingNumber(lane))
                
                # 车辆类型统计
                vehicles = traci.lane.getLastStepVehicleIDs(lane)
                emergency = sum(1 for v in vehicles if traci.vehicle.getTypeID(v) == 'emergency')
                commuter = sum(1 for v in vehicles if traci.vehicle.getTypeID(v) == 'commuter')
                obs.extend([emergency, commuter])
            
            # 补零对齐
            while len(obs) < 4 * self.max_lanes:
                obs.append(0.0)
            
            # 添加相位信息
            phase = [0] * self.n_phase
            current_phase = traci.trafficlight.getPhase(tl_id)
            if current_phase < len(phase):
                phase[current_phase] = 1
            obs.extend(phase)
            
            states.append(obs)
        
        return np.array(states, dtype=np.float32)

    def apply_action(self, actions):
        for i, tl_id in enumerate(traci.trafficlight.getIDList()):
            max_phase = self.phase_limits.get(tl_id, 0)
            safe_action = min(actions[i], max_phase)  # 确保不超限
            current_phase = traci.trafficlight.getPhase(tl_id)
            if safe_action != current_phase:
                traci.trafficlight.setPhase(tl_id, safe_action)

    def step(self, actions):
        self.apply_action(actions)
        for _ in range(self.decision_time):
            for tl_id in traci.trafficlight.getIDList():
                self.adaptive_phase_control(tl_id)
            traci.simulationStep()
            self.time += 1
        state = self.get_state()
        reward = self.get_reward()
        done = self.get_done()
        return state, reward, done

    def get_reward(self):
        rewards = np.zeros(self.n_intersections)
        for i, tl_id in enumerate(traci.trafficlight.getIDList()):
            # 1. 基础惩罚项（永远为负）
            queue_penalty = 0
            wait_penalty = 0
            lanes = traci.trafficlight.getControlledLanes(tl_id)
            
            for lane in lanes:
                # 排队车辆惩罚（按车道长度归一化）
                lane_length = traci.lane.getLength(lane)
                queue_penalty += traci.lane.getLastStepHaltingNumber(lane) / max(lane_length, 1)
                
                # 等待时间惩罚（秒）
                for veh_id in traci.lane.getLastStepVehicleIDs(lane):
                    wait_penalty += traci.vehicle.getWaitingTime(veh_id) / 10  # 除以10缩放
            
            # 2. 通行效率奖励（通过车辆数）
            throughput = traci.lane.getLastStepVehicleNumber(lane) - \
                        traci.lane.getLastStepHaltingNumber(lane)
            
            # 3. 特殊车辆处理（紧急车辆优先）
            vehicle_counts = self.detect_special_vehicles(tl_id)
            emergency_penalty = -2 * vehicle_counts['emergency']  # 紧急车辆权重加倍
            
            # 综合计算（权重可调整）
            rewards[i] = (-0.5 * queue_penalty          # 排队惩罚
                        -0.3 * wait_penalty           # 等待时间惩罚
                        +0.1 * throughput             # 通行奖励
                        + emergency_penalty)          # 特殊车辆
            
            # 确保奖励为负（表示需要优化的成本）
            rewards[i] = min(-0.01, rewards[i])  # 最小-0.01保证始终为负
        
        return rewards

    def get_done(self):
        return traci.simulation.getMinExpectedNumber() == 0

    def close(self):
        self._update_edge_congestion()
        
        # 确保仿真完全结束
        while traci.simulation.getMinExpectedNumber() > 0:
            traci.simulationStep()
        
        # 显式触发文件写入
        traci.simulation.writeMessage("Flushing output...")
        time.sleep(0.5)  # 给SUMO写入时间
        
        # 最后关闭连接
        traci.close()
        
        # 处理输出文件
        tripinfo_path = './scenario/openmap.tripinfo.xml'
        if os.path.exists(tripinfo_path):
            self._parse_tripinfo(tripinfo_path)
            self._modify_rou_file()


if __name__ == "__main__":
    env = TrafficEnv()
    state = env.reset()