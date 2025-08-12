import xml.etree.ElementTree as ET
from collections import defaultdict
import os
import csv

def calculate_tripinfo_averages(xml_file):
    """解析单个tripinfo文件并计算指标"""
    stats = defaultdict(list)
    record_count = 0
    
    try:
        for event, elem in ET.iterparse(xml_file, events=('end',)):
            if elem.tag == 'tripinfo':
                record_count += 1
                
                # 提取基础数据
                time_loss = float(elem.get('timeLoss', 0))
                waiting_time = float(elem.get('waitingTime', 0))
                duration = float(elem.get('duration', 1))
                depart_delay = float(elem.get('departDelay', 0))
                
                # 存储原始指标
                stats['timeLoss'].append(time_loss)
                stats['waitingTime'].append(waiting_time)
                stats['waitingCount'].append(float(elem.get('waitingCount', 0)))
                stats['departDelay'].append(depart_delay)
                
                # 计算衍生指标
                stats['combined_delay_ratio'].append(
                    (time_loss + waiting_time + depart_delay) / max(duration, 1)
                )
                stats['green_utilization'].append(
                    (duration - waiting_time) / max(duration, 1)
                )
                
                elem.clear()
    
    except Exception as e:
        print(f"解析错误 {xml_file}: {e}")
        return None
    
    if record_count == 0:
        return None
        
    return {
        'episode': int(xml_file.split('_')[-1].split('.')[0]),
        'avg_timeLoss': sum(stats['timeLoss']) / record_count,
        'avg_waitingTime': sum(stats['waitingTime']) / record_count,
        'avg_departDelay': sum(stats['departDelay']) / record_count,
        'avg_waitingCount': sum(stats['waitingCount']) / record_count,
        'avg_combined_delay': sum(stats['combined_delay_ratio']) / record_count * 100,
        'avg_green_utilization': sum(stats['green_utilization']) / record_count * 100,
        'total_vehicles': record_count
    }

def process_all_episodes(output_csv):
    """处理所有episode文件并生成CSV报告"""
    header = [
        'Episode', '平均时间损失(timeLoss)', '平均等待时间(waitingTime)',
        '平均出发延误(departDelay)', '平均停车次数(waitingCount)',
        '综合延误率', '绿灯利用率', '总车辆数'
    ]
    
    with open(output_csv, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(header)
        
        for i in range(20):
            file_path = f"results/tripinfo/tripinfo_episode_{i}.xml"
            if not os.path.exists(file_path):
                print(f"文件不存在: {file_path}")
                continue
                
            print(f"\n正在处理 Episode {i}...")
            results = calculate_tripinfo_averages(file_path)
            
            if results:
                writer.writerow([
                    i+1,  # Episode编号从1开始
                    round(results['avg_timeLoss'], 2),
                    round(results['avg_waitingTime'], 2),
                    round(results['avg_departDelay'], 2),
                    round(results['avg_waitingCount'], 1),
                    f"{results['avg_combined_delay']:.1f}%",
                    f"{results['avg_green_utilization']:.1f}%",
                    results['total_vehicles']
                ])
                print(f"Episode {i} 处理完成")
            else:
                print(f"Episode {i} 数据无效")

if __name__ == "__main__":
    # 配置输出路径
    output_csv = "analysis_results.csv"
    
    # 处理所有文件
    process_all_episodes(output_csv)
    
    print(f"\n所有数据处理完成，结果已保存到: {output_csv}")