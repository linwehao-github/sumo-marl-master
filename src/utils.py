import os
import shutil
import pandas as pd
import xml.etree.ElementTree as et

def get_average_travel_time(episode):
    # 原始文件路径
    src_file = "./scenario/openmap.tripinfo.xml"
    
    # 如果指定了轮次，则复制文件到 tripinfo 文件夹
    if episode is not None:
        os.makedirs("results/tripinfo", exist_ok=True)
        dst_file = f"results/tripinfo/tripinfo_episode_{episode}.xml"
        shutil.copy2(src_file, dst_file)  # 保留元数据复制
    
    # 解析XML并计算平均时间
    xtree = et.parse(src_file)
    xroot = xtree.getroot()

    rows = []
    for node in xroot:
        vehicle_id = node.attrib.get("id")
        if vehicle_id.startswith("fixed_") or node.attrib.get("vType") in ["commuter", "flexible"]:
            travel_time = float(node.attrib.get("duration"))
            rows.append({
                "id": vehicle_id,
                "travel_time": travel_time,
                "type": node.attrib.get("vType")
            })

    df = pd.DataFrame(rows)
    avg_time = df["travel_time"].mean()
    return avg_time