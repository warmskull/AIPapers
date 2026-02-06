import json
import os
import shutil

# 读取分类结果
base_dir = r"D:\1. 技术资料\0. 论文s\1. Paper AI相关"
with open(os.path.join(base_dir, 'papers_classification_result.json'), 'r', encoding='utf-8') as f:
    classification = json.load(f)

# 创建主题文件夹的映射简化名称
folder_mapping = {
    "Agent & Memory": "01_Agent_Memory",
    "Multimodal & Vision": "02_Multimodal_Vision",
    "DeepSeek": "03_DeepSeek",
    "RAG & Retrieval": "04_RAG_Retrieval",
    "Language Model": "05_Language_Model",
    "MoE": "06_MoE",
    "Consciousness & Neuroscience": "07_Consciousness_Neuroscience",
    "Thermal Comfort": "08_Thermal_Comfort",
    "Object Detection": "09_Object_Detection",
    "Optimization": "10_Optimization",
    "Survey & Review": "11_Survey_Review",
    "Image Generation": "12_Image_Generation",
    "Reinforcement Learning": "13_Reinforcement_Learning",
    "PINN": "14_PINN",
    "Other": "15_Other"
}

# 创建已处理的论文集合（避免重复处理）
processed_files = set()

# 为每个主题创建文件夹并移动论文
def organize_papers():
    moved_count = 0
    renamed_count = 0
    skip_count = 0
    error_count = 0
    
    for topic, papers_list in classification.items():
        # 创建主题文件夹
        topic_folder = os.path.join(base_dir, folder_mapping.get(topic, topic))
        if not os.path.exists(topic_folder):
            os.makedirs(topic_folder)
            print(f"Created folder: {topic_folder}")
        
        print(f"\nProcessing topic: {topic}")
        
        for year, filename, original_path in papers_list:
            # 检查是否已经处理过
            if filename in processed_files:
                print(f"  Skipping duplicate: {filename}")
                skip_count += 1
                continue
            
            # 跳过不存在的文件
            if not os.path.exists(original_path):
                print(f"  File not found: {original_path}")
                error_count += 1
                continue
            
            # 跳过特殊文件（已组织好的文件夹和临时文件）
            skip_folders = ['智能空调', '诺奖', 'ai+bi峰会', 'agent记忆', 'rt - detr', 'attention is all you need', 'deepseek', 'survey 综述', '旧论文归档']
            rel_path = os.path.relpath(original_path, base_dir)
            is_in_skip_folder = any(skip_folder in rel_path.lower() for skip_folder in skip_folders)
            
            if is_in_skip_folder:
                # 对于这些文件夹中的文件，只重命名添加年份前缀
                try:
                    # 计算新完整路径
                    dir_path = os.path.dirname(original_path)
                    if year != "unknown":
                        new_filename = f"{year}_{filename}"
                    else:
                        new_filename = filename
                    
                    new_path = os.path.join(dir_path, new_filename)
                    
                    # 只在文件名没有年份前缀时才重命名
                    if not filename.startswith("20"):
                        shutil.move(original_path, new_path)
                        print(f"  Renamed in folder: {filename} -> {new_filename}")
                        renamed_count += 1
                    else:
                        print(f"  Already has year prefix: {filename}")
                    skip_count += 1
                except Exception as e:
                    print(f"  Error renaming {filename}: {e}")
                    error_count += 1
                continue
            
            try:
                # 构建新文件名（添加年份前缀）
                if year != "unknown":
                    new_filename = f"{year}_{filename}"
                else:
                    new_filename = filename
                
                # 目标路径
                target_path = os.path.join(topic_folder, new_filename)
                
                # 如果目标路径已存在，添加序号
                counter = 1
                while os.path.exists(target_path):
                    base, ext = os.path.splitext(new_filename)
                    target_path = os.path.join(topic_folder, f"{base}_{counter}{ext}")
                    counter += 1
                
                # 复制文件（而不是移动，保持原文件）
                shutil.copy2(original_path, target_path)
                print(f"  Copied: {filename} -> {folder_mapping.get(topic, topic)}/{new_filename}")
                
                moved_count += 1
                if year != "unknown":
                    renamed_count += 1
                
                processed_files.add(filename)
                
            except Exception as e:
                print(f"  Error processing {filename}: {e}")
                error_count += 1
    
    print(f"\n{'='*60}")
    print(f"Organization complete:")
    print(f"  Files moved/copied: {moved_count}")
    print(f"  Files renamed with year: {renamed_count}")
    print(f"  Files skipped: {skip_count}")
    print(f"  Errors: {error_count}")

if __name__ == "__main__":
    organize_papers()
