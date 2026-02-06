import PyPDF2
import os
import re
import json

def extract_publication_date_and_topic(text):
    """从PDF文本中提取发表年份和主题信息"""
    # 提取年份 (2020-2026)
    year_match = re.search(r'\b(20[1-2][0-9])\b', text)
    year = year_match.group(1) if year_match else "unknown"
    
    # 提取关键词用于主题分类
    topic = "Other"
    lower_text = text.lower()
    
    # 主题关键词映射 (按优先级排序，更先匹配的优先级更高)
    topics = [
        ("Consciousness & Neuroscience", ["consciousness", "global workspace", "hopfield", "boltzmann"]),
        ("Thermal Comfort", ["thermal", "comfort", "hvac", "caloric", "refrigeration", "magnetocaloric"]),
        ("DeepSeek", ["deepseek", "fire-flyer"]),
        ("Agent & Memory", ["agent", "persona", "episodic memory", "memgpt", "letta", "procedural", "reflection", "react", "toolformer", "tree of thoughts", "mas_memory", "data agent"]),
        ("Multimodal & Vision", ["vision", "multimodal", "clip", "blip", "llava", "vlm", "openvla", "phi-3-vision", "qwen", "mobile", "depth anything", "dino", "egocentric"]),
        ("Object Detection", ["detr", "yolo", "detection", "real-time"]),
        ("MoE", ["moe", "mixture of experts", "deepseekmoe", "/mo"]),
        ("RAG & Retrieval", ["rag", "retrieval", "augmented generation", "cache augmented", "emg"]),
        ("Optimization", ["dpo", "preference optimization", "self-play-preference"]),
        ("Survey & Review", ["survey", "review", "overview", "personalization_survey", "cot survey", "long cot"]),
        ("Image Generation", ["stable diffusion", "diffusion", "generation", "titan", "titans"]),
        ("Reinforcement Learning", ["reinforcement", "q-learning", "multi-agent", "actuator control"]),
        ("PINN", ["pinn", "physics-informed"]),
        ("Language Model", ["language model", "llm", "large language", "gpt", "chatgpt", "gpt-4", "o1"]),
    ]
    
    for topic_name, keywords in topics:
        if any(keyword in lower_text for keyword in keywords):
            topic = topic_name
            break
    
    return year, topic

def analyze_pdf(pdf_path):
    """分析PDF文件并提取信息"""
    try:
        with open(pdf_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            text = ""
            # 只读取前5页来提取信息，提高速度
            for page in reader.pages[:5]:
                text += page.extract_text() + " "
            
            year, topic = extract_publication_date_and_topic(text)
            return year, topic
    except Exception as e:
        print(f"Error reading {pdf_path}: {e}")
        return "unknown", "Other"

def collect_all_papers(base_dir):
    """收集所有PDF论文"""
    papers = []
    for root, dirs, files in os.walk(base_dir):
        for file in files:
            if file.lower().endswith('.pdf') and not file.startswith('~'):
                pdf_path = os.path.join(root, file)
                # 计算相对路径
                rel_path = os.path.relpath(pdf_path, base_dir)
                
                # 从文件名尝试提取年份
                year_match = re.search(r'\b(20[1-2][0-9])\b', file)
                filename_year = year_match.group(1) if year_match else None
                
                print(f"Analyzing: {file}")
                year, topic = analyze_pdf(pdf_path)
                
                # 如果文件名中有年份且分析失败，使用文件名中的年份
                if year == "unknown" and filename_year:
                    year = filename_year
                
                papers.append({
                    'path': pdf_path,
                    'rel_path': rel_path,
                    'filename': file,
                    'year': year,
                    'topic': topic
                })
    
    return papers

# 设置基础目录
base_dir = r"D:\1. 技术资料\0. 论文s\1. Paper AI相关"
papers = collect_all_papers(base_dir)

# 按主题分组
topic_groups = {}
for paper in papers:
    if paper['topic'] not in topic_groups:
        topic_groups[paper['topic']] = []
    topic_groups[paper['topic']].append(paper)

# 将结果保存到JSON文件
result = {topic: sorted([(p['year'], p['filename'], p['path']) for p in papers_list], key=lambda x: x[0], reverse=True)
          for topic, papers_list in topic_groups.items()}
with open(os.path.join(base_dir, 'papers_classification_result.json'), 'w', encoding='utf-8') as f:
    json.dump(result, f, ensure_ascii=False, indent=2)

print(f"\nAnalysis complete. Results saved to papers_classification_result.json")
