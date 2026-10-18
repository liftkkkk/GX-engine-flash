import os
from modelscope.hub.snapshot_download import snapshot_download

def save_model_to_local():
    # 模型信息
    model_id = 'Qwen/Qwen3-VL-Embedding-2B'
    
    # 保存目录（当前目录下新建model_folder文件夹）
    save_dir = os.path.join(os.getcwd(), 'model_folder')
    
    print(f"准备下载模型：{model_id}")
    print(f"保存目录：{save_dir}")
    
    # 确保保存目录存在
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        print(f"已创建保存目录：{save_dir}")
    
    try:
        # 下载模型
        downloaded_path = snapshot_download(
            model_id=model_id,
            cache_dir=save_dir
        )
        
        print(f"\n模型下载成功！")
        print(f"模型文件保存在：{downloaded_path}")
        
        # 列出下载的文件
        print(f"\n下载的模型文件：")
        for root, dirs, files in os.walk(downloaded_path):
            level = root.replace(downloaded_path, '').count(os.sep)
            indent = ' ' * 2 * level
            print(f"{indent}{os.path.basename(root)}/")
            subindent = ' ' * 2 * (level + 1)
            for file in files:
                print(f"{subindent}{file}")
                
    except Exception as e:
        print(f"\n模型下载失败：{e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    save_model_to_local()
